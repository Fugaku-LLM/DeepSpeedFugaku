"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

import math
import os
import types
from typing import Callable, Iterable
from enum import Enum
import functools
import itertools
from typing import List

import torch
from torch import Tensor
from deepspeed import comm as dist
from torch.nn import Module
from torch.nn import Parameter

from .linear import zero3_linear_wrap

import deepspeed
from ..utils import get_only_unique_item, see_memory_usage
from deepspeed.runtime.zero.utils import assert_ints_same_as_other_ranks
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.utils import instrument_w_nvtx, logger
from deepspeed.comm.comm import init_distributed
from deepspeed.utils.debug import (debug_param2name_id_shape,
                                   debug_param2name_id_shape_device,
                                   debug_module2name,
                                   debug_param2name_id,
                                   debug_param2name_id_shape_status)
from ..swap_tensor.partitioned_param_swapper import AsyncPartitionedParameterSwapper, PartitionedParamStatus

param_count = 0
partitioned_param_data_shape = [0]
zero_init_enabled = False


def _dist_allgather_fn(input_tensor: Tensor, output_tensor: Tensor, group=None):
    return instrument_w_nvtx(dist.allgather_fn)(output_tensor,
                                                input_tensor,
                                                group=group,
                                                async_op=True)


def print_rank_0(message, debug=False, force=False):
    rank = dist.get_rank()
    if rank == 0 and (debug or force):
        print(message)
    # other variations
    # - print for all ranks w/o interleaving
    # printflock(f"[{rank}] {message}")
    # - print to log file per rank
    # log_rank_file(rank, message)


def debug_rank0(msg: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(msg)


def is_zero_param(parameter):
    if not torch.is_tensor(parameter):
        return False
    return hasattr(parameter, 'ds_id')


def _init_external_params(module):
    if not hasattr(module, '_external_params'):
        module._external_params = {}

        def external_parameters(self):
            return self._external_params.items()

        def all_parameters(self):
            return itertools.chain(self.named_parameters(self,
                                                         recurse=False),
                                   external_parameters(self))

        module.ds_external_parameters = types.MethodType(external_parameters, module)
        module.all_parameters = types.MethodType(all_parameters, module)


def register_external_parameter(module, parameter):
    """Instruct DeepSpeed to coordinate ``parameter``'s collection and partitioning in
    the forward and backward passes of ``module``.

    This is used when a parameter is accessed outside of its owning module's
    ``forward()``. DeepSpeed must know to collect it from its partitioned
    state and when to release the memory.

    .. note::
        This is only applicable to training with ZeRO stage 3.

    Args:
        module (``torch.nn.Module``): The module that requires ``parameter`` in its forward pass.
        parameter (``torch.nn.Parameter``): The parameter to register.

    Raises:
        RuntimeError: If ``parameter`` is not of type ``torch.nn.Parameter``.


    Examples
    ========

    #. Register a weight that is used in another module's forward pass (line 6).
       Parameter ``layer1.weight`` is used by ``layer2`` (line 11).

        .. code-block:: python
            :linenos:
            :emphasize-lines: 6,11

            class ModuleZ3(torch.nn.Module):
                def __init__(self, *args):
                    super().__init__(self, *args)
                    self.layer1 = SomeLayer()
                    self.layer2 = OtherLayer()
                    deepspeed.zero.register_external_parameter(self, self.layer1.weight)

                def forward(self, input):
                    x = self.layer1(input)
                    # self.layer1.weight is required by self.layer2.forward
                    y = self.layer2(x, self.layer1.weight)
                    return y
    """
    if not isinstance(parameter, torch.nn.Parameter):
        raise RuntimeError('Parameter is not a torch.nn.Parameter')

    if not hasattr(module, '_external_params'):
        _init_external_params(module)

    key = id(parameter)
    module._external_params[key] = parameter


def unregister_external_parameter(module, parameter):
    """Reverses the effects of :meth:`register_external_parameter`.

    Args:
        module (``torch.nn.Module``): The module to affect.
        parameter (``torch.nn.Parameter``): The parameter to unregister.

    Raises:
        RuntimeError: If ``parameter`` is not of type ``torch.nn.Parameter``.
        RuntimeError: If ``parameter`` is not a registered external parameter of ``module``.
    """
    if not isinstance(parameter, torch.nn.Parameter):
        raise RuntimeError('Parameter is not a torch.nn.Parameter')

    if not hasattr(module,
                   '_external_params') or id(parameter) not in module._external_params:
        raise RuntimeError('Parameter is not a registered external parameter of module.')

    key = id(parameter)
    del module._external_params[key]


class ZeroParamType(Enum):

    # same as regular pytorch parameters
    NORMAL = 1

    # parameters are partitioned across data parallel process
    PARTITIONED = 2

    # the parameter is held with a unique process rank
    # and is not available on all other process
    REMOTE = 3


class ZeroParamStatus(Enum):
    # parameters are fully present and ready for use on all processes
    AVAILABLE = 1

    # parameters are either partitioned or remote in some or all process
    NOT_AVAILABLE = 2

    # parameters are being gathered.
    INFLIGHT = 3


_orig_torch_empty = torch.empty
_orig_torch_zeros = torch.zeros
_orig_torch_ones = torch.ones
_orig_torch_full = torch.full


def zero_wrapper_for_fp_tensor_constructor(fn: Callable,
                                           target_fp_dtype: torch.dtype) -> Callable:
    def wrapped_fn(*args, **kwargs) -> Tensor:
        if kwargs.get("device", None) is None:
            kwargs['device'] = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
        tensor: Tensor = fn(*args, **kwargs)
        if tensor.is_floating_point():
            tensor = tensor.to(target_fp_dtype)

        return tensor

    return wrapped_fn


def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:
    def new_tensor(cls, *args) -> Tensor:
        device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
        tensor = _orig_torch_empty(0, device=device).new_empty(*args)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)

        return tensor

    return new_tensor


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


@instrument_w_nvtx
def free_param(param: Parameter) -> None:
    """Free underlying storage of a parameter."""
    assert not param.ds_active_sub_modules, param.ds_summary()
    if param.data.is_cuda:
        # need to make sure that we don't free the parameter while it is still
        # being used for computation
        param.data.record_stream(torch.cuda.current_stream())
    # param.data doesn't store anything meaningful in partitioned state
    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    param.ds_status = ZeroParamStatus.NOT_AVAILABLE


reuse_buffers = False
temp_contiguous_tensor = None
empty_buffers = {}


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self,
                 enabled=True,
                 mem_efficient_linear=True,
                 ds_config=None,
                 dtype=None):
        self.mem_efficient_linear = mem_efficient_linear
        self.enabled = enabled
        self._set_dtype(ds_config, dtype)
        assert self.dtype in [torch.half, torch.bfloat16, torch.float], f"Invalid data type {self.dtype}, allowed values are [torch.half, torch.bfloat16, torch.float]"

    def __enter__(self):
        global zero_init_enabled
        if not self.enabled:
            return
        zero_init_enabled = True

        def apply_with_gather(orig_module_apply_fn: Callable) -> Callable:
            """many models make use of child modules like Linear or Embedding which
            perform their own weight initialization in their __init__ methods,
            but will then have more weight initialization in a parent module's __init__
            method that modifies weights of child modules, which is typically done
            using the Module.apply method.

            since the Init context manager partitions child modules immediately after
            they are initialized, without modifying apply we would entirely skip
            any initialization done by parent modules.

            to get around this issue, we wrap the function passed to Module.apply
            so that the applied function is applied to child modules correctly.
            """
            def get_wrapped_fn_to_apply(fn_to_apply: Callable) -> Callable:
                if hasattr(fn_to_apply, "wrapped"):
                    return fn_to_apply

                @functools.wraps(fn_to_apply)
                def wrapped_fn_to_apply(module_to_apply_fn_to: Module) -> None:
                    """gathers parameters before calling apply function. afterwards
                    parameters are broadcasted to ensure consistency across all ranks
                    then re-partitioned.

                    takes the following steps:
                    1. allgathers parameters for the current module being worked on
                    2. calls the original function
                    3. broadcasts root rank's parameters to the other ranks
                    4. re-partitions the parameters
                    """
                    if not all(
                            is_zero_param(p)
                            for p in module_to_apply_fn_to.parameters(recurse=False)):
                        raise RuntimeError(
                            f"not all parameters for {module_to_apply_fn_to.__class__.__name__}, "
                            f"were zero params, is it possible that the parameters were "
                            f"overwritten after they were initialized? "
                            f"params: {[p for p in module_to_apply_fn_to.parameters(recurse=False)]} "
                        )

                    params_to_apply_fn_to: Iterable[Parameter] = list(
                        sorted(module_to_apply_fn_to.parameters(recurse=False),
                               key=lambda p: p.ds_id))

                    for param in params_to_apply_fn_to:
                        param.all_gather()

                    fn_to_apply(module_to_apply_fn_to)

                    for param in params_to_apply_fn_to:
                        dist.broadcast(param.data, 0, group=param.ds_process_group)

                    for param in params_to_apply_fn_to:
                        param.partition(has_been_updated=True)

                wrapped_fn_to_apply.wrapped = True

                return wrapped_fn_to_apply

            @functools.wraps(orig_module_apply_fn)
            def wrapped_apply(module: Module, fn_to_apply: Callable) -> None:
                orig_module_apply_fn(module, get_wrapped_fn_to_apply(fn_to_apply))

            return wrapped_apply

        def partition_after(f):
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):

                # important logic: We want to run post_init only after child's __init__ is
                # completed, and do nothing after __init__ of any of its parents and grandparents in
                # the inheritance ancestry. This way the partitioning will need to happen only once
                # when the whole object is ready to be partitioned and not before. This is because
                # often the child module will need to tweak the weights - for example running a
                # custom weights init function. So if a parent created the weights param, the child
                # won't need to gather it in order to tweak it

                print_rank_0(f'Before initializing {module.__class__.__name__}',
                             force=False)

                is_child_module = False
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_ds_child_entered", True)

                f(module, *args, **kwargs)

                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_ds_child_entered")

                    print_rank_0(f'Running post_init for {module.__class__.__name__}',
                                 force=False)
                    self._post_init_method(module)

                print_rank_0(
                    f'After initializing followed by post init for {module.__class__.__name__}',
                    force=False)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module recursively
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            # print(f"subclass={subclass.__module__}.{subclass.__qualname__}")
            _enable_class(subclass)

        # holding onto some methods so we can put them back the way they were in __exit__
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        torch.nn.modules.module.Module.apply = apply_with_gather(
            torch.nn.modules.module.Module._old_apply)

        torch.Tensor.__new__ = get_new_tensor_fn_for_dtype(self.dtype)
        torch.empty = zero_wrapper_for_fp_tensor_constructor(_orig_torch_empty,
                                                             self.dtype)
        torch.zeros = zero_wrapper_for_fp_tensor_constructor(_orig_torch_zeros,
                                                             self.dtype)
        torch.ones = zero_wrapper_for_fp_tensor_constructor(_orig_torch_ones, self.dtype)
        torch.full = zero_wrapper_for_fp_tensor_constructor(_orig_torch_full, self.dtype)

        if self.mem_efficient_linear:
            print_rank_0(
                "nn.functional.linear has been overridden with a more memory efficient version. This will persist unless manually reset.",
                force=False)
            self.linear_bk = torch.nn.functional.linear
            torch.nn.functional.linear = zero3_linear_wrap

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        shutdown_init_context()

        if dist.get_rank() == 0:
            logger.info("finished initializing model with %.2fB parameters",
                        param_count / 1e9)

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _set_dtype(self, ds_config, dtype):
        if ds_config is not None and dtype is None:
            if ds_config.bfloat16_enabled and ds_config.fp16_enabled:
                raise RuntimeError("bfloat16 and fp16 cannot be enabled at once")

            if ds_config.bfloat16_enabled:
                self.dtype = torch.bfloat16
            elif ds_config.fp16_enabled:
                self.dtype = torch.half
            else:
                self.dtype = torch.float
        else:
            self.dtype = dtype or torch.half


def shutdown_init_context():
    global zero_init_enabled

    if not zero_init_enabled:
        return

    def _disable_class(cls):
        cls.__init__ = cls._old_init

    # Replace .__init__() for all existing subclasses of torch.nn.Module
    for subclass in get_all_subclasses(torch.nn.modules.module.Module):
        _disable_class(subclass)

    # putting methods back the way we found them
    torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass
    torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply

    torch.Tensor.__new__ = torch.Tensor.__old_new__
    torch.empty = _orig_torch_empty
    torch.zeros = _orig_torch_zeros
    torch.ones = _orig_torch_ones
    torch.full = _orig_torch_full

    # un doing it here will undo it during training
    # if self.mem_efficient_linear:
    #    torch.nn.functional.linear = self.linear_bk
    #        if self.mem_efficient_linear:
    #            torch.nn.functional.linear = self.linear_bk

    zero_init_enabled = False


class AllGatherHandle:
    def __init__(self, handle, param: Parameter) -> None:
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"expected param {param.ds_summary()} to be available")

        self.__handle = handle
        self.__param = param

    def wait(self) -> None:
        instrument_w_nvtx(self.__handle.wait)()
        self.__param.ds_status = ZeroParamStatus.AVAILABLE


class AllGatherCoalescedHandle:
    def __init__(
        self,
        allgather_handle,
        params: List[Parameter],
        partitions: List[Tensor],
        world_size: int,
    ) -> None:
        self.__allgather_handle = allgather_handle
        self.__params = params
        self.__partitions = partitions
        self.__world_size = world_size
        self.__complete = False

        for param in self.__params:
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(
                    f"expected param {param.ds_summary()} to not be available")

    @instrument_w_nvtx
    def wait(self) -> None:
        if self.__complete:
            return

        instrument_w_nvtx(self.__allgather_handle.wait)()

        # split the single tensor out into individual tensors
        param_offset = 0
        for param in self.__params:
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            partitions: List[Tensor] = []
            for rank in range(self.__world_size):
                param_start = rank * param.ds_tensor.ds_numel
                if param_start < param.ds_numel:
                    part_to_copy = self.__partitions[rank].narrow(
                        0,
                        param_offset,
                        min(param.ds_numel - param_start,
                            param.ds_tensor.ds_numel))
                    partitions.append(part_to_copy)

            param.data = instrument_w_nvtx(torch.cat)(partitions).view(param.ds_shape)
            param.ds_status = ZeroParamStatus.AVAILABLE

            for part_to_copy in partitions:
                part_to_copy.record_stream(torch.cuda.current_stream())

            param_offset += param.ds_tensor.ds_numel

        self.__complete = True


# Replaces all parameters in module with Scattered Parameters
class Init(InsertPostInitMethodToModuleSubClasses):
    param_id = 0

    def __init__(self,
                 module=None,
                 data_parallel_group=None,
                 mem_efficient_linear=True,
                 remote_device=None,
                 pin_memory=False,
                 config_dict_or_path=None,
                 config=None,
                 enabled=True,
                 dtype=None,
                 mpu=None):
        """A context to enable massive model construction for training with
        ZeRO-3. Models are automatically partitioned (or, sharded) across the
        system and converted to half precision.

        Args:
            module (``torch.nn.Module``, optional): If provided, partition the model as
                if it was constructed in the context.
            data_parallel_group (``deepspeed.comm`` process group, optional):
                The group of processes to partition among. Defaults to all processes.
            mem_efficient_linear (bool, optional): Replace
                torch.nn.functional.linear with an implementation that allows
                DeepSpeed to partition parameters. Defaults to ``True``.
            remote_device (string, optional): The initial device to store model
                weights e.g., ``cpu``, ``nvme``. Passing ``"cpu"`` will create the model in CPU
                memory. The model may still be moved to GPU based on the
                offload settings for training. Defaults to param offload device if a config is
                defined, otherwise GPU.
            pin_memory (bool, optional): Potentially increase performance by
                using pinned memory for model weights. ``remote_device`` must be
                ``"cpu"``. Defaults to pin_memory value in config, otherwise ``False``.
            config_dict_or_path (dict or ``json file``, optional): If provided, provides configuration
                for swapping fp16 params to NVMe.
            config (dict or ``json file``, optional): Deprecated, use config_dict_or_path instead.
            enabled (bool, optional): If ``False``, this context has no
                effect. Defaults to ``True``.
            dtype (``dtype``, optional): Can be used to change the data type of the parameters.
                Supported options are ``torch.half`` and ``torch.float``. Defaults to ``None``
            mpu (``object``, optional): A model parallelism unit object that implements get_{model,data}_parallel_{rank,group,world_size}.

        This context accelerates model initialization and enables models that
        are too large to allocate in their entirety in CPU memory. It has the
        following effects:

        #. allocates tensors to either GPU or CPU memory or NVMe
        #. converts floating point tensors to half precision
        #. immediately partitions tensors among the group of data-parallel devices
        #. (*optional*) replaces ``torch.nn.functional.linear`` with a more
           memory-efficient implementation

        These modifications allow for models that exceed the size of local CPU/GPU
        memory/NVMe, but fit within the total NVMe capacity (*i.e.*, aggregate CPU
        or GPU memory or NVMe) across all nodes. Consider initializing a model with one
        trillion parameters, whose weights occupy two terabytes (TB) in half
        precision. The initial CPU allocation in full precision requires 4TB of
        memory *per process*, and so a system with 8 GPUs per node would need 32TB of
        CPU memory due to data-parallel redundancies. Instead, by immediately
        partitioning tensors we remove the redundancies. The result is that
        regardless of the number of GPUs, we still only require the original 4TB. This
        allows for a linear increase in model size with the aggregate system memory.
        For example, if a node has 1TB of memory and 8 GPUs, we could fit a trillion
        parameter model with 4 nodes and 32 GPUs.

        Important: If the fp16 weights of the model can't fit onto a single GPU memory
        this feature must be used.

        .. note::
            Initializes ``deepspeed.comm`` if it has not already been done so.
            See :meth:`deepspeed.init_distributed` for more information.

        .. note::
            Can also be used as a decorator:

            .. code-block:: python

                @deepspeed.zero.Init()
                def get_model():
                    return MyLargeModel()

        .. note::
            Only applicable to training with ZeRO-3.

        Examples
        --------

        #. Allocate a model and partition it among all processes:

            .. code-block:: python

                with deepspeed.zero.Init():
                    model = MyLargeModel()


        #. Allocate a model in pinned CPU memory and partition it among a subgroup of processes:

            .. code-block:: python

                with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                         remote_device="cpu",
                                         pin_memory=True):
                    model = MyLargeModel()


        #. Partition an already-allocated model in CPU memory:

            .. code-block:: python

                model = deepspeed.zero.Init(module=model)
        """
        if config is not None:
            config_dict_or_path = config
            logger.warning(
                f'zero.Init: the `config` argument is deprecated. Please use `config_dict_or_path` instead.'
            )

        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(
            config_dict_or_path,
            mpu) if config_dict_or_path is not None else None
        super().__init__(enabled=enabled,
                         mem_efficient_linear=mem_efficient_linear,
                         ds_config=_ds_config,
                         dtype=dtype)
        if not dist.is_initialized():
            init_distributed()
            assert dist.is_initialized(), "Parameters cannot be scattered without initializing deepspeed.comm"
        if data_parallel_group is None:
            self.ds_process_group = dist.get_world_group()
        else:
            self.ds_process_group = data_parallel_group

        self.rank = dist.get_rank(group=self.ds_process_group)
        self.world_size = dist.get_world_size(group=self.ds_process_group)

        # Local device is the device where the parameters are consumed, must be default device.
        # It is the device where parameters are fully instantiated using allgather
        #self.local_device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
        self.local_device = torch.device('cpu')
        #torch.cuda.set_device(self.local_device)

        if _ds_config is not None and _ds_config.zero_config.offload_param is not None:
            remote_device = _ds_config.zero_config.offload_param.device
            pin_memory = _ds_config.zero_config.offload_param.pin_memory

        self._validate_remote_device(remote_device, _ds_config)

        # Remote device is the device where parameter partitions are stored
        # It can be same as local_device or it could be CPU or NVMe.
        self.remote_device = self.local_device if remote_device in [
            None,
            OffloadDeviceEnum.none
        ] else remote_device
        self.pin_memory = pin_memory if (
            self.remote_device in [OffloadDeviceEnum.cpu,
                                   OffloadDeviceEnum.nvme]) else False

        # Enable fp16 param swapping to NVMe
        if self.remote_device == OffloadDeviceEnum.nvme:
            self.param_swapper = AsyncPartitionedParameterSwapper(_ds_config, self.dtype)
        else:
            self.param_swapper = None

        # If we are provided an already-allocated module to prepare.
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            self._convert_to_zero_parameters(module.parameters(recurse=True))

        self.use_all_gather_base = False
        if dist.has_allgather_base():
            self.use_all_gather_base = True
        else:
            logger.info(
                f"_all_gather_base API is not available in torch {torch.__version__}")

    def _convert_to_zero_parameters(self, param_list):
        for param in param_list:
            if is_zero_param(param):
                continue
            self._convert_to_deepspeed_param(param)
            param.partition()

    def _validate_remote_device(self, remote_device, ds_config):
        if ds_config is not None:
            if remote_device in [None, OffloadDeviceEnum.cpu]:
                if ds_config.zero_config.offload_param is not None:
                    offload_param_device = ds_config.zero_config.offload_param.device
                    assert offload_param_device != OffloadDeviceEnum.nvme, \
                        f"'device' in DeepSpeed Config cannot be {offload_param_device} if remote device is {remote_device}."

            if remote_device == OffloadDeviceEnum.nvme:
                assert ds_config.zero_config.offload_param is not None, \
                f'"offload_param" must be defined in DeepSpeed Config if remote device is {OffloadDeviceEnum.nvme}.'

                assert ds_config.zero_config.offload_param.nvme_path is not None, \
                f'"nvme_path" in DeepSpeed Config cannot be None if remote device is {OffloadDeviceEnum.nvme}'

    def _post_init_method(self, module):
        #see_memory_usage(f"Before converting parmas in {module.__class__.__name__}", force=False)
        print_rank_0(f'Converting Params in {module.__class__.__name__}', force=False)
        see_memory_usage(
            f"Before converting and partitioning parmas in {module.__class__.__name__}",
            force=False)

        global param_count
        for name, param in module.named_parameters(recurse=False):
            param_count += param.numel()
            if not is_zero_param(param):
                self._convert_to_deepspeed_param(param)
                print_rank_0(
                    f"Partitioning param {debug_param2name_id_shape(param)} module={debug_module2name(module)}"
                )

                if param.is_cuda:
                    dist.broadcast(param, 0, self.ds_process_group)
                else:
                    if dist.get_rank() == 0:
                        logger.warn(f"param `{name}` in {module.__class__.__name__} "
                                    f"not on GPU so was not broadcasted from rank 0")

                param.partition()
        see_memory_usage(
            f"Param count {param_count}. After converting and partitioning parmas in {module.__class__.__name__}",
            force=False)

    def _convert_to_deepspeed_param(self, param):

        # Partitioned, Normal, Remote
        param.ds_param_type = ZeroParamType.PARTITIONED

        # Replicated vs Partitioned vs Inflight
        param.ds_status = ZeroParamStatus.AVAILABLE

        # Stores the shape of the original tensor
        param.ds_shape = param.shape

        # Stores the number of elements in the original parameter without padding
        param.ds_numel = param.numel()

        # Stores the partitioned copy of the tensor
        param.ds_tensor = None

        # Keeps track of how many active sub-modules need this param at any given point in time
        param.ds_active_sub_modules = set()

        # If this flag is true, then the parameters are replicated throughput training
        # And only partitioned before the step
        param.ds_persist = False

        param.is_external_param = False

        # The group that the parameter is scattered across.
        param.ds_process_group = self.ds_process_group

        # This is set to the Async Param swapper if remote device is nvme
        # else this is set to None
        param.nvme_swapper = self.param_swapper

        # DeepSpeed Param ID
        param.ds_id = Init.param_id
        Init.param_id += 1

        def all_gather(param_list=None, async_op=False, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            return self._all_gather(param_list, async_op=async_op, hierarchy=hierarchy)

        @instrument_w_nvtx
        def all_gather_coalesced(params: Iterable[Parameter],
                                 safe_mode: bool = False) -> AllGatherCoalescedHandle:

            # fetches from nvme if the partition is not available and in nvme
            self._ensure_availability_of_partitioned_params(params)

            for param in params:
                if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                    raise RuntimeError(param.ds_summary())
                param.ds_status = ZeroParamStatus.INFLIGHT

            # ensure that each rank has params in same order. the allgather
            # is done by flattening the parameter list into a single tensor that
            # can be allgathered in a single call - this means that if each rank
            # gives a list of the same parameters in a different order we will
            # silently get incorrect parameter values, and have very difficult
            # to debug correctness issues.
            params = sorted(params, key=lambda p: p.ds_id)

            debug_rank0(f"-allgather_coalesced: {[p.ds_id for p in params]}")

            if safe_mode:
                # ensure that same list (with same ordering) of parameters are
                # being allgathered across all ranks, otherwise could mix
                # data between tensors.
                assert_ints_same_as_other_ranks([p.ds_id for p in params])
                # ensure that tensors from each rank agree on the same ds_numel
                # otherwise could mix data between tensors.
                assert_ints_same_as_other_ranks([p.ds_tensor.ds_numel for p in params])

            if len(params) == 1:
                # have an opportunity to avoid some intermediate memory allocations
                param, = params
                param_buffer = torch.empty(
                    math.ceil(param.ds_numel / self.world_size) * self.world_size,
                    dtype=param.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                handle = _dist_allgather_fn(
                    param.ds_tensor.to(torch.cuda.current_device()),
                    param_buffer,
                    self.ds_process_group)
                param.data = param_buffer.narrow(0,
                                                 0,
                                                 param.ds_numel).view(param.ds_shape).to(
                                                     param.device)
                return AllGatherHandle(handle, param)
            else:
                partition_sz = sum(p.ds_tensor.ds_numel for p in params)
                flat_tensor = torch.empty(partition_sz * self.world_size,
                                          dtype=get_only_unique_item(p.dtype
                                                                     for p in params),
                                          device=torch.cuda.current_device(),
                                          requires_grad=False)
                partitions: List[Parameter] = []
                for i in range(self.world_size):
                    partitions.append(
                        flat_tensor.narrow(0,
                                           partition_sz * i,
                                           partition_sz))

                instrument_w_nvtx(torch.cat)(
                    [p.ds_tensor.to(torch.cuda.current_device()) for p in params],
                    out=partitions[self.rank])
                handle = _dist_allgather_fn(partitions[self.rank],
                                            flat_tensor,
                                            self.ds_process_group)

                return AllGatherCoalescedHandle(
                    allgather_handle=handle,
                    params=params,
                    partitions=partitions,
                    world_size=self.world_size,
                )

        def partition(param_list=None, hierarchy=0, has_been_updated=False):
            cls = param
            print_rank_0(
                f"{'--'*hierarchy}----Partitioning param {debug_param2name_id_shape_device(cls)}"
            )
            if param_list is None:
                param_list = [cls]
            self._partition(param_list, has_been_updated=has_been_updated)

        def reduce_gradients_at_owner(param_list=None, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            print_rank_0(
                f"{'--'*hierarchy}----Reducing Gradients for param with ids {[param.ds_id for param in param_list]} to owner"
            )
            self._reduce_scatter_gradients(param_list)

        def partition_gradients(param_list=None,
                                partition_buffers=None,
                                hierarchy=0,
                                accumulate=False):
            cls = param
            print_rank_0(
                f"{'--'*hierarchy}----Partitioning param gradient with id {debug_param2name_id_shape_device(cls)}"
            )
            if param_list is None:
                param_list = [cls]
                if isinstance(partition_buffers, torch.Tensor):
                    partition_buffers = [partition_buffers]

            self._partition_gradients(param_list,
                                      partition_buffers=partition_buffers,
                                      accumulate=accumulate)

        def aligned_size():
            return self._aligned_size(param)

        def padding_size():
            return self._padding_size(param)

        def partition_numel():
            return self._partition_numel(param)

        def item_override():
            param.all_gather()
            return param._orig_item()

        def ds_summary(slf: torch.Tensor, use_debug_name: bool = False) -> dict:
            return {
                "id": debug_param2name_id(slf) if use_debug_name else slf.ds_id,
                "status": slf.ds_status.name,
                "numel": slf.numel(),
                "ds_numel": slf.ds_numel,
                "shape": tuple(slf.shape),
                "ds_shape": tuple(slf.ds_shape),
                "requires_grad": slf.requires_grad,
                "grad_shape": tuple(slf.grad.shape) if slf.grad is not None else None,
                "persist": slf.ds_persist,
                "active_sub_modules": slf.ds_active_sub_modules,
            }

        def convert_to_zero_parameters(param_list):
            self._convert_to_zero_parameters(param_list)

        def allgather_before(func: Callable) -> Callable:
            def wrapped(*args, **kwargs):
                param.all_gather()
                return func(*args, **kwargs)

            return wrapped

        # Collectives for gathering and partitioning parameters
        param.all_gather = all_gather
        param.all_gather_coalesced = all_gather_coalesced
        param.partition = partition

        # Collective for averaging gradients
        param.reduce_gradients_at_owner = reduce_gradients_at_owner
        param.partition_gradients = partition_gradients

        # Partitioning size utilities
        param.aligned_size = aligned_size
        param.padding_size = padding_size
        param.partition_numel = partition_numel
        param.ds_summary = types.MethodType(ds_summary, param)

        param.item = allgather_before(param.item)

        param.convert_to_zero_parameters = convert_to_zero_parameters

    def _aligned_size(self, param):
        return param.ds_numel + self._padding_size(param)

    def _padding_size(self, param):
        remainder = param.ds_numel % self.world_size
        return (self.world_size - remainder) if remainder else 0

    def _partition_numel(self, param):
        return param.ds_tensor.ds_numel

    def _ensure_availability_of_partitioned_params(self, params):
        swap_in_list = []
        swap_in_flight = []
        for param in params:
            if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                assert param.ds_tensor.final_location == OffloadDeviceEnum.nvme and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_list.append(param)
            if param.ds_tensor.status == PartitionedParamStatus.INFLIGHT:
                assert param.ds_tensor.final_location == OffloadDeviceEnum.nvme and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_flight.append(param)
        if len(swap_in_list) > 0:
            swap_in_list[0].nvme_swapper.swap_in(swap_in_list, async_op=False)
        elif len(swap_in_flight) > 0:
            swap_in_flight[0].nvme_swapper.synchronize_reads()

    @instrument_w_nvtx
    def _all_gather(self, param_list, async_op=False, hierarchy=None):

        # fetches from nvme if the partition is not available and in nvme
        self._ensure_availability_of_partitioned_params(param_list)

        handles = []
        all_gather_list = []
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if async_op:
                    handle = self._allgather_param(param,
                                                   async_op=async_op,
                                                   hierarchy=hierarchy)
                    param.ds_status = ZeroParamStatus.INFLIGHT  # if async_op else ZeroParamStatus.AVAILABLE
                    handles.append(handle)
                else:
                    all_gather_list.append(param)

        if not async_op:
            if len(param_list) == 1:
                ret_value = self._allgather_params(all_gather_list, hierarchy=hierarchy)
            else:
                ret_value = self._allgather_params_coalesced(all_gather_list, hierarchy)

            for param in all_gather_list:
                param.ds_status = ZeroParamStatus.AVAILABLE
            return ret_value

        return handles

    def _partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            #print_rank_0(f"Before Partitioning Param {param.ds_id}")
            # self._param_status(param)
            self._partition_param(param, has_been_updated=has_been_updated)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            # if param.ds_tensor is not None:
            #    assert id(param.data) == id(param.ds_tensor.data), \
            #    "After the parameters are initially partitioned, make sure we are not recreating the partition."
            #print_rank_0(f"After Partitioning Param {param.ds_id}")
            # self._param_status(param)

    @instrument_w_nvtx
    def _partition_param(self, param, buffer=None, has_been_updated=False):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot partition a param in flight"

        global reuse_buffers
        #print_rank_0(f"Param id {param.ds_id} status is {param.ds_status}")
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            print_rank_0(
                f"Partitioning param id {param.ds_id} reuse buffers {reuse_buffers}",
                force=False)
            # if reuse_buffers and False:
            #     numel = buffer.numel()
            #     buffer = param.data.view(-1)
            #     print_rank_0(
            #         "Returning buffer for param {param.ds_id} with numel {param.ds_numel} to empty buffers",
            #         force=False)
            #     if numel in empty_buffers:
            #         empty_buffers[numel].append(buffer)

            # if deepspeed.comm.get_rank():
            #    print(f"Releasing {param.data.numel()}")
            if param.ds_tensor is not None and not has_been_updated:

                #param.data = param.ds_tensor.data

                see_memory_usage(
                    f'Before partitioning param {param.ds_id} {param.shape}',
                    force=False)
                # param.data does not store anything meaningful in partitioned state
                free_param(param)
                see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}',
                                 force=False)

                if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                    print_rank_0(
                        f"Param {param.ds_id} partition released since it exists in nvme",
                        force=False)
                    param.nvme_swapper.remove_partition_and_release_buffers([param])

                return

            tensor_size = self._aligned_size(param)
            partition_size = tensor_size // self.world_size

            if param.ds_tensor is None:
                final_location = None
                if self.remote_device == OffloadDeviceEnum.nvme and self.param_swapper.swappable_tensor(
                        numel=partition_size):
                    final_location = OffloadDeviceEnum.nvme
                    buffer = self.param_swapper.get_buffer(param, partition_size)
                    partitioned_tensor = torch.empty(0,
                                                     dtype=param.dtype,
                                                     device=buffer.device)
                    partitioned_tensor.data = buffer.data
                    print_rank_0(
                        f"ID {param.ds_id} Initializing partition for the first time for nvme offload."
                    )

                else:
                    partitioned_tensor = torch.empty(
                        partition_size,
                        dtype=param.dtype,
                        device=OffloadDeviceEnum.cpu if self.remote_device
                        == OffloadDeviceEnum.nvme else self.remote_device)
                    if self.pin_memory:
                        partitioned_tensor = partitioned_tensor.pin_memory()

                partitioned_tensor.requires_grad = False
                param.ds_tensor = partitioned_tensor
                param.ds_tensor.ds_numel = partition_size
                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = final_location

            start = partition_size * self.rank
            end = start + partition_size

            one_dim_param = param.contiguous().view(-1)

            if start < param.ds_numel and end <= param.ds_numel:
                src_tensor = one_dim_param.narrow(0, start, partition_size)

                param.ds_tensor.copy_(src_tensor)
                #partitioned_tensor = src_tensor.clone().detach().to(self.remote_device)

            else:
                # partitioned_tensor = torch.zeros(partition_size,
                #                                  dtype=param.dtype,
                #                                  device=self.remote_device )

                if start < param.ds_numel:
                    elements_to_copy = param.ds_numel - start
                    param.ds_tensor.narrow(0,
                                           0,
                                           elements_to_copy).copy_(
                                               one_dim_param.narrow(
                                                   0,
                                                   start,
                                                   elements_to_copy))

            #print(f"Remote device {self.remote_device}")

            #param.ds_tensor = partitioned_tensor

            #param.data = param.ds_tensor.data

            # param.data does not store anything meaningful in partitioned state

            see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}',
                             force=False)
            free_param(param)
            see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}',
                             force=False)

            if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                self.param_swapper.swap_out_and_release([param])
                print_rank_0(
                    f"ID {param.ds_id} Offloaded to nvme offload and buffers released.")
                see_memory_usage(
                    f"ID {param.ds_id} Offloaded to nvme offload and buffers released.",
                    force=False)

            print_rank_0(
                f"ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}"
            )

    def _param_status(self, param):
        if param.ds_tensor is not None:
            print_rank_0(
                f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned numel {param.ds_tensor.numel()}, data numel {param.data.numel()}"
            )
        else:
            print_rank_0(
                f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned ds_tensor {param.ds_tensor}, data numel {param.data.numel()}"
            )

    def _allgather_param(self, param, async_op=False, hierarchy=0):

        partition_size = param.ds_tensor.ds_numel

        tensor_size = partition_size * self.world_size
        aligned_param_size = self._aligned_size(param)
        assert tensor_size == aligned_param_size, f'param id {param.ds_id} aligned size {aligned_param_size} does not match tensor size {tensor_size}'

        print_rank_0(
            f"{'--'* hierarchy}---- Before allocating allgather param {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )

        see_memory_usage(
            f'Before allocate allgather param {debug_param2name_id_shape_status(param)} partition_size={partition_size} ',
            force=False)
        flat_tensor = torch.zeros(aligned_param_size,
                                  dtype=param.dtype,
                                  device=param.device).view(-1)
        see_memory_usage(
            f'After allocate allgather param {debug_param2name_id_shape_status(param)} {aligned_param_size} {partition_size} ',
            force=False)

        torch.cuda.synchronize()

        print_rank_0(
            f"{'--'* hierarchy}----allgather param with {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )
        #        if not flat_tensor.numel() > 100000:
        #            replicated_tensor = flat_tensor.narrow(0,
        #                                                   0,
        #                                                   param.ds_numel).view(param.ds_shape)
        #            param.data = replicated_tensor.data
        #            return None
        if self.use_all_gather_base:
            # try the _all_gather_base on PyTorch master branch
            handle = dist.all_gather_base(flat_tensor,
                                          param.ds_tensor.cuda(),
                                          group=self.ds_process_group,
                                          async_op=async_op)
        else:
            partitions = []
            for i in range(self.world_size):
                partitions.append(
                    flat_tensor.narrow(0,
                                       partition_size * i,
                                       partition_size))

                if i == dist.get_rank(group=self.ds_process_group):
                    partitions[i].data.copy_(param.ds_tensor.data, non_blocking=True)

            handle = dist.all_gather(partitions,
                                     partitions[self.rank],
                                     group=self.ds_process_group,
                                     async_op=async_op)

        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        return handle

    def _allgather_params_coalesced(self, param_list, hierarchy=0):
        """ blocking call
        avoid explicit memory copy in _allgather_params
        """
        if len(param_list) == 0:
            return
        # collect local tensors and partition sizes
        partition_sizes = []
        local_tensors = []
        for param in param_list:
            partition_sizes.append(param.ds_tensor.ds_numel)
            local_tensors.append(param.ds_tensor.cuda())

        # allocate memory for allgather params
        allgather_params = []
        for psize in partition_sizes:
            tensor_size = psize * self.world_size
            flat_tensor = torch.empty(tensor_size,
                                      dtype=param_list[0].dtype,
                                      device=self.local_device).view(-1)
            flat_tensor.requires_grad = False
            allgather_params.append(flat_tensor)

        # launch
        launch_handles = []
        # backend = get_backend(self.ds_process_group)
        # with _batch_p2p_manager(backend):
        for param_idx, param in enumerate(param_list):
            input_tensor = local_tensors[param_idx].view(-1)

            if self.use_all_gather_base:
                # try the _all_gather_base from Pytorch master
                h = dist.all_gather_base(allgather_params[param_idx],
                                         input_tensor,
                                         group=self.ds_process_group,
                                         async_op=True)
            else:
                output_list = []
                for i in range(self.world_size):
                    psize = partition_sizes[param_idx]
                    partition = allgather_params[param_idx].narrow(0, i * psize, psize)
                    output_list.append(partition)
                    if not partition.is_cuda:
                        logger.warning(
                            f'param {param_idx}, partition {i} is not on CUDA, partition shape {partition.size()}'
                        )

                # back to old all_gather function signature
                h = dist.all_gather(output_list,
                                    input_tensor,
                                    group=self.ds_process_group,
                                    async_op=True)
            launch_handles.append(h)

        # Wait ensures the operation is enqueued, but not necessarily complete.
        launch_handles[-1].wait()

        # assign to param.data (not copy)
        for i, param in enumerate(param_list):
            gathered_tensor = allgather_params[i]
            param.data = gathered_tensor.narrow(0,
                                                0,
                                                param.ds_numel).view(param.ds_shape).data

        # guarantee the communication to be completed
        torch.cuda.synchronize()

        return None

    def _allgather_params(self, param_list, hierarchy=0):
        if len(param_list) == 0:
            return

        partition_size = sum([param.ds_tensor.ds_numel for param in param_list])

        tensor_size = partition_size * self.world_size
        flat_tensor = torch.empty(tensor_size,
                                  dtype=param_list[0].dtype,
                                  device=self.local_device)
        flat_tensor.requres_grad = False
        partitions = []
        for i in range(self.world_size):
            start = partition_size * i

            partitions.append(flat_tensor.narrow(0, start, partition_size))

            if i == self.rank:
                offset = 0
                for param in param_list:
                    param_numel = param.ds_tensor.ds_numel

                    partitions[i].narrow(0,
                                         offset,
                                         param_numel).copy_(param.ds_tensor.data)

                    offset += param_numel

        dist.all_gather(partitions,
                        partitions[self.rank],
                        group=self.ds_process_group,
                        async_op=False)
        param_offset = 0

        for param in param_list:
            param_partition_size = param.ds_tensor.ds_numel
            param_size = param.ds_numel
            replicated_tensor = torch.empty(param.ds_shape,
                                            dtype=param.dtype,
                                            device=self.local_device)

            for i in range(self.world_size):

                start = i * partition_size

                param_start = i * param_partition_size

                if param_start < param_size:
                    numel_to_copy = min(param_size - param_start, param_partition_size)

                    part_to_copy = partitions[i].narrow(0, param_offset, numel_to_copy)

                    replicated_tensor.view(-1).narrow(0,
                                                      param_start,
                                                      numel_to_copy).copy_(part_to_copy)
            #param_offset += param.data.numel()
            param_offset += param.ds_tensor.ds_numel

            param.data = replicated_tensor.data

        return None

    def _reduce_scatter_gradients(self, param_list):
        #print_rank_0([param.grad for param in param_list])
        #assert any([param.grad is None for param in param_list]), "None gradients cannot be reduce scattered"

        handles_and_reduced_partitions = []
        for param in param_list:
            assert param.grad.numel(
            ) == param.ds_numel, f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter gradients whose size is not same as the params"

            handles_and_reduced_partitions.append(self._reduce_scatter_gradient(param))

        for param, (handle, reduced_partition) in zip(param_list, handles_and_reduced_partitions):
            if handle is not None:
                handle.wait()

            # some ranks may have partitions that are padded to go beyond the grad size.
            # For these ranks the output of reduce scatter is a separate buffer and needs
            # to be copied in
            partition_size = param.ds_tensor.ds_numel
            start = self.rank * partition_size
            end = start + partition_size
            #print_rank_0("REduce scatter was executed for praam {param.ds_id}")
            if start < param.ds_numel and end > param.ds_numel:
                elements = param.ds_numel - start
                param.grad.view(-1).narrow(0,
                                           start,
                                           elements).copy_(
                                               reduced_partition.narrow(0,
                                                                        0,
                                                                        elements))

    def _reduce_scatter_gradient(self, param):

        partition_size = param.ds_tensor.ds_numel
        #output = torch.empty(partition_size, dtype=param.dtype, device=param.device)

        total_size = partition_size * self.world_size
        input_list = []

        for i in range(self.world_size):

            start = i * partition_size
            end = start + partition_size

            #print("before reduce scatter gradients")
            if start < param.ds_numel and end <= param.ds_numel:
                input = param.grad.view(-1).narrow(0, start, partition_size)
            else:
                input = torch.zeros(partition_size,
                                    dtype=param.dtype,
                                    device=param.device)

                if start < param.ds_numel:
                    elements = param.ds_numel - start
                    input.narrow(0,
                                 0,
                                 elements).copy_(
                                     param.grad.view(-1).narrow(0,
                                                                start,
                                                                elements))
            #print("after reduce scatter gradients")
            input_list.append(input)

        rank = dist.get_rank(group=self.ds_process_group)
        handle = dist.reduce_scatter(input_list[rank],
                                     input_list,
                                     group=self.ds_process_group,
                                     async_op=True)

        return handle, input_list[rank]

    def _partition_gradients(self, param_list, partition_buffers=None, accumulate=False):
        if partition_buffers is None:
            partition_buffers = [None] * len(param_list)

        for param, partition_buffer in zip(param_list, partition_buffers):
            self._partition_gradient(param,
                                     partition_buffer=partition_buffer,
                                     accumulate=accumulate)

    def _partition_gradient(self, param, partition_buffer=None, accumulate=False):
        #import pdb;pdb.set_trace()
        # param.grad=None
        # param.grad.test()
        print_rank_0(
            f"Partitioning param {param.ds_id} gradient of size {param.grad.numel()} type {param.grad.dtype} part_size {param.ds_tensor.ds_numel}"
        )
        see_memory_usage("Before partitioning gradients", force=False)
        partition_size = param.ds_tensor.ds_numel

        if partition_buffer is None:
            assert not accumulate, "No buffer to accumulate to"
            partition_buffer = torch.zeros(partition_size,
                                           dtype=param.dtype,
                                           device=param.device)
        else:
            assert partition_buffer.numel(
            ) >= partition_size, f"The partition buffer size {partition_buffer.numel()} should match the size of param.ds_tensor {partition_size}"

        rank = dist.get_rank(group=self.ds_process_group)
        start = partition_size * rank
        end = start + partition_size

        dest_tensor_full_buffer = partition_buffer.view(-1).narrow(0, 0, partition_size)

        #print("before partition gradients")
        if start < param.ds_numel:
            elements = min(param.ds_numel - start, partition_size)

            dest_tensor = dest_tensor_full_buffer.narrow(0, 0, elements)
            src_tensor = param.grad.view(-1).narrow(0, start, elements)

            # just copy the grad partition to the buffer
            if not accumulate:
                dest_tensor.copy_(src_tensor)

            # if source and destination are on same device,
            # add to the provided buffer
            elif src_tensor.device == dest_tensor.device:
                dest_tensor.add_(src_tensor)

            # if source and destination are on different device, copy first to src
            # then add and move back to the destination. This seems to run faster
            # when src is gpu and dest is cpu
            # adding directly to cpu is very slow
            else:
                acc_tensor = torch.empty(src_tensor.numel(),
                                         dtype=param.dtype,
                                         device=param.device)

                acc_tensor.copy_(dest_tensor)
                acc_tensor.add_(src_tensor)
                dest_tensor.copy_(acc_tensor)

            # partition_buffer.view(-1).narrow(
            #     0,
            #     0,
            #     elements).copy_(param.grad.view(-1).narrow(0,
            #                                             start,
            #                                             elements))

        #print("after partition gradients")
        param.grad.data = dest_tensor_full_buffer.data
        see_memory_usage("After partitioning gradients", force=False)


class GatheredParameters:
    def __init__(self, params, modifier_rank=None, fwd_module=None, enabled=True):
        """A context that collects parameters that were partitioned via a
        :class:`deepspeed.zero.Init` context. The parameters are partitioned
        again upon exit.

        Args:
            params (``torch.nn.Parameter``): A single parameter, a list, or a tuple of parameters to collect.
                It's assumed that all parameters are zero params.
            modifier_rank (int, optional): If specified, this rank's parameter will be
                broadcasted on exit from the context. This argument is required if ``params`` are
                modified, so that all processes have a consistent view of the data. Defaults
                to ``None``.
            fwd_module (``torch.nn.Module``, optional): If specified, ``params`` will be
                registered as external parameters of ``fwd_module``. See :meth:`deepspeed.zero.register_external_parameter`.
            enabled (bool, optional): If ``False``, this context is a no-op. Defaults to ``True``.

        Important: Make sure to use ``modifier_rank`` that is not ``None`` (e.g., ``modifier_rank=0``)
        if you need the GPU memory allocated by gather to be released upon exit from the context manager.

        Examples
        ========

        #. Allocate a partitioned module, initialize its weight on rank 0, and update all
           processes.

            .. code-block:: python

                with deepspeed.zero.Init():
                    linear = torch.nn.Linear(1000,1000)

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        linear.weight.zero_()

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        linear.weight.zero_()

        #. Collect a partitioned weight to pass to another module during
           training. The parameter will be registered as an external parameter
           and made available during the backward pass.

            .. code-block:: python
                :emphasize-lines: 6

                def forward(self, input):
                    x = self.layer1(input)

                    # self.layer1.weight is required by self.layer2.forward
                    with deepspeed.zero.GatheredParameters(self.layer1.weight,
                                                           fwd_module=self):
                        y = self.layer2(x, self.layer1.weight)
                    return y


        #. Pretrained model loading

            .. code-block:: python

                with deepspeed.zero.Init():
                    model = MyModel()

                state_dict = torch.load(model_path, map_location="cpu")

                def load(module: nn.Module, prefix=""):
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                        if deepspeed.comm.get_rank() == 0:
                            module._load_from_state_dict(state_dict, prefix)

                    for name, child in module._modules.items():
                        if child is not None:
                            load(child, prefix + name + ".")

                load(model, prefix="")

        If this approach is not used, then the full model will first be copied to each GPU. For models
        bigger than the memory of a single GPU, this method is required.
        """

        self.enabled = enabled
        if not enabled:
            return

        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        # enable if at least one is zero-param, otherwise a noop
        if not any(is_zero_param(p) for p in params):
            self.enabled = False
            return

        self.params = [p for p in params if hasattr(p, "ds_id")]
        self.src_rank = None
        if modifier_rank is not None:
            if self.params[0].ds_process_group == dist.get_world_group():
                self.src_rank = modifier_rank
            else:
                # A group was specified; convert DP rank to global rank
                self.src_rank = dist.get_global_rank(self.params[0].ds_process_group,
                                                     modifier_rank)
        self.fwd_module = fwd_module
        if self.fwd_module is not None:
            # is a no-op if already registered
            for p in self.params:
                register_external_parameter(self.fwd_module, p)

    def __enter__(self):
        if not self.enabled:
            return
        self.params[0].all_gather(param_list=self.params)

    def __exit__(self, *exc):
        if not self.enabled:
            return
        if self.src_rank is None:
            return

        handles = [
            dist.broadcast(p,
                           self.src_rank,
                           group=p.ds_process_group,
                           async_op=True) for p in self.params
        ]
        for h in handles:
            h.wait()
        self.params[0].partition(param_list=self.params, has_been_updated=True)
