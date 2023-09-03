# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron global variables."""
from __future__ import annotations

import argparse
import os
import sys
import time
import typing

import torch
from deepspeed.accelerator import get_accelerator

from megatron.tokenizer import build_tokenizer
from megatron.microbatches import ConstantNumMicroBatches, RampupBatchsizeNumMicroBatches

from .arguments import parse_args
from .microbatches import build_num_microbatches_calculator

_GLOBAL_ARGS: typing.Optional[argparse.Namespace] = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR: typing.Optional[
    typing.Union[ConstantNumMicroBatches, RampupBatchsizeNumMicroBatches]
] = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS: typing.Optional[Timers] = None


def get_args() -> argparse.Namespace:
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS  # type: ignore


def get_num_microbatches() -> int:
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()  # type: ignore


def get_current_global_batch_size() -> int:
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()  # type: ignore


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """
    Return wandb writer. It can be None so no need
    to check if it is initialized.
    """
    return _GLOBAL_WANDB_WRITER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers() -> Timers:
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS  # type: ignore


def set_global_variables(extra_args_provider=None, args_defaults={}, ignore_unknown_args=False):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""
    args = _parse_args(
        extra_args_provider=extra_args_provider,
        defaults=args_defaults,
        ignore_unknown_args=ignore_unknown_args,
    )
    _build_num_microbatches_calculator(args)
    if args.vocab_file:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)
    _set_adlr_autoresume(args)
    _set_timers()


def _parse_args(
    extra_args_provider=None, defaults={}, ignore_unknown_args=False
) -> argparse.Namespace:
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
    _GLOBAL_ARGS = parse_args(
        extra_args_provider=extra_args_provider,
        defaults=defaults,
        ignore_unknown_args=ignore_unknown_args,
    )
    return _GLOBAL_ARGS


def _build_num_microbatches_calculator(args: argparse.Namespace) -> None:
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR, "num microbatches calculator"
    )

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(args)


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

    if (
        hasattr(args, "tensorboard_dir")
        and args.tensorboard_dir
        and args.rank == (args.world_size - 1)
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter

            print("> setting tensorboard ...")
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
            )
        except ModuleNotFoundError:
            print(
                "WARNING: TensorBoard writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no TensorBoard logs will be written.",
                flush=True,
            )


def _set_wandb_writer(args: argparse.Namespace) -> None:
    """Set wandb writer."""
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER, "wandb writer")

    if (
        hasattr(args, "wandb_name")
        and (args.wandb_name or args.wandb_id)
        and args.rank == (args.world_size - 1)
    ):
        try:
            from datetime import datetime

            import wandb

            now = datetime.now()
            exp_name = args.wandb_name + "-" + str(now).replace(" ", "-")
            wandb_input = {
                "entity": "gpt-fugaku",
                "name": exp_name,
                "config": args,
                "project": "megatron-cpu-test",
            }
            if args.wandb_id is not None:
                wandb_input["id"] = args.wandb_id
                wandb_input["resume"] = "must"
            wandb.init(**wandb_input)
            _GLOBAL_WANDB_WRITER = True
            print("> wandb ...")
        except ModuleNotFoundError:
            print(
                "WARNING: wandb writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no wandb logs will be written.",
                flush=True,
            )


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, "adlr autoresume")

    if args.adlr_autoresume:
        if args.rank == 0:
            print("enabling autoresume ...", flush=True)
        sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print("ADLR autoresume is not available, exiting ...")
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers() -> None:
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


class _Timer:
    """Timer."""

    def __init__(self, name: str) -> None:
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self) -> None:
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        # torch.cuda.synchronize()
        # torch.distributed.barrier()
        self.start_time = time.time()
        self.started_ = True

    def stop(self) -> None:
        """Stop the timer."""
        assert self.started_, "timer is not started"
        # torch.cuda.synchronize()
        # torch.distributed.barrier()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self) -> None:
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset: bool = True) -> float:
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers: dict[str, _Timer] = {}

    def __call__(self, name: str) -> _Timer:
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(
        self,
        names: list[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = False,
    ) -> None:
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + "-time", value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
                print(string, flush=True)
        else:
            print(string, flush=True)

    def out(self, names: list[str], normalizer: float = 1.0, reset: bool = True) -> None:
        """
        Writes the elapsed time for each given timer name into a specific file.

        For each name in the provided list, the function calculates the
        elapsed time, optionally resets the timer, and writes the output
        into a file named 'timer.{rank:06d}' in a directory specified by
        the TIMER environment variable. If TIMER is not set, it defaults
        to a directory named 'timer'.

        The elapsed time is divided by the 'normalizer' value and is
        reported in milliseconds.

        Args:
            names (list[str]): List of timer names to report.
            normalizer (float, optional): The value by which the elapsed
                time is divided. It must be a positive number. Defaults to 1.0.
            reset (bool, optional): If True, resets each timer after
                reporting. Defaults to True.
        """
        assert normalizer > 0.0, "`normalizer` must be a positive number."
        timer_output_message: str = "time (ms)"

        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            timer_output_message += " | {}: {:.2f}".format(name, elapsed_time)

        rank: int
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        timer_dir_name = os.environ.get("TIMER", "timer")

        with open(f"{timer_dir_name}/timer.{rank:06d}", "a") as f:
            f.write(timer_output_message + "\n")
