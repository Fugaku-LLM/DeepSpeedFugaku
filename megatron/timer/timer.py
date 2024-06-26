# coding=utf-8
# Copyright (c) 2023, Tokyo Institute of Technology.  All rights reserved.
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

from megatron.global_vars import Timers

TIMER_PROFILE_TARGET_NAMES: list[str] = [
    "iteration",
    "train_step",
    "forward_backward",
    "forward_step",
    "get_tokenizer",
    "next",
    "gptdataset_shuffle_idx",
    "gptdataset_pre",
    "gptdataset_get",
    "frombuffer",
    "gptdataset_otherwise",
    "gptdataset_dict1",
    "gptdataset_dict2",
    "broadcast_data",
    "_build_key_size_numel_dictionaries",
    "pack",
    "broadcast",
    "unpack",
    "long",
    "contiguous_labels",
    "contiguous_tokens",
    "get_ltor_masks_and_position_ids",
    "forward-compute",
    "unwrap_model",
    "set_input_tensor",
    "forward_step_func",
    "pipeline_last_stage",
    "loss_func",
    "average_losses_across_data_parallel_group",
    "encoder",
    "attention",
    "qkv",
    "adjust_key_value",
    "raw_attention_scores",
    "baddbmm",
    "update_attention_mask",
    "scale_mask_softmax",
    "attention_dropout",
    "context_layer",
    "bmm",
    "dense",
    "row_par_lin_mm",
    "row_par_lin_allreduce",
    "mlp",
    "dense_h_to_4h",
    "activation_func",
    "save_for_backward",
    "bias_gelu",
    "dense_4h_to_h",
    "backward_step",
    "backward-compute",
    "allreduce_params",
]


def collect_active_timers(timers: Timers) -> list[str]:
    valid_target_names: list[str] = []

    for target_name in TIMER_PROFILE_TARGET_NAMES:
        if target_name in timers.timers:
            valid_target_names.append(target_name)

    return valid_target_names
