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
