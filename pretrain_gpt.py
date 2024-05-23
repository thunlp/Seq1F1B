# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
import torch.distributed
from megatron.core import parallel_state
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.pipeline_parallel.sp_utils import get_splits
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

def get_batch_sp():
    offset = -1
    global_data = None
    count = 0
    def get_data(*args, **kwargs):
        pipe_sp = get_args().pipe_sp_splits
        nonlocal global_data, offset
        nonlocal count
        if offset == -1 or offset+1 == pipe_sp:
            global_data = get_batch(*args,**kwargs)
            # torch.save(global_data, f"./cache/data/global_data_{count}.pt")
            count += 1
        
        offset = (offset+1) % pipe_sp 
        tokens, labels, loss_mask, attention_mask, position_ids = global_data
            
        seq_length = tokens.size(1)
        global_args = get_args()
        global_args.pipe_sp_strategy = "average" if global_args.pipe_sp_splits == 1 else global_args.pipe_sp_strategy
        if global_args.pipe_sp_strategy == "uniform_comp":
            l_s = 0
            for idx,split in enumerate(get_splits()):
                _tokens = tokens[:, l_s:l_s+split]
                _labels = labels[:, l_s:l_s+split]
                # _loss_mask = loss_mask[:, l_s:l_s+split]
                _loss_mask = loss_mask
                _loss_mask._start = l_s
                _loss_mask._end = l_s+split
                _position_ids = position_ids[:, l_s:l_s+split]
                local_data = (_tokens, _labels, _loss_mask, attention_mask, _position_ids, offset)
                l_s += split
                if idx == offset:
                    break

        elif global_args.pipe_sp_strategy == "average":
            tokens = tokens.chunk(pipe_sp, dim=1)[offset]
            labels = labels.chunk(pipe_sp, dim=1)[offset]
            loss_mask._start = seq_length // pipe_sp * offset
            loss_mask._end = seq_length // pipe_sp * (offset+1)
            # loss_mask = loss_mask.chunk(pipe_sp, dim=1)[offset]
            position_ids = position_ids.chunk(pipe_sp, dim=1)[offset]
            local_data = (tokens, labels, loss_mask, attention_mask, position_ids, offset)

        return local_data

    return get_data
get_batch_sp_func = get_batch_sp()
        
def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    start = loss_mask._start
    end = loss_mask._end
    loss_mask_p = loss_mask[:, start:end]
    loss_mask = loss_mask.contiguous()
    loss_mask = loss_mask.view(-1).float()
    args = get_args()
    if get_args().pipe_sp_splits > 1:
        loss_mask_p = loss_mask_p.contiguous().view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask_p) / loss_mask.sum() * args.pipe_sp_splits
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, offset = get_batch_sp_func(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels, micro_sp_idx=offset)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
