from dataclasses import dataclass
from megatron.core.pipeline_parallel.split_solver import solver
from megatron import get_args
import torch
@dataclass
class SeqTFlops:
    num_layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    dim_head: int
    vocab_size: int

    def get_ffn_tflops(self, seqlen):
        ffn_tflops = 4 * seqlen * self.hidden_size * self.ffn_size
        return ffn_tflops

    def get_emb_tflops(self, seqlen):
        embed_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        embed_proj_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        return embed_tflops, embed_proj_tflops

    def get_seq_tflops(self, seqlen, causal=False):
        scale = 0.5 if causal else 1
        config = self
        hidden_size = config.hidden_size
        num_heads = config.num_heads
        dim_head = config.dim_head
        embed_tflops, embed_proj_tflops = self.get_emb_tflops(seqlen)
        ffn_tflops = self.get_ffn_tflops(seqlen)
        attn_proj_tflops = 2 * seqlen * 3 * hidden_size * (dim_head * num_heads)
        attn_qk_tflops = 2 * seqlen * seqlen * dim_head * num_heads * scale
        attn_softmax_tflops = 3 * seqlen * seqlen * num_heads + 2 * seqlen * seqlen * num_heads * dim_head
        attn_softmax_tflops *= scale
        attn_o_proj_tflops = 2 * seqlen * hidden_size * (dim_head * num_heads)
        attn_total = attn_proj_tflops + attn_qk_tflops + attn_softmax_tflops + attn_o_proj_tflops
        total = embed_tflops + config.num_layers * (attn_total + ffn_tflops) + embed_proj_tflops
        return total / 10 ** 12

    def get_prefix_tflops(self, seqlen, prefix):
        attn_part = seqlen * prefix * (self.dim_head * 4 + 3 ) \
            * self.num_heads + seqlen * 8 * self.hidden_size \
            * self.num_heads * self.dim_head - seqlen ** 2 * (4 * self.dim_head + 3) \
            * self.num_heads / 2
        ffn_tflops = self.get_ffn_tflops(seqlen)
        embed_tflops,emb_proj_tflops = self.get_emb_tflops(seqlen)
        tf = embed_tflops + self.num_layers * (attn_part + ffn_tflops) + emb_proj_tflops
        return tf / 10 ** 12

class sp_queue:
    def __init__(self, pipe_sp_splits=4, print=False, chunk=None, add_msg=""):
        # two stage queue
        # first stage use offset to track the current queue
        # second stage use idx to track the current item
        self.queues = [[]]
        self.p = print
        self.c = chunk
        self.info = add_msg
        self._offset = 0
        self._idx = 0
        self.count = 0
        self.pipe_sp_splits = pipe_sp_splits
        self.tail_obj = None

    def __len__(self):
        return self.count

    def print_log(self,msg):
        if torch.distributed.get_rank() == 3 and self.p:
            print(f"{self.info} chunk {self.c}: "+msg)

    def append(self, obj):
        self.print_log("append inp")
        self.tail_obj = obj
        self.queues[self._offset].append(obj)
        self._idx += 1
        if self._idx == self.pipe_sp_splits:
            self.print_log("full queue , create new one")
            self.queues.append([])
            self._idx = 0
            self._offset += 1
        self.count += 1
    
    def pop(self, idx=0):
        self.print_log(f"pop head inp of first queue")
        assert idx == 0, "only pop head item"
        self.count -= 1
        if len(self.queues[0]) == 1:
            if self._offset > 0:
                self._offset -= 1
                return self.queues.pop(0)[0]
            else:
                return self.queues[0].pop(-1)
        else:
            return self.queues[0].pop(-1)

    def __getitem__(self, idx):
        self.print_log(f"get tail inp ")
        assert idx == -1
        return self.tail_obj
        

partitions = None
def get_tflops():
    args = get_args()
    config = {
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_hidden_size,
        "num_heads": args.num_attention_heads,
        "dim_head": args.hidden_size // args.num_attention_heads,
        "vocab_size": args.padded_vocab_size
    }
    config = SeqTFlops(**config)
    tflops = config.get_seq_tflops(args.seq_length, causal=True)
    return tflops

    args.total_tflops = sol.total_tflops 
def get_splits():
    global partitions
    args = get_args()
    if args.pipe_sp_strategy == "average":
        return [args.seq_length // args.pipe_sp_splits] * args.pipe_sp_splits
    if args.pipe_sp_splits == 1:
        return [args.seq_length]
    if partitions is None:
        assert args is not None
        seqlen = args.seq_length
        config = {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "ffn_size": args.ffn_hidden_size,
            "num_heads": args.num_attention_heads,
            "dim_head": args.hidden_size // args.num_attention_heads,
            "vocab_size": args.padded_vocab_size
        }
        tflops_config = SeqTFlops(**config)
        sol = solver(seqlen, tflops_config)
        if args.sequence_parallel:
            mod = args.tensor_model_parallel_size
        else:
            mod = 1
        partitions = sol.solve_partition(args.pipe_sp_splits, mod)
        return partitions
    else:
        return partitions


class sp_shape_queue:
    def __init__(self, seqlen, bs, sz, backward=False):
        self.splits = get_splits() 
        self.idx = 0 if not backward else len(self.splits) - 1
        self.shape = [[[s, bs, sz]] for s in self.splits] 
        args = get_args()
        if args.sequence_parallel:
            self.shape = [[[s // args.tensor_model_parallel_size, bs, sz]] for s in self.splits]
        self.backward = backward
    def __iter__(self):
        iter = self.shape[self.idx].__iter__()
        if not self.backward:
            self.idx = (self.idx +1) % len(self.splits)
        else:
            self.idx = (self.idx -1) % len(self.splits)
        return iter
    def get(self):
        res = self.shape[self.idx][0]
        if not self.backward:
            self.idx = (self.idx +1) % len(self.splits)
        else:
            self.idx = (self.idx -1) % len(self.splits)
        return res
