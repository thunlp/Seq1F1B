from sympy import symbols, Eq, solve
import torch
from megatron.core.pipeline_parallel.seq_utils import SeqTFlops


class sequence_1f1b_queue:
    def __init__(self, seq1f1b_splits=4, print=False, chunk=None, add_msg=""):
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
        self.seq1f1b_splits = seq1f1b_splits
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
        if self._idx == self.seq1f1b_splits:
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


class solver:
    def __init__(self, total_seqlen, config, causal=True):
        self.total_seqlen = total_seqlen 
        self.config = config
        self.total_tflops = config.get_seq_tflops(total_seqlen, causal)
        

    def solve_partition(self, num_splits, tp_size=1):
        res = []
        prefix = self.total_seqlen
        for i in range(1, num_splits):
            seqlen = symbols('seqlen')
            tflops = self.config.get_prefix_tflops(seqlen, prefix)
            eq = Eq(tflops, self.total_tflops / num_splits)
            sol = solve(eq, seqlen)
            sol = round_down(int(sol[0]), tp_size)
            res.insert(0, int(sol))
            prefix -= int(sol)
        res.insert(0, prefix)
        return res

def get_splits(args):
    global partitions
    if args.seq1f1b_balance_method == "average":
        return [args.seq_length // args.seq1f1b_splits] * args.seq1f1b_splits
    if args.seq1f1b_splits == 1:
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
            "vocab_size": args.padded_vocab_size,
        }
        tflops_config = SeqTFlops(**config)
        sol = solver(seqlen, tflops_config)
        args.total_tflops = sol.total_tflops
        if args.sequence_parallel:
            mod = args.tensor_model_parallel_size
        else:
            mod = 1
        partitions = sol.solve_partition(args.seq1f1b_splits, mod)
        return partitions
    else:
        return partitions


def round_down(x, tp_size):
    return x // tp_size * tp_size

if __name__ == "__main__":
    kw = {
        "num_layers": 24,
        "hidden_size": 4096,
        "ffn_size": 16384,
        "num_heads": 32,
        "dim_head": 128,
        "vocab_size": 32000,
    }
    config = SeqTFlops(**kw)
    s = solver(16384, config)
    s.solve_partition(4, 2)
