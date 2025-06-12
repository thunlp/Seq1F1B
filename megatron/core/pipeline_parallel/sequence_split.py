from sympy import symbols, Eq, solve
import torch
from megatron.core.pipeline_parallel.seq_utils import SeqTFlops
from sklearn.linear_model import LinearRegression
import numpy as np


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

    def print_log(self, msg):
        if torch.distributed.get_rank() == 3 and self.p:
            print(f"{self.info} chunk {self.c}: " + msg)

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


def get_tflops(args):
    config = {
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_size": args.ffn_hidden_size,
        "num_heads": args.num_attention_heads,
        "dim_head": args.hidden_size // args.num_attention_heads,
        "vocab_size": args.padded_vocab_size,
        "causal": True,
    }
    config = SeqTFlops(**config)
    tflops = config.get_seq_tflops(args.seq_length)
    return tflops


class FlopsSplitSolver:
    def __init__(self, total_seqlen, args):
        self.total_seqlen = total_seqlen
        self.args = args
        self.partitions = None
        self.config = SeqTFlops(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            ffn_size=args.ffn_hidden_size,
            num_heads=args.num_attention_heads,
            dim_head=args.hidden_size // args.num_attention_heads,
            vocab_size=args.padded_vocab_size,
            causal=True,
        )
        self.total_tflops = self.config.get_seq_tflops(total_seqlen)

    def get_splits(self):
        args = self.args
        if args.seq1f1b_balance_method == "average":
            return [args.seq_length // args.seq1f1b_splits] * args.seq1f1b_splits
        if args.seq1f1b_splits == 1:
            return [args.seq_length]
        if self.partitions is None:
            seqlen = args.seq_length
            args.total_tflops = self.total_tflops
            mod = args.tensor_model_parallel_size if args.sequence_parallel else 1
            partitions = self.solve_partition(args.seq1f1b_splits, mod)
            self.partitions = partitions
            return partitions
        else:
            return self.partitions

    def solve_partition(self, num_splits, tp_size=1):
        res = []
        prefix = self.total_seqlen
        for i in range(1, num_splits):
            seqlen = symbols("seqlen")
            tflops = self.config.get_prefix_tflops(seqlen, prefix)
            eq = Eq(tflops, self.total_tflops / num_splits)
            sol = solve(eq, seqlen)
            sol = round_down(int(sol[0]), tp_size)
            res.insert(0, int(sol))
            prefix -= int(sol)
        res.insert(0, prefix)
        return res


class SplitSolver:
    def __init__(self, total_seqlen, args):
        self.total_seqlen = total_seqlen
        self.base_solver = FlopsSplitSolver(total_seqlen, args)
        self.cost_table = {}

        self.alpha1 = None
        self.alpha2 = None
        self.beta = None

        self.iteration = 0
        self.mode = args.seq1f1b_balance_method
        self.fitted = False if self.mode == "linear_fit" else True
        if args.seq1f1b_linear_params is not None:
            assert self.mode == "linear_fit", "the params is used in linear fit mode"
            self.fitted = True
            args.seq1f1b_linear_warmup_step = 0
            args.seq1f1b_linear_fitting_step = 0
            self.alpha1, self.alpha2, self.beta = [
                float(i) for i in args.seq1f1b_linear_params.split(",")
            ]
        self.mod = args.tensor_model_parallel_size if args.sequence_parallel else 1
        self.num_splits = args.seq1f1b_splits
        self.fix_splits = args.seq1f1b_fix_splits
        self.split_span = None
        self.reg = LinearRegression(fit_intercept=True)
        self.latest_step = None

        self.stride = total_seqlen // (self.num_splits**2)

    def add_cost(self, seqlen, prefix, cost):
        self.cost_table[(seqlen, prefix)] = cost

    def predict_cost(self, seqlen, prefix):
        return (
            self.alpha1 * (seqlen * prefix - seqlen**2 / 2)
            + self.alpha2 * seqlen
            + self.beta
        )

    def fit(self):
        self.split_span = None
        cost_table = self.cost_table
        X = []
        y = []
        for (seqlen, prefix), cost in cost_table.items():
            X.append([seqlen * prefix - seqlen**2 / 2, seqlen])
            y.append(cost)

        self.reg.fit(X, y)
        score = self.reg.score(X, y)
        print(f"R2 score: {score:.4f}")

        self.alpha1 = self.reg.coef_[0]
        self.alpha2 = self.reg.coef_[1]
        self.beta = self.reg.intercept_
        self.fitted = True
        self.get_splits()

    def fit_span_cost(self, costs):
        avg_cost = sum(costs) / len(costs)
        overflow = []
        for i in range(self.num_splits):
            if costs[i] > avg_cost:
                overflow.append(i)
        budget = 0
        for ov in overflow:
            self.split_span[ov] -= 128
            budget += 128
        for i in range(self.num_splits):
            if i not in overflow:
                delta = budget // (self.num_splits - len(overflow))
                self.split_span[i] += delta
                budget -= delta
        self.split_span[-1] += budget

    def get_splits(self):
        if self.split_span is not None:
            return self.split_span

        assert self.fitted, "Only when solver fitted, you can get splits"
        if self.mode == "uniform_comp":
            splits = self.base_solver.get_splits()
        elif self.mode == "fix":
            splits = [int(span) for span in self.fix_splits.split(",")]
        elif self.mode == "average":
            splits = [
                self.total_seqlen // self.num_splits for i in range(self.num_splits)
            ]
        elif self.mode == "linear_fit":
            splits = []
            prefix = self.total_seqlen
            full_cost = (
                self.predict_cost(prefix, prefix) + (self.num_splits - 1) * self.beta
            )
            for i in range(1, self.num_splits):
                seqlen = symbols("seqlen")
                cost = self.predict_cost(seqlen, prefix)
                eq = Eq(cost, full_cost / self.num_splits)
                sol = solve(eq, seqlen)
                sol = round_down(int(sol[0]), self.mod)
                splits.insert(0, int(sol))
                prefix -= int(sol)
            splits.insert(0, prefix)
        for idx, i in enumerate(splits):
            assert i > 0 and i % self.mod == 0, f"splits is wrong: {splits}"
        self.split_span = splits

        return splits


def round_down(x, tp_size):
    return x // tp_size * tp_size
