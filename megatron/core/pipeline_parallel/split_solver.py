from sympy import symbols, Eq, solve

def round_down(x, tp_size):
    return x // tp_size * tp_size
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
        

if __name__ == "__main__":
    from sp_utils import SeqTFlops
    kw = {
        "num_layers": 24,
        "hidden_size": 4096,
        "ffn_size": 16384,
        "num_heads": 32,
        "dim_head": 128,
        "vocab_size": 32000
    }
    config = SeqTFlops(**kw)
    s = solver(16384, config)
    s.solve_partition(4, 2)
        
        
        
        
    
    
    
    
