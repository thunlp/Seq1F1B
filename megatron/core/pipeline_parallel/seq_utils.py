from dataclasses import dataclass
from torch import Tensor
from typing import List

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
        attn_softmax_tflops = (
            3 * seqlen * seqlen * num_heads + 2 * seqlen * seqlen * num_heads * dim_head
        )
        attn_softmax_tflops *= scale
        attn_o_proj_tflops = 2 * seqlen * hidden_size * (dim_head * num_heads)
        attn_total = (
            attn_proj_tflops + attn_qk_tflops + attn_softmax_tflops + attn_o_proj_tflops
        )
        total = (
            embed_tflops
            + config.num_layers * (attn_total + ffn_tflops)
            + embed_proj_tflops
        )
        return total / 10**12

    def get_prefix_tflops(self, seqlen, prefix):
        attn_part = (
            seqlen * prefix * (self.dim_head * 4 + 3) * self.num_heads
            + seqlen * 8 * self.hidden_size * self.num_heads * self.dim_head
            - seqlen**2 * (4 * self.dim_head + 3) * self.num_heads / 2
        )
        ffn_tflops = self.get_ffn_tflops(seqlen)
        embed_tflops, emb_proj_tflops = self.get_emb_tflops(seqlen)
        tf = embed_tflops + self.num_layers * (attn_part + ffn_tflops) + emb_proj_tflops
        return tf / 10**12
@dataclass
class Seq1F1BInfo:
    '''
    Info for Seq1F1B pipeline parallel schedule
    '''

    micro_batch_idx: int
    span_idx_in_micro: int
    span_start: int
    span_end: int
    num_spans: int
    splits: List

    def __hash__(self):
        return hash(
            (
                self.micro_batch_idx,
                self.span_idx_in_micro,
                self.span_start,
                self.span_end,
                self.num_spans,
            )
        )
