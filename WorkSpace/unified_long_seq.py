"""
Unified Long-Sequence Modeling Framework: Baseline / LONGER / STCA / HyFormer
============================================================================

A single framework that subsumes the vanilla dense baseline and three
long-sequence recommendation architectures as special cases of one
configurable 5-block layer:

  Block 1  QueryGen     -- generate N_q query tokens from NS / target / SS
  Block 2  SeqEnc       -- produce per-layer (keys, values) from the sequence SS
  Block 3  QueryDecode  -- cross-attention: query attends to sequence K/V
  Block 4  HighOrder    -- high-order interaction on the decoded query
  Block 5  QueryFuse    -- fuse the layer output back into the query

Degradation rule ("a block being empty degrades into another method"):
  * All blocks active (HighOrder = Mixer)                        -> HyFormer
  * HighOrder = SelfAttn, SeqEnc = TokenMerge                     -> LONGER
  * HighOrder = None (empty), QueryGen = target, SeqEnc = id      -> STCA
  * All blocks at "base" (full self-attn inside HighOrder)        -> Baseline

Each block is an abstract Base module; each method provides concrete subclasses
selected by the Configuration. An "empty" block is realized by a Null (identity)
module, which is exactly how the degradation takes effect.

Inputs (per sample):
  target                 in R^{N_t x d}    target item representation token(s)
  non_sequence_features  in R^{N_ns x d}   user/context/cross feature tokens
  sequence_features      in R^{N_ss x d}   user behavior tokens (N_ss is large)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Primitive neural operators
# ============================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    causal=True applies a standard lower-triangular mask (query i attends only
    to keys j <= i), used by self-attention blocks (LongerHighOrder /
    BaselineHighOrder) for temporal ordering. The cross-attention decode is
    NON-causal: the sequence is isolated as KV and never acts as a query, so it
    structurally cannot see the target -- the premise for KV-Cache / RLB reuse,
    achieved without an explicit mask.
    """

    def __init__(self, dim: int, num_heads: int = 4, causal: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        # Input projections (one per query / key / value) and the output projection.
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, keys=None, values=None):
        # If keys/values are omitted, this is self-attention on `query`.
        if keys is None:
            keys = query
            values = query
        batch_size, num_query_tokens, _ = query.shape
        # Project and reshape to [batch, num_heads, num_tokens, head_dim].
        query = self.query_proj(query).reshape(batch_size, num_query_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(keys).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(values).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention scores: [batch, num_heads, num_query, num_key].
        attention_scores = (query @ keys.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if self.causal:
            # Mask position i from seeing keys j > i (strictly upper triangle).
            causal_mask = torch.triu(
                torch.ones(num_query_tokens, keys.size(-2), device=query.device), 1
            ).bool()
            attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))
        # Softmax over keys, then take the weighted sum of values.
        attention_output = attention_scores.softmax(-1) @ values
        # Merge heads back to [batch, num_query, dim] and apply the output projection.
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, num_query_tokens, -1)
        return self.output_proj(attention_output)


class FeedForwardNetwork(nn.Module):
    """4x-expansion feed-forward network with GELU activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, 4 * dim)     # expand 4x
        self.down_proj = nn.Linear(4 * dim, dim)   # project back to dim

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class SwiGLU(nn.Module):
    """SwiGLU activation block (HyFormer lightweight sequence encoding, tier iii)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, 4 * dim)
        self.up_proj = nn.Linear(dim, 4 * dim)
        self.down_proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        # SwiGLU(x) = ( silu(x @ W_gate) * (x @ W_up) ) @ W_down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TokenMixingMixer(nn.Module):
    """RankMixer-style token mixing with residual + LayerNorm (Query Boosting).

    Mixes information across the token axis via a learned (num_tokens x
    num_tokens) matrix, applied transposed so that all `dim` channels share the
    same token-mixing weights.
    """

    def __init__(self, dim: int, num_tokens: int):
        super().__init__()
        self.token_mixing_proj = nn.Linear(num_tokens, num_tokens)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [batch, num_tokens, dim] -> mix across tokens -> back to [batch, num_tokens, dim].
        mixed = self.token_mixing_proj(x.transpose(1, 2)).transpose(1, 2)
        return self.layer_norm(x + mixed)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (self-attention + FFN with residuals)."""

    def __init__(self, dim: int, causal: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, causal=causal)
        self.ffn = FeedForwardNetwork(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.self_attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ============================================================
# Base block interfaces (abstract)
# ============================================================

class BaseQueryGen(nn.Module):
    """Block 1: produce N_q query tokens from NS, target, SS and previous query."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, non_sequence_features, target, sequence_features, previous_query):
        raise NotImplementedError


class BaseSeqEnc(nn.Module):
    """Block 2: produce per-layer (keys, values) from SS. (None, None) when empty."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sequence_features):
        raise NotImplementedError


class BaseQueryDecode(nn.Module):
    """Block 3: cross-attention query -> (keys, values). Pass-through when keys is None."""

    def __init__(self, dim: int, causal: bool = False):
        super().__init__()
        self.dim = dim
        self.causal = causal

    def forward(self, query, keys, values):
        raise NotImplementedError


class BaseHighOrder(nn.Module):
    """Block 4: high-order interaction on the decoded query (+NS/target).

    An empty (identity) implementation degrades the stack to STCA.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, decoded_query, context_tokens):
        raise NotImplementedError


class BaseQueryFuse(nn.Module):
    """Block 5: fuse the layer output back into the query for the next layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, enhanced_query, previous_query):
        raise NotImplementedError


# ============================================================
# Null (identity) modules -- realize "empty block" degradation
# ============================================================

class NullQueryGen(BaseQueryGen):
    """Reuse the previous layer's query (LONGER self-attention layers)."""

    def forward(self, non_sequence_features, target, sequence_features, previous_query):
        return previous_query


class NullSeqEnc(BaseSeqEnc):
    """Do not access the sequence; returns (None, None)."""

    def forward(self, sequence_features):
        return None, None


class NullQueryDecode(BaseQueryDecode):
    """Skip decoding; pass the query through unchanged (used when keys is None)."""

    def forward(self, query, keys, values):
        return query


class NullHighOrder(BaseHighOrder):
    """No high-order interaction -> degrades the stack to STCA (pure cross stacking)."""

    def forward(self, decoded_query, context_tokens):
        return decoded_query


class ResidualFuse(BaseQueryFuse):
    """query_next = enhanced_query + previous_query (Baseline / LONGER / HyFormer default)."""

    def forward(self, enhanced_query, previous_query):
        if previous_query is None:
            return enhanced_query
        return enhanced_query + previous_query


# ============================================================
# Baseline concrete modules (vanilla dense self-attention reference)
# ============================================================

class BaselineQueryGen(BaseQueryGen):
    """Baseline: query = the full token set [target; NS; SS].

    A vanilla dense self-attention layer treats every token as a query, so
    'query generation' is just the concatenation of all input tokens.
    """

    def forward(self, non_sequence_features, target, sequence_features, previous_query):
        # Full set order: [target; non_sequence; sequence] -> [batch, N_t+N_ns+N_ss, dim].
        return torch.cat([target, non_sequence_features, sequence_features], dim=1)


class BaselineHighOrder(BaseHighOrder):
    """Baseline: full bidirectional self-attention + FFN over the whole token set.

    This is the single dense transformer block whose attention cost is O(L^2) in
    the sequence length -- the cost that LONGER / STCA / HyFormer each avoid.
    `context_tokens` is unused because the full set (incl. target and NS) is
    already the decoded query.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.self_attn_block = TransformerBlock(dim, causal=False)

    def forward(self, decoded_query, context_tokens):
        return self.self_attn_block(decoded_query)


# ============================================================
# LONGER concrete modules
# ============================================================

class LongerQueryGen(BaseQueryGen):
    """[global tokens (target+CLS+UID...); recent-k sampled sequence tokens].

    Output length N_q = N_t + k.
    """

    def __init__(self, dim: int, k_recent_sample: int = 100):
        super().__init__(dim)
        self.k_recent_sample = k_recent_sample
        self.global_proj = nn.Linear(dim, dim)

    def forward(self, non_sequence_features, target, sequence_features, previous_query):
        global_tokens = self.global_proj(target)                              # [batch, N_t, dim]
        # Most-recent k sequence tokens (recent-k sampling).
        sampled_sequence = sequence_features[:, -self.k_recent_sample:, :]   # [batch, k, dim]
        # Query = [global tokens; sampled sequence tokens] -> [batch, N_t + k, dim].
        return torch.cat([global_tokens, sampled_sequence], dim=1)


class LongerSeqEnc(BaseSeqEnc):
    """Token Merge (sum-pool adjacent K tokens) + InnerTrans -> L -> L/K."""

    def __init__(self, dim: int, token_merge_group_size: int = 4):
        super().__init__(dim)
        self.token_merge_group_size = token_merge_group_size
        self.inner_transformer = TransformerBlock(dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, sequence_features):
        batch_size, seq_len, dim = sequence_features.shape
        group_size = self.token_merge_group_size
        # Sum-pool every K adjacent tokens: [batch, L, dim] -> [batch, L/K, dim].
        merged = sequence_features.reshape(batch_size, seq_len // group_size, group_size, dim).sum(dim=2)
        # InnerTrans: a transformer over the merged groups.
        encoded = self.inner_transformer(merged)                             # [batch, L/K, dim]
        return self.key_proj(encoded), self.value_proj(encoded)


class CrossAttnDecode(BaseQueryDecode):
    """Shared cross-attention decode used by LONGER and HyFormer.

    Query (N_q) attends to the sequence (keys, values); + FFN. Skipped when
    keys is None.
    """

    def __init__(self, dim: int, causal: bool = False):
        super().__init__(dim, causal)
        self.cross_attn = MultiHeadAttention(dim, causal=causal)
        self.ffn = FeedForwardNetwork(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, query, keys, values):
        if keys is None:
            return query                                   # no sequence -> skip decode
        query = query + self.cross_attn(self.attn_norm(query), keys, values)
        query = query + self.ffn(self.ffn_norm(query))
        return query


class LongerHighOrder(BaseHighOrder):
    """Self-causal attention on the compressed query (layers 2..N of LONGER).

    Operates only on `decoded_query` (the m+k compressed tokens); `context_tokens`
    is unused because global tokens (incl. target) are already part of the query.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.self_attn_block = TransformerBlock(dim, causal=True)

    def forward(self, decoded_query, context_tokens):
        return self.self_attn_block(decoded_query)


# ============================================================
# STCA concrete modules
# ============================================================

class STCAQueryGen(BaseQueryGen):
    """Target only, N_q = N_t (the single-query premise of strict O(L))."""

    def forward(self, non_sequence_features, target, sequence_features, previous_query):
        return target


class STCASeqEnc(BaseSeqEnc):
    """Identity: SS unchanged, only per-layer key/value projections differ.

    Because the sequence is never re-encoded, history tokens stay target-agnostic,
    which (with the structural query/KV isolation) enables RLB reuse.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, sequence_features):
        return self.key_proj(sequence_features), self.value_proj(sequence_features)


class STCAQueryDecode(BaseQueryDecode):
    """Single-query cross-attention (query = target, N_q = 1).

    The paper reorders computation to avoid materializing X*W_K / X*W_V; this is
    a memory win that does not change the operation count.
    """

    def __init__(self, dim: int, causal: bool = False):
        super().__init__(dim, causal)
        self.cross_attn = MultiHeadAttention(dim, causal=causal)
        self.ffn = FeedForwardNetwork(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, query, keys, values):
        if keys is None:
            return query
        query = query + self.cross_attn(self.attn_norm(query), keys, values)
        query = query + self.ffn(self.ffn_norm(query))
        return query


class STCAQueryFuse(BaseQueryFuse):
    """Target-conditioned query fusion: concat[enhanced; previous] -> Linear -> + previous."""

    def __init__(self, dim: int):
        super().__init__(dim)
        self.fusion_proj = nn.Linear(2 * dim, dim)

    def forward(self, enhanced_query, previous_query):
        if previous_query is None:
            return enhanced_query
        # Concat current and previous query, project, then add a residual.
        fused = self.fusion_proj(torch.cat([enhanced_query, previous_query], dim=-1))
        return fused + previous_query


# ============================================================
# HyFormer concrete modules
# ============================================================

class HyFormerQueryGen(BaseQueryGen):
    """N independent FFNs fire from GlobalInfo = [NS; MeanPool(SS)] -> N queries."""

    def __init__(self, dim: int, n_global_queries: int = 3):
        super().__init__(dim)
        self.n_global_queries = n_global_queries
        # One independent linear head per global query token.
        self.query_heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_global_queries)])

    def forward(self, non_sequence_features, target, sequence_features, previous_query):
        # GlobalInfo = [NS; MeanPool(SS)] -> mean-pool to a single [batch, dim] vector.
        sequence_summary = sequence_features.mean(dim=1, keepdim=True)             # [batch, 1, dim]
        global_info = torch.cat([non_sequence_features, sequence_summary], dim=1).mean(dim=1)  # [batch, dim]
        # Each head fires one differentiated global query -> stack -> [batch, N, dim].
        global_queries = torch.stack([head(global_info) for head in self.query_heads], dim=1)
        return global_queries


class HyFormerSeqEnc(BaseSeqEnc):
    """Three capacity/efficiency tiers for sequence K/V encoding.

    tier='full'        : Full Transformer self-attention (highest capacity)
    tier='longer_style': S_short as Q, SS as KV cross-attention (default deploy)
    tier='swiglu'      : no-attention SwiGLU map (lowest latency)
    """

    def __init__(self, dim: int, tier: str = "longer_style", short_query_length: int = 256):
        super().__init__(dim)
        self.tier = tier
        self.short_query_length = short_query_length
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        if tier == "full":
            self.full_transformer = TransformerBlock(dim)
        elif tier == "longer_style":
            self.short_cross_attn = MultiHeadAttention(dim)
        elif tier == "swiglu":
            self.swiglu = SwiGLU(dim)
        else:
            raise ValueError(f"unknown HyFormerSeqEnc tier: {tier}")

    def forward(self, sequence_features):
        if self.tier == "full":
            encoded = self.full_transformer(sequence_features)
        elif self.tier == "longer_style":
            # S_short (first L_s tokens) as Q against the full SS as KV.
            short_query = sequence_features[:, : self.short_query_length, :]
            encoded = self.short_cross_attn(short_query, sequence_features, sequence_features)
        else:  # swiglu
            encoded = self.swiglu(sequence_features)
        return self.key_proj(encoded), self.value_proj(encoded)


class HyFormerHighOrder(BaseHighOrder):
    """MLP-Mixer Query Boosting on [decoded query; NS; target] + per-token FFN.

    The token-mixing Linear(num_tokens, num_tokens) is built lazily on the first
    forward, because num_tokens = N_q + N_ns + N_t is only known at runtime from
    the actual input shapes.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self.per_token_ffn = FeedForwardNetwork(dim)
        # The token-mixing Linear(num_tokens, num_tokens) is built lazily on
        # the first forward, because num_tokens = N_q + N_ns + N_t is only
        # known at runtime once the actual input shapes are available.
        self.token_mixer: Optional[TokenMixingMixer] = None   # built lazily

    def _maybe_build_token_mixer(self, num_tokens: int, reference_tensor):
        """Build (or rebuild) the token mixer once the token count is known.

        The mixer is rebuilt if `num_tokens` changes across forwards so the
        module stays correct under varying batch-level token counts.
        """
        mixer_is_stale = (
            self.token_mixer is None
            or self.token_mixer.token_mixing_proj.in_features != num_tokens
        )
        if mixer_is_stale:
            self.token_mixer = TokenMixingMixer(self.dim, num_tokens)
            self.token_mixer = self.token_mixer.to(reference_tensor.device).to(reference_tensor.dtype)

    def forward(self, decoded_query, context_tokens):
        # Boosting is applied over [decoded query; context (NS; target)] so
        # the original non-sequence signal has a direct path back into the
        # mixer (the key reason HyFormer beats MTGR's self-attn degradation).
        boosting_input_tokens = torch.cat([decoded_query, context_tokens], dim=1)
        num_boosting_tokens = boosting_input_tokens.size(1)
        self._maybe_build_token_mixer(num_boosting_tokens, boosting_input_tokens)
        mixed_tokens = self.token_mixer(boosting_input_tokens)
        boosted_query = self.per_token_ffn(mixed_tokens)
        # Return only the decoded-query positions for the next block.
        return boosted_query[:, : decoded_query.size(1)]


# ============================================================
# Configuration
# ============================================================

@dataclass
class Configuration:
    """Per-layer configuration.

    Each of the five module-selection fields picks a concrete implementation for
    that block. The placeholder "base" resolves to the Baseline dense
    self-attention implementation, so a bare Configuration() describes one
    Baseline layer; the three long-sequence methods override these fields.
    """

    # ---------------- Common Hyperparameters ----------------
    # 5-block module selection (each names the concrete implementation to build).
    # The placeholder "base" resolves to the Baseline dense self-attention
    # implementation; the three long-sequence methods override these fields.
    query_gen: str = "base"        # "base" | "longer" | "stca" | "hyformer" | "reuse"
    # "base"        -> Baseline (KV kept inside HighOrder, SeqEnc is a no-op)
    # "stca"        -> STCA identity K/V projection (target-agnostic, RLB-friendly)
    # "longer"      -> LONGER Token-Merge + InnerTrans (L -> L/K)
    # "hyformer"    -> HyFormer tier selected by `hyformer_seq_enc_tier` below
    # "full" | "longer_style" | "swiglu"  -> HyFormer tier used directly
    # "none"        -> empty (LONGER self-attention layers skip the sequence)
    seq_enc: str = "base"
    query_decode: str = "base"     # "base" | "cross" | "stca_single"
    high_order: str = "base"       # "base" | "self_attn" | "mixer" | "none"
    query_fuse: str = "base"       # "base" | "residual" | "stca"
    # Cross-attention decode is NON-causal here: the sequence is isolated as KV
    # and never acts as a query, so it structurally cannot see the target (the
    # KV-Cache / RLB premise holds without an explicit mask). Causal masking is
    # only used by LongerHighOrder's self-attention for temporal ordering.
    causal: bool = False
    # Number of stacked unified layers and the hidden dimension.
    n_layers: int = 8
    dim: int = 64

    # ---------------- LONGER Hyperparameters ----------------
    # Number of most-recent sequence tokens sampled into the query (recent-k).
    ss_k_recent_sample: int = 100
    # Token-Merge group size K: adjacent K sequence tokens are sum-pooled -> L/K.
    token_merge_group_size: int = 4

    # ---------------- STCA Hyperparameters ----------------
    # STCA is fully specified by the common module selections
    # (query_gen="stca", seq_enc="stca", query_decode="stca_single",
    #  high_order="none", query_fuse="stca"); it exposes no extra hyperparameters.

    # ---------------- HyFormer Hyperparameters ----------------
    # Number of independent global query tokens fired by the FFN heads.
    n_global_queries: int = 3
    # Short-query length L_s for the "longer_style" SeqEnc tier (S_short as Q).
    short_query_length: int = 256
    # SeqEnc tier for HyFormer: "full" | "longer_style" | "swiglu".
    hyformer_seq_enc_tier: str = "longer_style"


# ============================================================
# Module factory functions (Configuration -> concrete block modules)
# ============================================================

def build_query_gen(config: Configuration) -> BaseQueryGen:
    """Build the query-generation block (Block 1) from the Configuration.

    The placeholder "base" yields the Baseline full-token-set query; "reuse"
    yields the NullQueryGen that reuses the previous layer's query, which is
    how LONGER self-attention layers avoid regenerating the query.
    """
    if config.query_gen == "base":
        return BaselineQueryGen(config.dim)
    if config.query_gen == "longer":
        return LongerQueryGen(config.dim, config.ss_k_recent_sample)
    if config.query_gen == "stca":
        return STCAQueryGen(config.dim)
    if config.query_gen == "hyformer":
        return HyFormerQueryGen(config.dim, config.n_global_queries)
    if config.query_gen == "reuse":
        return NullQueryGen(config.dim)
    raise ValueError(f"unknown query_gen mode: {config.query_gen}")


def build_seq_enc(config: Configuration) -> BaseSeqEnc:
    """Build the sequence-encoding block (Block 2) from the Configuration.

    The returned module produces per-layer (keys, values) from SS. An empty
    selection ("none" / "base") returns a NullSeqEnc whose forward yields
    (None, None), which makes the downstream QueryDecode skip itself -- this
    is exactly how LONGER self-attention layers and the Baseline drop the
    cross-attention path.
    """
    # Empty / baseline selections -> NullSeqEnc (no sequence access).
    if config.seq_enc in (None, "none", "base"):
        return NullSeqEnc(config.dim)
    # STCA: identity K/V projection, target-agnostic (RLB-friendly).
    if config.seq_enc == "stca":
        return STCASeqEnc(config.dim)
    # LONGER: Token-Merge (sum-pool adjacent K) + InnerTrans -> L/K.
    if config.seq_enc == "longer":
        return LongerSeqEnc(config.dim, config.token_merge_group_size)
    # HyFormer: "hyformer" defers the tier to `hyformer_seq_enc_tier`; the
    # tier values ("full" | "longer_style" | "swiglu") can also be written
    # directly as `seq_enc` for convenience.
    if config.seq_enc == "hyformer":
        hyformer_tier = config.hyformer_seq_enc_tier
        return HyFormerSeqEnc(
            config.dim,
            tier=hyformer_tier,
            short_query_length=config.short_query_length,
        )
    if config.seq_enc in ("full", "longer_style", "swiglu"):
        return HyFormerSeqEnc(
            config.dim,
            tier=config.seq_enc,
            short_query_length=config.short_query_length,
        )
    raise ValueError(f"unknown seq_enc mode: {config.seq_enc}")


def build_query_decode(config: Configuration) -> BaseQueryDecode:
    """Build the cross-attention decode block (Block 3) from the Configuration.

    "base" returns a NullQueryDecode whose forward is a pass-through; combined
    with a NullSeqEnc (which yields keys=None) this skips the decode entirely
    -- the path the Baseline and LONGER self-attention layers take.
    """
    if config.query_decode == "stca_single":
        return STCAQueryDecode(config.dim, config.causal)
    if config.query_decode == "base":
        return NullQueryDecode(config.dim, config.causal)   # baseline skips (keys is None)
    return CrossAttnDecode(config.dim, config.causal)        # "cross" / default


def build_high_order(config: Configuration) -> BaseHighOrder:
    """Build the high-order interaction block (Block 4) from the Configuration.

    An empty selection ("none") returns a NullHighOrder (identity) -- this is
    the single switch that degrades the stack to STCA (pure cross stacking),
    because the cross-attention decode still runs but no self / mixer
    interaction is applied on top.
    """
    if config.high_order == "base":
        return BaselineHighOrder(config.dim)        # full self-attn over the query set
    if config.high_order in (None, "none"):
        return NullHighOrder(config.dim)            # empty -> STCA degradation
    if config.high_order == "self_attn":
        return LongerHighOrder(config.dim)
    if config.high_order == "mixer":
        return HyFormerHighOrder(config.dim)
    raise ValueError(f"unknown high_order mode: {config.high_order}")


def build_query_fuse(config: Configuration) -> BaseQueryFuse:
    """Build the query-fusion block (Block 5) from the Configuration.

    STCA uses a learned concat-and-project fusion; every other method uses a
    plain residual add. "base" and "residual" are aliases for ResidualFuse.
    """
    if config.query_fuse == "stca":
        return STCAQueryFuse(config.dim)
    return ResidualFuse(config.dim)                  # "base" / "residual" / default


# ============================================================
# Unified layer and model
# ============================================================

class UnifiedLayer(nn.Module):
    """One unified layer wiring the five blocks together.

    Forward flow (per layer l):
      1) QueryGen    : produce query Q^{(l)} from NS, target, SS, prev query
      2) SeqEnc      : produce (K_l, V_l) from SS                       (may be None)
      3) QueryDecode : cross-attention Q -> (K, V) -> decoded query    (skipped if K is None)
      4) HighOrder   : self-attn / mixer / identity on decoded query   (identity -> STCA)
      5) QueryFuse   : fuse the enhanced query back for the next layer

    The `previous_query` argument threads Q across layers (None for the
    first layer); NullQueryGen reuses it as-is, which is how LONGER's
    self-attention layers keep the query fixed after the cross-compression.
    """

    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        # Build each block from the Configuration; an "empty" selection
        # produces a Null module whose forward is an identity / no-op.
        self.query_gen = build_query_gen(config)
        self.seq_enc = build_seq_enc(config)
        self.query_decode = build_query_decode(config)
        self.high_order = build_high_order(config)
        self.query_fuse = build_query_fuse(config)

    def forward(self, non_sequence_features, target, sequence_features, previous_query):
        # (1) Generate the query tokens (or reuse the previous layer's).
        query = self.query_gen(non_sequence_features, target, sequence_features, previous_query)
        # (2) Encode the sequence into (keys, values); (None, None) when empty.
        keys, values = self.seq_enc(sequence_features)
        # (3) Cross-attention: query attends to the sequence; the decode
        #     module skips itself when keys is None (Baseline / LONGER-self).
        decoded_query = self.query_decode(query, keys, values)
        # (4) High-order interaction (self-attn / mixer / identity).
        #     Context = [NS; target] is fed to HighOrder so HyFormer's mixer
        #     can give NS a direct path; other HighOrder variants ignore it.
        context_tokens = torch.cat([non_sequence_features, target], dim=1)
        enhanced_query = self.high_order(decoded_query, context_tokens)
        # (5) Fuse the enhanced query back into the query for the next layer.
        return self.query_fuse(enhanced_query, query)


class UnifiedModel(nn.Module):
    """Stacks unified layers and applies the prediction head.

    The stack is a list of per-layer Configurations: homogeneous stacks
    (Baseline / STCA / HyFormer) repeat one config; LONGER is heterogeneous
    (one cross-compression config followed by N self-attention configs).
    """

    def __init__(self, layer_configs):
        super().__init__()
        self.layer_configs = layer_configs
        self.layers = nn.ModuleList([UnifiedLayer(config) for config in layer_configs])
        # Hidden dim is taken from the first layer's config (all layers share it).
        model_dim = layer_configs[0].dim
        # Two-layer MLP prediction head -> scalar click/score probability.
        self.prediction_head = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, 1)
        )

    def forward(self, target, non_sequence_features, sequence_features):
        # Thread the query through the layer stack; None on the first layer
        # lets QueryGen ignore `previous_query` (or NullQueryGen reuse it).
        query = None
        for layer in self.layers:
            query = layer(non_sequence_features, target, sequence_features, query)
        # Read out the target-position slots (first N_t tokens) for prediction.
        target_query = query[:, : target.size(1)]
        return torch.sigmoid(self.prediction_head(target_query))


# ============================================================
# Four methods = different configurations of the same framework
# ============================================================

def build_baseline(dim: int = 64, n_layers: int = 4) -> UnifiedModel:
    """Baseline: vanilla dense self-attention over the full token set (O(L^2)).

    Every block stays at "base": QueryGen concatenates [target; NS; SS] into
    the query, SeqEnc and QueryDecode are no-ops (keys=None -> skip), and
    HighOrder is a full bidirectional self-attention + FFN over the whole set.
    This is the O(L^2) reference that the three long-sequence methods avoid.
    """
    config = Configuration(dim=dim, n_layers=n_layers)   # all blocks stay at "base"
    return UnifiedModel([config] * n_layers)


def build_hyformer(dim: int = 64, n_layers: int = 4,
                   n_global_queries: int = 3, short_query_length: int = 256,
                   hyformer_seq_enc_tier: str = "longer_style") -> UnifiedModel:
    """HyFormer: all blocks active, Decode <-> Boosting alternated for L layers.

    All five blocks are active: QueryGen fires N global queries from
    [NS; MeanPool(SS)], SeqEnc encodes the sequence (tier configurable), the
    cross-attention decode produces the decoded query, HighOrder applies the
    MLP-Mixer Query Boosting on [decoded query; NS; target], and QueryFuse
    adds the residual. The SeqEnc tier is configurable via
    `hyformer_seq_enc_tier`:
      "longer_style" (default deploy) | "full" | "swiglu".
    """
    config = Configuration(
        query_gen="hyformer",
        seq_enc="hyformer",                       # defers to hyformer_seq_enc_tier
        query_decode="cross",
        high_order="mixer",
        query_fuse="residual",
        n_layers=n_layers,
        dim=dim,
        n_global_queries=n_global_queries,
        short_query_length=short_query_length,
        hyformer_seq_enc_tier=hyformer_seq_enc_tier,
    )
    return UnifiedModel([config] * n_layers)


def build_longer(dim: int = 64, n_self_layers: int = 3,
                 ss_k_recent_sample: int = 100, token_merge_group_size: int = 4) -> UnifiedModel:
    """LONGER: 1 cross-compression layer + N self-attention layers (KV-Cache serving).

    Layer 1 (cross-compression): QueryGen samples [global; recent-k] tokens,
    SeqEnc does Token-Merge (sum-pool K) + InnerTrans, QueryDecode attends the
    compressed sequence, HighOrder is empty (no self-interaction yet).
    Layers 2..N (self-attention): QueryGen reuses the previous query, SeqEnc
    is empty (no sequence access), QueryDecode is skipped, HighOrder is a
    causal self-attention. KV-Cache serving applies to these self layers.
    """
    # Layer 1: cross-compression layer -- query attends the compressed sequence.
    cross_layer_config = Configuration(
        query_gen="longer",
        seq_enc="longer",
        query_decode="cross",
        high_order="none",                        # no self-interaction in layer 1
        query_fuse="residual",
        dim=dim,
        ss_k_recent_sample=ss_k_recent_sample,
        token_merge_group_size=token_merge_group_size,
    )
    # Layers 2..N: self-attention layers -- query is reused, sequence skipped.
    self_layer_config = Configuration(
        query_gen="reuse",                        # reuse previous layer's query
        seq_enc="none",                           # no sequence access
        query_decode="base",                      # keys is None here -> skipped at runtime
        high_order="self_attn",                   # causal self-attention on the query
        query_fuse="residual",
        dim=dim,
    )
    return UnifiedModel([cross_layer_config] + [self_layer_config] * n_self_layers)


def build_stca(dim: int = 64, n_layers: int = 8) -> UnifiedModel:
    """STCA: M layers of pure cross stacking (HighOrder empty, RLB reuse).

    QueryGen returns only the target (N_q = 1, the single-query premise of
    strict O(L)), SeqEnc is an identity K/V projection (target-agnostic, the
    basis for RLB reuse), QueryDecode is single-query cross-attention,
    HighOrder is empty (pure cross stacking), and QueryFuse is the learned
    concat-and-project fusion. The empty HighOrder is the single switch that
    degrades the framework to STCA.
    """
    config = Configuration(
        query_gen="stca",
        seq_enc="stca",
        query_decode="stca_single",
        high_order="none",                        # empty -> STCA degradation
        query_fuse="stca",
        n_layers=n_layers,
        dim=dim,
    )
    return UnifiedModel([config] * n_layers)


# ============================================================
# Degradation summary
# ============================================================
#
#   all blocks active                                 -> HyFormer
#   HighOrder: mixer->self_attn, SeqEnc=TokenMerge     -> LONGER
#   HighOrder=None (empty), QueryGen=target, SeqEnc=id  -> STCA  (strict O(L), pure cross stack)
#   all blocks at "base" (full self-attn in HighOrder)  -> Baseline (O(L^2) reference)
#   LONGER self layers: QueryGen=reuse + SeqEnc=none + Decode=skip + HighOrder=SelfAttn
#


# ============================================================
# Demo: build all four methods and run a forward pass
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    dim = 64
    batch_size = 2
    n_target, n_non_sequence, n_sequence = 1, 13, 512

    target = torch.randn(batch_size, n_target, dim)
    non_sequence_features = torch.randn(batch_size, n_non_sequence, dim)
    sequence_features = torch.randn(batch_size, n_sequence, dim)

    print(f"Input dims (per sample): N_t={n_target}, N_ns={n_non_sequence}, N_ss={n_sequence}, d={dim}\n")

    models = [
        ("Baseline", build_baseline(dim=dim, n_layers=4)),
        ("HyFormer", build_hyformer(dim=dim, n_layers=4, n_global_queries=3,
                                    short_query_length=256, hyformer_seq_enc_tier="longer_style")),
        ("LONGER", build_longer(dim=dim, n_self_layers=3, ss_k_recent_sample=100, token_merge_group_size=4)),
        ("STCA", build_stca(dim=dim, n_layers=8)),
    ]

    for name, model in models:
        model.eval()
        with torch.no_grad():
            score = model(target, non_sequence_features, sequence_features)
        print(f"{name:10s} -> score shape {tuple(score.shape)}")
