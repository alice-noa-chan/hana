"""Decoder-only LLaMA-style Transformer implemented in PyTorch.

The model intentionally keeps the dense attention baseline first-class.  Long
context experiments are guarded at configuration time because a stable dense
baseline is the reference for speed/VRAM/loss comparisons.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .model_config import DecoderConfig


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm used by LLaMA-family models.

    RMSNorm scales activations by their root-mean-square value but does not
    subtract the mean.  That makes it cheaper than LayerNorm while preserving
    stable residual-stream magnitudes.  Normalization runs in float32 for
    numerical stability and is cast back to the input dtype.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * normed.to(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE cache builder.

    RoPE rotates pairs of hidden dimensions by a position-dependent angle.  The
    rotation injects relative position information directly into Q/K vectors and
    works naturally with KV cache because new tokens only need their absolute
    position offset.  cos/sin are precomputed once up to max_position_embeddings
    in float32 and gathered per position id, which avoids rebuilding the table
    on every forward pass and keeps positional precision under bf16/fp16.
    """

    def __init__(self, dim: int, max_position_embeddings: int, scaling: dict[str, Any], theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        if scaling.get("enabled"):
            factor = float(scaling.get("factor", 1.0))
            if scaling.get("type", "linear") == "linear" and factor > 0:
                inv_freq = inv_freq / factor
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq] -> cos/sin: [batch, seq, dim] in float32.
        cos = self.cos_cached.to(position_ids.device)[position_ids]
        sin = self.sin_cached.to(position_ids.device)[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension in pairs for RoPE."""

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q/K in float32 for precision, returning the original dtype.

    q and k are [batch, heads, seq, head_dim].  cos/sin are [batch, seq,
    head_dim]; unsqueezing the heads axis makes them broadcast across heads.
    """

    in_dtype = q.dtype
    cos = cos[:, None, :, :].float()
    sin = sin[:, None, :, :].float()
    qf = q.float()
    kf = k.float()
    q_out = (qf * cos) + (rotate_half(qf) * sin)
    k_out = (kf * cos) + (rotate_half(kf) * sin)
    return q_out.to(in_dtype), k_out.to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    The gate projection decides which hidden features should pass through while
    the up projection creates candidate features.  Multiplying SiLU(gate) by
    up is the LLaMA-family FFN pattern.
    """

    def __init__(self, hidden_size: int, use_bias: bool, dropout: float) -> None:
        super().__init__()
        intermediate = int(8 * hidden_size / 3)
        intermediate = 256 * math.ceil(intermediate / 256)
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=use_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=use_bias)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat KV heads for GQA/MQA without copying when repeats=1."""

    if repeats == 1:
        return x
    bsz, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, kv_heads, repeats, seq_len, head_dim)
    return x.reshape(bsz, kv_heads * repeats, seq_len, head_dim)


def select_attention_backend(requested: str) -> tuple[str, list[str]]:
    """Choose FlashAttention -> SDPA -> eager attention and explain fallbacks."""

    notes: list[str] = []
    if requested in {"flash3", "auto"}:
        try:
            import flash_attn_interface  # noqa: F401

            return "flash3", notes
        except Exception as exc:
            notes.append(f"FlashAttention-3 unavailable: {exc}")
    if requested in {"flash2", "flash", "auto"}:
        try:
            import flash_attn  # noqa: F401

            return "flash2", notes
        except Exception as exc:
            notes.append(f"FlashAttention-2 unavailable: {exc}")
    if requested in {"sdpa", "auto"} and hasattr(F, "scaled_dot_product_attention"):
        return "sdpa", notes
    if requested not in {"eager", "auto"}:
        notes.append(f"Requested backend '{requested}' is unavailable; using eager attention.")
    return "eager", notes


def build_attention_bias(
    attention_mask: torch.Tensor | None,
    document_ids: torch.Tensor | None,
    q_len: int,
    kv_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int | None,
    attention_mode: str = "causal",
    prefix_lengths: torch.Tensor | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """Build an additive causal or block-causal attention bias.

    Supports KV-cache decoding (past_len > 0), sliding windows, padding masks,
    and packed-sequence document segmentation.  Query positions are absolute
    positions [past_len, past_len + q_len); key positions are [0, kv_len).  A
    key is visible only when key_pos <= query_pos, when it lies inside the
    sliding window, and (for packed sequences) when it belongs to the same
    document as the query.
    """

    q_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    if attention_mode == "causal":
        allowed = (k_pos <= q_pos)[None, :, :]
    elif attention_mode == "bidirectional":
        allowed = torch.ones((1, q_len, kv_len), dtype=torch.bool, device=device)
    elif attention_mode == "prefix_block":
        if past_len:
            raise ValueError("prefix_block attention does not support a KV cache.")
        if prefix_lengths is None:
            raise ValueError("prefix_lengths is required for prefix_block attention.")
        if prefix_lengths.ndim != 1:
            raise ValueError("prefix_lengths must have shape [batch].")
        width = int(block_size or q_len)
        if width <= 0:
            raise ValueError("block_size must be positive for prefix_block attention.")
        prefix = prefix_lengths.to(device=device, dtype=torch.long)[:, None, None]
        query = q_pos[None, :, :]
        key = k_pos[None, :, :]
        prefix_query = query < prefix
        query_block = torch.div((query - prefix).clamp_min(0), width, rounding_mode="floor")
        key_block = torch.div((key - prefix).clamp_min(0), width, rounding_mode="floor")
        # Prefix tokens remain token-causal. Suffix tokens see the clean prefix,
        # every earlier block, and their entire current noisy block.
        block_visible = (key < prefix) | ((key >= prefix) & (key_block <= query_block))
        allowed = torch.where(prefix_query, key <= query, block_visible)
    else:
        raise ValueError(f"Unsupported attention_mode: {attention_mode}")
    if sliding_window is not None and sliding_window > 0:
        allowed = allowed & (k_pos[None, :, :] >= (q_pos[None, :, :] - sliding_window + 1))
    allowed = allowed[:, None, :, :]
    if document_ids is not None:
        # Intra-document attention: a query token can only see keys in the same
        # packed document.  document_ids covers the current window; past tokens
        # (KV cache) are not used together with packing.
        q_doc = document_ids[:, None, :, None]
        k_doc = document_ids[:, None, None, :]
        allowed = allowed & (q_doc == k_doc)
    if attention_mask is not None:
        key_mask = attention_mask[:, None, None, :kv_len].to(torch.bool)
        allowed = allowed & key_mask
    min_value = torch.finfo(dtype).min
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, min_value)


class CausalSelfAttention(nn.Module):
    """MHA/MQA/GQA attention with RoPE, KV cache, and backend fallback."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.kv_repeats = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=cfg.use_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.use_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.use_bias)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.attention_dropout)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.backend, self.backend_notes = select_attention_backend(cfg.attention_backend)
        self.window = int(cfg.sliding_window.get("window_size", 0)) if cfg.sliding_window.get("enabled") else None
        # FlashAttention varlen packing path is verified numerically against the
        # SDPA reference on first use: "unverified" -> "ok" | "disabled".
        self._varlen_state = "unverified"

    def _flash_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
        """Run FlashAttention when the installed package supports the simple causal path."""

        if self.window is not None:
            return None
        try:
            if self.backend == "flash3":
                from flash_attn_interface import flash_attn_func
            else:
                from flash_attn import flash_attn_func

            # FlashAttention uses [batch, seq, heads, head_dim].  Dropout is
            # applied by the kernel only during training.
            qf = q.transpose(1, 2)
            kf = k.transpose(1, 2)
            vf = v.transpose(1, 2)
            out = flash_attn_func(
                qf,
                kf,
                vf,
                dropout_p=self.cfg.attention_dropout if self.training else 0.0,
                causal=True,
            )
            return out.transpose(1, 2)
        except Exception:
            return None

    def _flash_varlen_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        document_ids: torch.Tensor,
        dropout_p: float,
    ) -> torch.Tensor | None:
        """Block-diagonal attention for packed sequences via FlashAttention varlen.

        Each packed document becomes one variable-length sequence described by
        cu_seqlens, so attention never crosses document boundaries and no dense
        L x L mask is materialized.  Returns None (caller falls back to SDPA) for
        padded batches or any unsupported case.
        """

        if (document_ids < 0).any():
            return None  # padding present; unpadding is left to the SDPA path
        try:
            if self.backend == "flash3":
                from flash_attn_interface import flash_attn_varlen_func
            else:
                from flash_attn import flash_attn_varlen_func

            bsz, nheads, seq_len, head_dim = q.shape
            total = bsz * seq_len
            doc_flat = document_ids.reshape(-1)
            idx = torch.arange(total, device=q.device)
            row_start = (idx % seq_len) == 0
            doc_change = torch.zeros(total, dtype=torch.bool, device=q.device)
            doc_change[1:] = doc_flat[1:] != doc_flat[:-1]
            new_seq = row_start | doc_change
            new_seq[0] = True
            starts = idx[new_seq]
            cu_seqlens = torch.cat([starts, torch.tensor([total], device=q.device)]).to(torch.int32)
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

            qf = q.transpose(1, 2).reshape(total, nheads, head_dim)
            kf = k.transpose(1, 2).reshape(total, k.shape[1], head_dim)
            vf = v.transpose(1, 2).reshape(total, v.shape[1], head_dim)
            out = flash_attn_varlen_func(
                qf,
                kf,
                vf,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=dropout_p,
                causal=True,
            )
            return out.reshape(bsz, seq_len, nheads, head_dim).transpose(1, 2)
        except Exception:
            return None

    def _packed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: torch.Tensor | None,
        document_ids: torch.Tensor,
        dropout_p: float,
    ) -> torch.Tensor:
        """Use varlen flash for packing once it matches the SDPA reference.

        The first packed batch computes both the SDPA reference and the varlen
        output and compares them; only if they agree does this layer trust the
        varlen kernel thereafter.  Any mismatch or error permanently disables
        varlen so a silent correctness bug can never affect training.
        """

        if self._varlen_state == "ok":
            candidate = self._flash_varlen_forward(q, k, v, document_ids, dropout_p)
            if candidate is not None:
                return candidate
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)
        check = self._flash_varlen_forward(q, k, v, document_ids, 0.0)
        if check is not None and torch.allclose(check.float(), reference.float(), atol=1e-2, rtol=1e-2):
            self._varlen_state = "ok"
        else:
            self._varlen_state = "disabled"
        return reference

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None,
        simple_causal: bool,
        document_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        allow_varlen: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        k_full = repeat_kv(k, self.kv_repeats)
        v_full = repeat_kv(v, self.kv_repeats)
        dropout_p = self.cfg.attention_dropout if self.training else 0.0

        if simple_causal and self.backend in {"flash2", "flash3"}:
            flashed = self._flash_forward(q, k_full, v_full)
            if flashed is not None:
                out = flashed
            else:
                out = F.scaled_dot_product_attention(
                    q, k_full, v_full, attn_mask=None, dropout_p=dropout_p, is_causal=True
                )
        elif simple_causal and self.backend == "sdpa":
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        elif self.backend in {"flash2", "flash3", "sdpa"}:
            use_varlen = (
                allow_varlen
                and self.backend in {"flash2", "flash3"}
                and document_ids is not None
                and self.window is None
                and past_key_value is None
                and self._varlen_state != "disabled"
            )
            if use_varlen:
                out = self._packed_attention(q, k_full, v_full, attn_bias, document_ids, dropout_p)
            else:
                out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_bias, dropout_p=dropout_p)
        else:
            scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_bias is not None:
                scores = scores + attn_bias
            probs = self.dropout(torch.softmax(scores.float(), dim=-1).to(q.dtype))
            out = torch.matmul(probs, v_full)

        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.cfg.hidden_size)
        return self.o_proj(out), present


class DecoderBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size)
        self.ffn = SwiGLU(cfg.hidden_size, cfg.use_bias, cfg.residual_dropout)
        self.residual_dropout = nn.Dropout(cfg.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None,
        simple_causal: bool,
        document_ids: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        use_cache: bool,
        allow_varlen: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            attn_bias,
            simple_causal,
            document_ids,
            past_key_value,
            use_cache,
            allow_varlen,
        )
        x = x + self.residual_dropout(attn_out)
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x, present


class CausalWorkspace(nn.Module):
    """Compress the running causal context and selectively broadcast it back.

    This is a machine-oriented global-workspace bottleneck, not a claim about
    consciousness.  A cumulative state makes cached decoding exactly match a
    full causal forward pass.
    """

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.hidden_size)
        self.collect = nn.Linear(cfg.hidden_size, cfg.workspace_bottleneck_size, bias=cfg.use_bias)
        self.broadcast = nn.Linear(cfg.workspace_bottleneck_size, cfg.hidden_size, bias=cfg.use_bias)
        self.gate = nn.Linear(cfg.hidden_size, 1, bias=True)
        self.gate_bias = cfg.workspace_gate_bias

    def reset_gate_bias(self) -> None:
        nn.init.constant_(self.gate.bias, self.gate_bias)

    def forward(
        self,
        x: torch.Tensor,
        past_state: tuple[torch.Tensor, int] | None = None,
        use_cache: bool = False,
        document_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, int] | None, torch.Tensor]:
        features = torch.tanh(self.collect(self.norm(x)))
        past_sum = None if past_state is None else past_state[0]
        past_count = 0 if past_state is None else int(past_state[1])
        if document_ids is not None:
            if past_state is not None or use_cache:
                raise ValueError("Packed document_ids cannot be combined with a workspace KV cache.")
            if document_ids.shape != x.shape[:2]:
                raise ValueError("document_ids must match the workspace batch and sequence dimensions.")
            # Packed samples are independent documents.  Reset the workspace
            # accumulator at every contiguous document boundary so one
            # document cannot alter another document's hidden states.
            features_fp32 = features.float()
            cumulative_all = features_fp32.cumsum(dim=1)
            positions = torch.arange(x.size(1), device=x.device).view(1, -1).expand(x.size(0), -1)
            boundaries = torch.ones_like(document_ids, dtype=torch.bool)
            boundaries[:, 1:] = document_ids[:, 1:].ne(document_ids[:, :-1])
            candidate_starts = torch.where(boundaries, positions, torch.zeros_like(positions))
            segment_starts = torch.cummax(candidate_starts, dim=1).values
            prefix = torch.cat(
                [
                    torch.zeros(
                        (x.size(0), 1, features_fp32.size(-1)),
                        dtype=features_fp32.dtype,
                        device=x.device,
                    ),
                    cumulative_all,
                ],
                dim=1,
            )
            segment_base = prefix.gather(1, segment_starts.unsqueeze(-1).expand(-1, -1, features_fp32.size(-1)))
            cumulative = cumulative_all - segment_base
            counts = (positions - segment_starts + 1).to(features_fp32.dtype).unsqueeze(-1)
            summary_fp32 = cumulative / counts
            summary = summary_fp32.to(x.dtype)
        else:
            cumulative = features.float().cumsum(dim=1)
            if past_sum is not None:
                cumulative = cumulative + past_sum.float().unsqueeze(1)
            counts = torch.arange(
                past_count + 1,
                past_count + x.size(1) + 1,
                dtype=cumulative.dtype,
                device=x.device,
            ).view(1, -1, 1)
            summary = (cumulative / counts).to(x.dtype)
        proposal = self.broadcast(summary)
        gate = torch.sigmoid(self.gate(self.norm(x)))
        output = x + gate * proposal
        # Keep the accumulator in fp32.  Casting it back to bf16/fp16 would
        # make token-by-token decoding drift from the mathematically
        # equivalent full-sequence computation as the context grows.
        present = (cumulative[:, -1], past_count + x.size(1)) if use_cache else None
        return output, present, summary


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only language model with optional MTP heads."""

    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_dropout = nn.Dropout(cfg.embedding_dropout)
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = (
            RotaryEmbedding(
                cfg.hidden_size // cfg.num_attention_heads,
                cfg.max_position_embeddings,
                cfg.rope_scaling,
                cfg.rope_theta,
            )
            if cfg.rope
            else None
        )
        self.mtp_heads = nn.ModuleList(
            [
                nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
                for _ in range(cfg.mtp_num_future_tokens if cfg.mtp_enabled else 0)
            ]
        )
        workspace_active = cfg.cognitive_enabled and cfg.workspace_enabled
        self.workspaces = nn.ModuleDict(
            {
                str(index): CausalWorkspace(cfg)
                for index in range(cfg.num_layers)
                if workspace_active and (index + 1) % cfg.workspace_every_n_layers == 0
            }
        )
        self.latent_predictor = (
            nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            if cfg.cognitive_enabled and cfg.predictive_coding_enabled
            else None
        )
        self.apply(self._init_weights)
        for workspace in self.workspaces.values():
            workspace.reset_gate_bias()
        # Scale residual output projections by 1/sqrt(2*num_layers) so the
        # residual stream does not grow with depth (GPT-2 / LLaMA practice).
        self._scale_residual_projections()
        # Weight tying must happen after initialization so lm_head and the
        # embedding share one initialized tensor.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.backend = self.layers[0].attn.backend if self.layers else "none"
        self.backend_notes = self.layers[0].attn.backend_notes if self.layers else []

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2 * max(1, self.cfg.num_layers))
        with torch.no_grad():
            for layer in self.layers:
                layer.attn.o_proj.weight.mul_(scale)
                layer.ffn.down_proj.weight.mul_(scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        past_workspace_states: list[tuple[torch.Tensor, int] | None] | None = None,
        use_cache: bool = False,
        label_smoothing: float = 0.0,
        attention_mode: str = "causal",
        prefix_lengths: torch.Tensor | None = None,
        block_size: int | None = None,
        loss_mode: str = "next_token",
        return_hidden_states: bool = False,
    ) -> dict[str, Any]:
        bsz, seq_len = input_ids.shape
        past_len = 0 if past_key_values is None or not past_key_values else past_key_values[0][0].shape[2]
        if past_len + seq_len > self.cfg.max_position_embeddings:
            raise ValueError(
                f"Sequence length {past_len + seq_len} exceeds max_position_embeddings "
                f"{self.cfg.max_position_embeddings}."
            )

        x = self.embed_dropout(self.embed_tokens(input_ids))
        if position_ids is None:
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            )
        if self.rotary is not None:
            cos, sin = self.rotary(position_ids)
        else:
            head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads
            shape = (*position_ids.shape, head_dim)
            cos = torch.ones(shape, dtype=torch.float32, device=input_ids.device)
            sin = torch.zeros(shape, dtype=torch.float32, device=input_ids.device)

        window = self.layers[0].attn.window if self.layers else None
        if attention_mode not in {"causal", "bidirectional", "prefix_block"}:
            raise ValueError(f"Unsupported attention_mode: {attention_mode}")
        if attention_mode == "prefix_block" and (
            prefix_lengths is None
            or prefix_lengths.shape != (bsz,)
            or (prefix_lengths < 0).any()
            or (prefix_lengths > seq_len).any()
        ):
            raise ValueError("prefix_lengths must contain one value in [0, sequence_length] per batch row.")
        if attention_mode != "causal" and past_key_values is not None:
            raise ValueError(f"{attention_mode} attention cannot reuse a causal KV cache.")
        if past_workspace_states is not None and len(past_workspace_states) != len(self.layers):
            raise ValueError("past_workspace_states must contain one entry per decoder layer.")
        if attention_mode != "causal" and past_workspace_states is not None:
            raise ValueError(f"{attention_mode} attention cannot reuse a causal workspace cache.")
        if self.workspaces and past_len > 0 and past_workspace_states is None:
            raise ValueError("Cognitive workspace decoding requires past_workspace_states with the KV cache.")
        if past_workspace_states is not None and past_key_values is None:
            raise ValueError("past_workspace_states cannot be used without a matching KV cache.")
        if past_workspace_states is not None:
            for index in range(len(self.layers)):
                state = past_workspace_states[index]
                workspace_active = str(index) in self.workspaces
                if workspace_active and (state is None or int(state[1]) != past_len):
                    raise ValueError("Every active workspace cache must match the KV cache length.")
                if not workspace_active and state is not None:
                    raise ValueError("Inactive decoder layers cannot contain workspace cache state.")
        needs_mask = (
            attention_mask is not None
            or document_ids is not None
            or window is not None
            or past_len > 0
            or attention_mode != "causal"
        )
        eager = bool(self.layers) and self.layers[0].attn.backend == "eager"
        simple_causal = attention_mode == "causal" and not needs_mask
        attn_bias = None
        if needs_mask or eager:
            attn_bias = build_attention_bias(
                attention_mask,
                document_ids,
                seq_len,
                past_len + seq_len,
                past_len,
                x.dtype,
                x.device,
                window,
                attention_mode,
                prefix_lengths,
                block_size,
            )
            simple_causal = False

        new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        new_workspace_states: list[tuple[torch.Tensor, int] | None] = []
        workspace_activations: list[torch.Tensor] | None = [] if return_hidden_states else None
        hidden_states = [x] if return_hidden_states else None
        for idx, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[idx]
            if self.training and self.cfg.gradient_checkpointing and not use_cache:
                # Checkpointing trades extra compute for lower activation memory.
                def custom_forward(hidden: torch.Tensor, current_layer=layer) -> torch.Tensor:
                    return current_layer(
                        hidden,
                        cos,
                        sin,
                        attn_bias,
                        simple_causal,
                        document_ids,
                        None,
                        False,
                        attention_mode == "causal",
                    )[0]

                x = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
                present = None
            else:
                x, present = layer(
                    x,
                    cos,
                    sin,
                    attn_bias,
                    simple_causal,
                    document_ids,
                    layer_past,
                    use_cache,
                    attention_mode == "causal",
                )
            if present is not None:
                new_past.append(present)
            try:
                workspace = self.workspaces[str(idx)]
            except KeyError:
                workspace = None
            workspace_present = None
            if workspace is not None:
                workspace_past = None if past_workspace_states is None else past_workspace_states[idx]
                x, workspace_present, workspace_summary = workspace(
                    x, workspace_past, use_cache, document_ids=document_ids
                )
                if workspace_activations is not None:
                    workspace_activations.append(workspace_summary)
            new_workspace_states.append(workspace_present)
            if hidden_states is not None:
                hidden_states.append(x)

        normed = self.norm(x)
        logits = self.lm_head(normed)
        logits = self._soft_cap(logits)

        loss = None
        ce_loss = None
        mtp_loss = None
        predictive_loss = None
        homeostatic_loss = None
        num_loss_tokens = torch.zeros((), device=input_ids.device)
        if labels is not None:
            if loss_mode == "next_token":
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
            elif loss_mode == "same_token":
                shift_logits = logits.contiguous()
                shift_labels = labels.contiguous()
            else:
                raise ValueError(f"Unsupported loss_mode: {loss_mode}")
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            valid_token_count = (flat_labels != -100).sum()
            if not valid_token_count.item():
                raise ValueError(f"Language-model labels contain zero supervised {loss_mode} targets.")
            ce_loss = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                label_smoothing=float(label_smoothing),
            )
            loss = ce_loss
            num_loss_tokens = valid_token_count.to(loss.dtype)
            if self.cfg.z_loss_weight > 0:
                valid = flat_labels != -100
                if valid.any():
                    lse = torch.logsumexp(flat_logits.float()[valid], dim=-1)
                    loss = loss + self.cfg.z_loss_weight * lse.pow(2).mean().to(loss.dtype)
            if self.cfg.mtp_enabled and loss_mode == "next_token":
                mtp_losses = []
                for offset, head in enumerate(self.mtp_heads, start=2):
                    if normed.shape[1] <= offset:
                        continue
                    # The MTP head at offset=2 predicts token[t+2] from hidden[t].
                    mtp_logits = self._soft_cap(head(normed[:, :-offset, :]))
                    mtp_labels = labels[:, offset:].clone()
                    if document_ids is not None:
                        same_document = document_ids[:, :-offset].eq(document_ids[:, offset:])
                        mtp_labels = mtp_labels.masked_fill(~same_document, -100)
                    if not mtp_labels.ne(-100).any():
                        continue
                    mtp_losses.append(
                        F.cross_entropy(
                            mtp_logits.reshape(-1, mtp_logits.size(-1)),
                            mtp_labels.reshape(-1),
                            ignore_index=-100,
                        )
                    )
                if mtp_losses:
                    mtp_loss = torch.stack(mtp_losses).mean()
                    loss = loss + self.cfg.mtp_loss_weight * mtp_loss
            if self.latent_predictor is not None and loss_mode == "next_token" and normed.size(1) > 1:
                predictive_mask = labels[:, 1:].ne(-100)
                if attention_mask is not None:
                    predictive_mask = predictive_mask & attention_mask[:, 1:].bool()
                if document_ids is not None:
                    predictive_mask = predictive_mask & document_ids[:, 1:].eq(document_ids[:, :-1])
                if predictive_mask.any():
                    predicted_latent = self.latent_predictor(normed[:, :-1])
                    target_latent = normed[:, 1:].detach()
                    token_error = (predicted_latent.float() - target_latent.float()).square().mean(dim=-1)
                    predictive_loss = token_error[predictive_mask].mean().to(loss.dtype)
                    loss = loss + self.cfg.predictive_coding_loss_weight * predictive_loss
            if self.cfg.cognitive_enabled and self.cfg.homeostasis_enabled:
                valid_positions = (
                    torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                    if attention_mask is None
                    else attention_mask.bool()
                )
                residual_rms = x.float().square().mean(dim=-1).add(1e-12).sqrt()
                target = torch.tensor(self.cfg.homeostasis_target_rms, device=x.device)
                homeostatic_loss = (residual_rms[valid_positions].log() - target.log()).square().mean().to(loss.dtype)
                loss = loss + self.cfg.homeostasis_loss_weight * homeostatic_loss

        return {
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss,
            "mtp_loss": mtp_loss,
            "predictive_loss": predictive_loss,
            "homeostatic_loss": homeostatic_loss,
            "num_loss_tokens": num_loss_tokens,
            "past_key_values": new_past if use_cache else None,
            "workspace_states": new_workspace_states if use_cache else None,
            "workspace_activations": workspace_activations,
            "hidden_states": hidden_states,
        }

    def _soft_cap(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.cfg.logit_softcap
        if cap and cap > 0:
            return cap * torch.tanh(logits / cap)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        suppress_ids: set[int] | frozenset[int] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Autoregressive decoding with KV cache and nucleus/top-k sampling."""

        self.eval()
        generated = input_ids
        if generated.size(1) > self.cfg.max_position_embeddings:
            raise ValueError(
                f"Prompt length {generated.size(1)} exceeds max_position_embeddings {self.cfg.max_position_embeddings}."
            )
        max_new_tokens = min(max(0, int(max_new_tokens)), max(0, self.cfg.max_position_embeddings - generated.size(1)))
        if max_new_tokens <= 0:
            return generated
        past = None
        workspace_states = None
        for _ in range(max_new_tokens):
            current = generated[:, -1:] if past is not None and use_cache else generated
            out = self(
                current,
                past_key_values=past,
                past_workspace_states=workspace_states,
                use_cache=use_cache,
            )
            logits = out["logits"][:, -1, :]
            past = out["past_key_values"] if use_cache else None
            workspace_states = out["workspace_states"] if use_cache else None
            if repetition_penalty and repetition_penalty != 1.0:
                penalty = float(repetition_penalty)
                token_scores = torch.gather(logits, 1, generated)
                token_scores = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
                logits.scatter_(1, generated, token_scores)
            if suppress_ids:
                valid_suppressed = [index for index in suppress_ids if 0 <= index < logits.size(-1)]
                if valid_suppressed:
                    logits[:, valid_suppressed] = -float("inf")
            raw_probabilities = torch.softmax(logits.float(), dim=-1)
            raw_entropy = -(raw_probabilities * raw_probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                sampling_probabilities = raw_probabilities
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = probs.cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                    logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                sampling_probabilities = probs
            if trace is not None:
                entropy = -(sampling_probabilities * sampling_probabilities.clamp_min(1e-12).log()).sum(dim=-1)
                top_probs, top_ids = torch.topk(sampling_probabilities, min(5, sampling_probabilities.size(-1)), dim=-1)
                trace.append(
                    {
                        "phase": "ar",
                        "index": len(trace),
                        "token_ids": next_id.squeeze(-1).detach().cpu().tolist(),
                        "entropy": entropy.detach().cpu().tolist(),
                        "raw_entropy": raw_entropy.detach().cpu().tolist(),
                        "top_token_ids": top_ids.detach().cpu().tolist(),
                        "top_probabilities": top_probs.detach().cpu().tolist(),
                    }
                )
            generated = torch.cat([generated, next_id], dim=1)
            if torch.all(next_id.squeeze(-1).eq(eos_id)).item():
                break
        return generated


def build_model(config: dict[str, Any]) -> DecoderOnlyTransformer:
    """Build the model and validate experimental feature gates."""

    exp = config["long_context_experimental"]
    unsupported = [
        name
        for name in ("activation_beacon", "ring_attention", "index_share", "csa")
        if exp[name].get("enabled", False)
    ]
    if unsupported:
        raise RuntimeError("Unimplemented long-context features cannot be enabled: " + ", ".join(unsupported))
    return DecoderOnlyTransformer(DecoderConfig.from_config(config))
