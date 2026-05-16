"""
Export the Cactus Compute Needle JAX/Flax model to ONNX format.

Usage:
    pip install -r requirements_export.txt
    pip install git+https://github.com/cactus-compute/needle
    python export_onnx.py [--output-dir ../models] [--fp16] [--validate]

The script produces:
    models/needle_encoder.onnx   (or needle_encoder_fp16.onnx)
    models/needle_decoder.onnx   (or needle_decoder_fp16.onnx)
    models/tokenizer/needle.model

Strategy: Re-implement in PyTorch (not jax2tf) because Flax's nn.scan
produces loop nodes in TF/ONNX that perform poorly in onnxruntime-web.
PyTorch gives us unrolled layers and a clean ONNX graph.
"""

import argparse
import math
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PyTorch re-implementation of Needle
# ---------------------------------------------------------------------------

def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Returns (cos, sin) each shape [max_seq_len, head_dim//2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings. x: [B, H, T, head_dim]."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim//2]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ZCRMSNorm(nn.Module):
    """Zero-centred RMSNorm: (1 + scale) * x / rms(x), scale init=0."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return ((1.0 + self.scale) * x / rms.to(x.dtype))


class MultiHeadAttention(nn.Module):
    """Grouped-query attention with RoPE, no bias, ZCRMSNorm on Q and K."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = ZCRMSNorm(self.head_dim)
        self.k_norm = ZCRMSNorm(self.head_dim)

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        apply_rope_to_q: bool = True,
    ) -> torch.Tensor:
        B = q_input.shape[0]
        q = self.q_proj(q_input).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).reshape(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).reshape(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            cos, sin = rope
            if apply_rope_to_q:
                q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            # mask: True = attend, False = block; set blocked to -inf
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "drelu"):
        super().__init__()
        self.activation = activation
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.activation == "swiglu":
            h = F.silu(gate) * up
        elif self.activation == "geglu":
            h = F.gelu(gate) * up
        else:  # drelu (default)
            h = F.relu(gate) * F.relu(up)
        return self.down_proj(h)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int, activation: str, no_feedforward: bool):
        super().__init__()
        self.no_feedforward = no_feedforward
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.norm1 = ZCRMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, num_kv_heads)
        if not no_feedforward:
            self.ffn_gate = nn.Parameter(torch.zeros(1))
            self.norm2 = ZCRMSNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff, activation)

    def forward(self, x: torch.Tensor, mask=None, rope=None) -> torch.Tensor:
        gate = torch.sigmoid(self.attn_gate)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, mask=mask, rope=rope)
        x = residual + gate * x

        if not self.no_feedforward:
            ffn_gate = torch.sigmoid(self.ffn_gate)
            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = residual + ffn_gate * x
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int, num_layers: int, activation: str,
                 no_feedforward: bool, max_seq_len: int, rope_theta: float):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, num_kv_heads, d_ff, activation, no_feedforward)
            for _ in range(num_layers)
        ])
        self.final_norm = ZCRMSNorm(d_model)
        cos, sin = precompute_rope(d_model // num_heads, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        rope = (self.rope_cos, self.rope_sin)
        for layer in self.layers:
            x = layer(x, mask=mask, rope=rope)
        return self.final_norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int, activation: str, no_feedforward: bool):
        super().__init__()
        self.no_feedforward = no_feedforward
        self.self_attn_gate = nn.Parameter(torch.zeros(1))
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        self.norm1 = ZCRMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, num_kv_heads)
        self.norm2 = ZCRMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, num_kv_heads)
        if not no_feedforward:
            self.ffn_gate = nn.Parameter(torch.zeros(1))
            self.norm3 = ZCRMSNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff, activation)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                self_mask=None, cross_mask=None, rope=None) -> torch.Tensor:
        self_gate = torch.sigmoid(self.self_attn_gate)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, mask=self_mask, rope=rope)
        x = residual + self_gate * x

        cross_gate = torch.sigmoid(self.cross_attn_gate)
        residual = x
        x = self.norm2(x)
        # Cross-attention: no RoPE on queries (rope_keys_only=False means both, but
        # cross-attn in original has rope=None so no RoPE)
        x = self.cross_attn(x, encoder_out, mask=cross_mask, rope=None)
        x = residual + cross_gate * x

        if not self.no_feedforward:
            ffn_gate = torch.sigmoid(self.ffn_gate)
            residual = x
            x = self.norm3(x)
            x = self.ffn(x)
            x = residual + ffn_gate * x
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int, num_layers: int, activation: str,
                 no_feedforward: bool, max_seq_len: int, rope_theta: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, num_kv_heads, d_ff, activation, no_feedforward)
            for _ in range(num_layers)
        ])
        self.final_norm = ZCRMSNorm(d_model)
        cos, sin = precompute_rope(d_model // num_heads, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                self_mask=None, cross_mask=None) -> torch.Tensor:
        rope = (self.rope_cos, self.rope_sin)
        for layer in self.layers:
            x = layer(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope)
        return self.final_norm(x)


class NeedleEncoderONNX(nn.Module):
    """Encoder-only wrapper: input_ids -> hidden_states."""

    def __init__(self, vocab_size, d_model, num_heads, num_kv_heads,
                 d_ff, num_enc_layers, activation, no_feedforward,
                 max_seq_len, rope_theta):
        super().__init__()
        self.embed_scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(d_model, num_heads, num_kv_heads, d_ff,
                               num_enc_layers, activation, no_feedforward,
                               max_seq_len, rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, enc_len] int32
        # Build padding mask: True = attend, False = ignore (pad=0)
        pad_mask = (input_ids != 0)  # [B, enc_len]
        # Expand to [B, 1, 1, enc_len] for self-attention
        mask = pad_mask.unsqueeze(1).unsqueeze(2)
        x = self.embedding(input_ids.long()) * self.embed_scale
        return self.encoder(x, mask=mask)  # [B, enc_len, d_model]


class NeedleDecoderONNX(nn.Module):
    """Decoder-only wrapper: (decoder_input_ids, encoder_hidden_states) -> logits."""

    def __init__(self, vocab_size, d_model, num_heads, num_kv_heads,
                 d_ff, num_dec_layers, activation, no_feedforward,
                 max_seq_len, rope_theta, embedding_weight: torch.Tensor):
        super().__init__()
        self.embed_scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding.weight = nn.Parameter(embedding_weight, requires_grad=False)
        self.decoder = Decoder(d_model, num_heads, num_kv_heads, d_ff,
                               num_dec_layers, activation, no_feedforward,
                               max_seq_len, rope_theta)

    def forward(self, decoder_input_ids: torch.Tensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # decoder_input_ids: [B, dec_len] int32
        # encoder_hidden_states: [B, enc_len, d_model]
        B, dec_len = decoder_input_ids.shape
        # Causal mask for self-attention: [1, 1, dec_len, dec_len]
        causal = torch.tril(torch.ones(dec_len, dec_len,
                                       device=decoder_input_ids.device, dtype=torch.bool))
        self_mask = causal.unsqueeze(0).unsqueeze(0)

        x = self.embedding(decoder_input_ids.long()) * self.embed_scale
        x = self.decoder(x, encoder_hidden_states,
                         self_mask=self_mask, cross_mask=None)
        logits = x.float() @ self.embedding.weight.float().T
        return logits  # [B, dec_len, vocab_size]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_jax_params(checkpoint_path: str):
    """Load JAX pickle checkpoint. Returns (params dict as float32 numpy, config dict)."""
    import jax
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    # Convert all leaves to float32 numpy immediately — JAX DeviceArrays can't
    # be pickled across processes and bfloat16 isn't supported by PyTorch directly.
    params = jax.tree.map(lambda x: np.array(x, dtype=np.float32), data["params"])
    config = data.get("config", {})
    if hasattr(config, '__dataclass_fields__'):
        # TransformerConfig object — convert to dict
        config = {k: getattr(config, k) for k in config.__dataclass_fields__}
    return params, config


def _np(x) -> np.ndarray:
    """Convert JAX/numpy array to float32 numpy."""
    return np.array(x, dtype=np.float32)


def _t(x) -> torch.Tensor:
    """JAX param -> float32 torch tensor."""
    return torch.tensor(_np(x))


def load_enc_block_weights(block: EncoderBlock, p: dict, layer_idx: int):
    """Copy weights for one encoder block from stacked JAX params."""
    # attn_gate: scalar per layer
    block.attn_gate.data = _t(p["attn_gate"][layer_idx]).reshape(1)
    # ZCRMSNorm_0 (norm before self-attn)
    block.norm1.scale.data = _t(p["ZCRMSNorm_0"]["scale"][layer_idx])
    # self_attn projections (Flax kernel: [in, out] -> PyTorch weight: [out, in])
    sa = p["self_attn"]
    block.self_attn.q_proj.weight.data = _t(sa["q_proj"]["kernel"][layer_idx]).T
    block.self_attn.k_proj.weight.data = _t(sa["k_proj"]["kernel"][layer_idx]).T
    block.self_attn.v_proj.weight.data = _t(sa["v_proj"]["kernel"][layer_idx]).T
    block.self_attn.out_proj.weight.data = _t(sa["out_proj"]["kernel"][layer_idx]).T
    block.self_attn.q_norm.scale.data = _t(sa["q_norm"]["scale"][layer_idx])
    block.self_attn.k_norm.scale.data = _t(sa["k_norm"]["scale"][layer_idx])

    if not block.no_feedforward:
        block.ffn_gate.data = _t(p["ffn_gate"][layer_idx]).reshape(1)
        block.norm2.scale.data = _t(p["ZCRMSNorm_1"]["scale"][layer_idx])
        ff = p["FeedForward_0"]
        block.ffn.gate_proj.weight.data = _t(ff["gate_proj"]["kernel"][layer_idx]).T
        block.ffn.up_proj.weight.data = _t(ff["up_proj"]["kernel"][layer_idx]).T
        block.ffn.down_proj.weight.data = _t(ff["down_proj"]["kernel"][layer_idx]).T


def load_dec_block_weights(block: DecoderBlock, p: dict, layer_idx: int):
    """Copy weights for one decoder block from stacked JAX params."""
    block.self_attn_gate.data = _t(p["self_attn_gate"][layer_idx]).reshape(1)
    block.norm1.scale.data = _t(p["ZCRMSNorm_0"]["scale"][layer_idx])
    sa = p["self_attn"]
    block.self_attn.q_proj.weight.data = _t(sa["q_proj"]["kernel"][layer_idx]).T
    block.self_attn.k_proj.weight.data = _t(sa["k_proj"]["kernel"][layer_idx]).T
    block.self_attn.v_proj.weight.data = _t(sa["v_proj"]["kernel"][layer_idx]).T
    block.self_attn.out_proj.weight.data = _t(sa["out_proj"]["kernel"][layer_idx]).T
    block.self_attn.q_norm.scale.data = _t(sa["q_norm"]["scale"][layer_idx])
    block.self_attn.k_norm.scale.data = _t(sa["k_norm"]["scale"][layer_idx])

    block.cross_attn_gate.data = _t(p["cross_attn_gate"][layer_idx]).reshape(1)
    block.norm2.scale.data = _t(p["ZCRMSNorm_1"]["scale"][layer_idx])
    ca = p["cross_attn"]
    block.cross_attn.q_proj.weight.data = _t(ca["q_proj"]["kernel"][layer_idx]).T
    block.cross_attn.k_proj.weight.data = _t(ca["k_proj"]["kernel"][layer_idx]).T
    block.cross_attn.v_proj.weight.data = _t(ca["v_proj"]["kernel"][layer_idx]).T
    block.cross_attn.out_proj.weight.data = _t(ca["out_proj"]["kernel"][layer_idx]).T
    block.cross_attn.q_norm.scale.data = _t(ca["q_norm"]["scale"][layer_idx])
    block.cross_attn.k_norm.scale.data = _t(ca["k_norm"]["scale"][layer_idx])

    if not block.no_feedforward:
        block.ffn_gate.data = _t(p["ffn_gate"][layer_idx]).reshape(1)
        block.norm3.scale.data = _t(p["ZCRMSNorm_2"]["scale"][layer_idx])
        ff = p["FeedForward_0"]
        block.ffn.gate_proj.weight.data = _t(ff["gate_proj"]["kernel"][layer_idx]).T
        block.ffn.up_proj.weight.data = _t(ff["up_proj"]["kernel"][layer_idx]).T
        block.ffn.down_proj.weight.data = _t(ff["down_proj"]["kernel"][layer_idx]).T


def build_encoder_from_params(params: dict, cfg: dict) -> NeedleEncoderONNX:
    vocab_size = cfg.get("vocab_size", 8192)
    d_model = cfg.get("d_model", 512)
    num_heads = cfg.get("num_heads", 8)
    num_kv_heads = cfg.get("num_kv_heads", 4)
    d_ff = cfg.get("d_ff", 2048)
    num_enc_layers = cfg.get("num_encoder_layers", 12)
    activation = cfg.get("activation", "drelu")
    no_feedforward = cfg.get("no_feedforward", True)
    max_seq_len = cfg.get("max_seq_len", 8192)
    rope_theta = cfg.get("rope_theta", 10000.0)

    model = NeedleEncoderONNX(vocab_size, d_model, num_heads, num_kv_heads,
                               d_ff, num_enc_layers, activation, no_feedforward,
                               max_seq_len, rope_theta)

    # Embeddings
    model.embedding.weight.data = _t(params["embedding"]["embedding"])

    # Encoder layer params (stacked by nn.scan)
    enc_layers_p = params["encoder"]["layers"]["EncoderBlock_0"]
    for i, layer in enumerate(model.encoder.layers):
        load_enc_block_weights(layer, enc_layers_p, i)

    model.encoder.final_norm.scale.data = _t(params["encoder"]["final_norm"]["scale"])
    return model


def build_decoder_from_params(params: dict, cfg: dict,
                               shared_embedding: torch.Tensor) -> NeedleDecoderONNX:
    vocab_size = cfg.get("vocab_size", 8192)
    d_model = cfg.get("d_model", 512)
    num_heads = cfg.get("num_heads", 8)
    num_kv_heads = cfg.get("num_kv_heads", 4)
    d_ff = cfg.get("d_ff", 2048)
    num_dec_layers = cfg.get("num_decoder_layers", 8)
    activation = cfg.get("activation", "drelu")
    no_feedforward = cfg.get("no_feedforward", True)
    max_seq_len = cfg.get("max_seq_len", 8192)
    rope_theta = cfg.get("rope_theta", 10000.0)

    model = NeedleDecoderONNX(vocab_size, d_model, num_heads, num_kv_heads,
                               d_ff, num_dec_layers, activation, no_feedforward,
                               max_seq_len, rope_theta, shared_embedding)

    dec_layers_p = params["decoder"]["layers"]["DecoderBlock_0"]
    for i, layer in enumerate(model.decoder.layers):
        load_dec_block_weights(layer, dec_layers_p, i)

    model.decoder.final_norm.scale.data = _t(params["decoder"]["ZCRMSNorm_0"]["scale"])
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_with_onnxruntime(onnx_path: str, pt_model: nn.Module,
                               *dummy_inputs) -> float:
    """Run ONNX and PyTorch on same inputs; return max absolute difference."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    ort_inputs = {}
    for name, inp in zip(input_names, dummy_inputs):
        arr = inp.numpy()
        if arr.dtype == np.int32:
            ort_inputs[name] = arr
        else:
            ort_inputs[name] = arr.astype(np.float32)

    ort_out = sess.run(None, ort_inputs)[0]

    with torch.no_grad():
        pt_out = pt_model(*dummy_inputs).numpy()

    diff = np.abs(ort_out.astype(np.float32) - pt_out.astype(np.float32)).max()
    return float(diff)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Download checkpoint ---
    from huggingface_hub import hf_hub_download
    if args.checkpoint:
        ckpt_path = args.checkpoint
        print(f"Using local checkpoint: {ckpt_path}")
    else:
        print("Downloading checkpoint from HuggingFace Hub...")
        ckpt_path = hf_hub_download(
            repo_id="Cactus-Compute/needle",
            filename="needle.pkl",
            repo_type="model",
        )
        print(f"Checkpoint: {ckpt_path}")

    # --- Load params ---
    print("Loading checkpoint...")
    params, cfg = load_jax_params(ckpt_path)
    print(f"Config: {cfg}")

    # Print param tree to verify structure
    import jax, jax.numpy as jnp
    flat = jax.tree_util.tree_map(lambda x: np.array(x).shape, params)
    print("\nParam tree (first 40 entries):")
    leaves = list(jax.tree_util.tree_leaves_with_path(flat))[:40]
    for path, shape in leaves:
        print(f"  {path}: {shape}")

    # --- Build PyTorch models ---
    print("\nBuilding PyTorch encoder...")
    encoder = build_encoder_from_params(params, cfg)
    encoder.eval()
    shared_emb = encoder.embedding.weight.data.clone()

    print("Building PyTorch decoder...")
    decoder = build_decoder_from_params(params, cfg, shared_emb)
    decoder.eval()

    d_model = cfg.get("d_model", 512)
    vocab_size = cfg.get("vocab_size", 8192)
    enc_len = 64
    dec_len = 16

    # --- Export encoder ---
    enc_suffix = "_fp16" if args.fp16 else ""
    enc_path_f32 = output_dir / "needle_encoder.onnx"
    enc_path = output_dir / f"needle_encoder{enc_suffix}.onnx"

    dummy_enc_ids = torch.randint(1, vocab_size, (1, enc_len), dtype=torch.int32)
    print(f"\nExporting encoder to {enc_path_f32}...")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (dummy_enc_ids,),
            str(enc_path_f32),
            input_names=["encoder_input_ids"],
            output_names=["encoder_hidden_states"],
            dynamic_axes={
                "encoder_input_ids": {0: "batch", 1: "enc_len"},
                "encoder_hidden_states": {0: "batch", 1: "enc_len"},
            },
            opset_version=18,  # torch 2.x uses 18; onnxruntime-web 1.20+ supports it
            do_constant_folding=True,
            dynamo=False,  # use legacy TorchScript exporter; dynamo exporter doesn't accept dynamic_axes
        )

    # --- Export decoder ---
    dec_path_f32 = output_dir / "needle_decoder.onnx"
    dec_path = output_dir / f"needle_decoder{enc_suffix}.onnx"

    dummy_dec_ids = torch.randint(1, vocab_size, (1, dec_len), dtype=torch.int32)
    dummy_enc_hidden = torch.randn(1, enc_len, d_model)
    print(f"Exporting decoder to {dec_path_f32}...")
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (dummy_dec_ids, dummy_enc_hidden),
            str(dec_path_f32),
            input_names=["decoder_input_ids", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "decoder_input_ids": {0: "batch", 1: "dec_len"},
                "encoder_hidden_states": {0: "batch", 1: "enc_len"},
                "logits": {0: "batch", 1: "dec_len"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    # --- Optional: convert to float16 ---
    if args.fp16:
        print("Converting to float16...")
        import onnx
        from onnxconverter_common import float16

        enc_model_f32 = onnx.load(str(enc_path_f32))
        enc_fp16 = float16.convert_float_to_float16(enc_model_f32, keep_io_types=False)
        onnx.save(enc_fp16, str(enc_path))
        os.remove(enc_path_f32)

        dec_model_f32 = onnx.load(str(dec_path_f32))
        dec_fp16 = float16.convert_float_to_float16(dec_model_f32, keep_io_types=False)
        onnx.save(dec_fp16, str(dec_path))
        os.remove(dec_path_f32)
    else:
        enc_path = enc_path_f32
        dec_path = dec_path_f32

    enc_size = enc_path.stat().st_size / 1e6
    dec_size = dec_path.stat().st_size / 1e6
    print(f"\nEncoder: {enc_path} ({enc_size:.1f} MB)")
    print(f"Decoder: {dec_path} ({dec_size:.1f} MB)")

    # --- Validate with onnxruntime ---
    if args.validate:
        print("\nValidating encoder...")
        enc_diff = validate_with_onnxruntime(str(enc_path), encoder, dummy_enc_ids)
        print(f"  Encoder max abs diff: {enc_diff:.6f}")

        with torch.no_grad():
            dummy_enc_out = encoder(dummy_enc_ids)
        print("Validating decoder...")
        dec_diff = validate_with_onnxruntime(str(dec_path), decoder,
                                              dummy_dec_ids, dummy_enc_out)
        print(f"  Decoder max abs diff: {dec_diff:.6f}")

        tol = 0.05 if args.fp16 else 1e-3
        assert enc_diff < tol, f"Encoder diff {enc_diff} exceeds tolerance {tol}"
        assert dec_diff < tol, f"Decoder diff {dec_diff} exceeds tolerance {tol}"
        print("Validation passed.")

    # --- Export tokenizer ---
    tok_out_dir = output_dir / "tokenizer"
    tok_out_dir.mkdir(exist_ok=True)
    print("\nDownloading tokenizer...")
    for fname in ["tokenizer/needle.model", "tokenizer/needle.vocab"]:
        local = hf_hub_download(
            repo_id="Cactus-Compute/needle",
            filename=fname,
            repo_type="model",
        )
        dest = tok_out_dir / Path(fname).name
        shutil.copy2(local, dest)
        print(f"  Saved {dest}")

    print("\nExport complete.")
    print(f"  Encoder: {enc_path}")
    print(f"  Decoder: {dec_path}")
    print(f"  Tokenizer: {tok_out_dir}/needle.model")


def main():
    parser = argparse.ArgumentParser(description="Export Needle model to ONNX")
    parser.add_argument("--output-dir", default="../models",
                        help="Directory for output files (default: ../models)")
    parser.add_argument("--fp16", action="store_true",
                        help="Convert to float16 after export (smaller, faster)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate ONNX output against PyTorch")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to local checkpoint.pkl (skips HF download)")
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
