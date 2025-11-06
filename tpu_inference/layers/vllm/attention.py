# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Optional, Tuple

import jax
import torch
from jax.sharding import Mesh
import jax.numpy as jnp
from torchax.interop import jax_view, torch_view
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention_interface import attention
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


class DummyPallasAttentionMetadataBuilder:
    def  build(
        self,
        common_prefix_len, 
        common_attn_metadata,
        fast_build,
    ):
        pass


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        return DummyPallasAttentionMetadataBuilder 

class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        use_irope: bool = False,
    ) -> None:
        if use_irope:
            logger.warning_once(
                "Using irope in Pallas is not supported yet, it will fall back "
                "to global attention for long context.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.attn_type = attn_type

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

        # if attn_type != AttentionType.DECODER:
        #     raise NotImplementedError("Encoder self-attention and "
        #                               "encoder/decoder cross-attention "
        #                               "are not implemented for "
        #                               "PallasAttentionBackendImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "PallasAttentionBackendImpl")

        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        mesh = vllm_model_wrapper_context.mesh

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            query, key, value = jax_view(query), jax_view(key), jax_view(value)
            return _jax_encoder_attn_func(
                query,
                key,
                value,
                attn_metadata,
                mesh,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                self.scale,
            )

        if kv_cache.numel():
            raise RuntimeError(
                "KV cache from vLLM Attention layer should be empty but has "
                "the size of %s.", kv_cache.numel())

        del kv_cache  # Use kv_cache from vllm wrapper context values instead.

        layer_to_cache = (
            vllm_model_wrapper_context.layer_name_to_kvcache_index or {}
        )
        kv_cache_index = layer_to_cache.get(layer.layer_name)

        if kv_cache_index is None:
            raise KeyError(
                f"Layer {layer.layer_name} not found in KV cache mapping"
            )

        kv_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]

        query, key, value = jax_view(query), jax_view(key), jax_view(value)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            key, value = utils.quantize_kv(key, value,
                                           self.kv_cache_quantized_dtype,
                                           layer._k_scale_float,
                                           layer._v_scale_float)
            # TODO(kyuyeunk): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = layer._q_scale_float
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float

        new_kv_cache, outputs = _jax_attn_func(kv_cache, query, key, value,
                                               attn_metadata, mesh, self.scale,
                                               self.head_size, self.num_heads,
                                               self.num_kv_heads, q_scale,
                                               k_scale, v_scale)
        vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

        return torch_view(outputs)


@functools.partial(
    jax.jit,
    static_argnums=(
        5, 6, 7, 8, 9, 10, 11, 12
    ),  # mesh, scale, head_size, num_heads, num_kv_heads, q_scale, k_scale, v_scale
    donate_argnums=(0, ),  # donate kv_cache
)
def _jax_attn_func(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    scale: float,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    del scale  # Unused for now, as the attention function applies a default scale.

    # Get shapes from vllm
    q_len, q_compute_dim = q.shape
    k_len, k_compute_dim = k.shape
    assert k.shape == v.shape
    assert q_compute_dim == head_size * num_heads
    assert k_compute_dim == head_size * num_kv_heads

    # Convert the shapes from vLLM's convetion to what the attention function expects
    # bs, num_heads, q_len, head_size
    q = q.reshape(q_len, num_heads, head_size)
    # bs, num_kv_heads, k_len, head_size
    k = k.reshape(k_len, num_kv_heads, head_size)
    v = v.reshape(k_len, num_kv_heads, head_size)

    new_kv_cache, outputs = attention(
        kv_cache,
        q,
        k,
        v,
        attention_metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    # Convert the shape back to vLLM's convention
    assert outputs.shape[0] == q_len
    assert outputs.shape[1] == num_heads
    assert outputs.shape[2] == head_size
    outputs = outputs.reshape(q_len, q_compute_dim)

    return new_kv_cache, outputs

def build_full_attention_mask(
    attention_metadata: AttentionMetadata
) -> jax.Array:
    seq_lens = attention_metadata.seq_lens
    if seq_lens is None:
        raise ValueError("attention metadata is missing seq_lens")

    doc_ids = jnp.repeat(
        jnp.arange(seq_lens.shape[0], dtype=jnp.int32),
        seq_lens,
    )

    mask = doc_ids[:, None] == doc_ids[None, :]
    return mask 


@functools.partial(
    jax.jit,
    static_argnums = (3, 4, 5, 6, 7, 8 )
)
def _jax_encoder_attn_func(
    query: jax.Array,
    key: jax.Array, 
    value: jax.Array, 
    attn_metadata: AttentionMetadata,
    mesh: Mesh,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
) -> jax.Array: 

    q_len, q_compute_dim = q.shape
    k_len, k_compute_dim = k.shape
    assert k.shape == v.shape
    assert q_compute_dim == head_size * num_heads
    assert k_compute_dim == head_size * num_kv_heads
    assert q_len == k_len == v.shape[0]

    q = q.reshape((-1, num_heads, head_dim))
    k = k.reshape((-1, num_kv_heads, head_dim))
    v = v.reshape((-1, num_kv_heads, head_dim))
    mask = build_full_mask(attention_metadata)

    assert mask.shape[0] == q.shape[0]

    qkv_spec = P(None, "model", None)

    @functools.partial(
        jax.shard_map,
        in_specs = (qkv_spec, qkv_spec, qkv_spec, qkv_spec),
        out_specs = (qkv_spec)
    )
    def _jax_nn_dot_product_attention(q, k, v, mask):
        return jax.nn.dot_product_attention(q, k, v, mask = mask)

    outputs = _jax_nn_dot_product_attention(q, k, v , mask)

    outputs = outputs.reshape((-1, num_heads * head_size))
    return outputs
