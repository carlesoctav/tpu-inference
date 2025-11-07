import enum
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from tpu_inference.layers.jax.pool.pooling_metadata import TPUSupportedPoolingMetadata, is_partial_prefill

from vllm.config.pooler import PoolerConfig


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("weight", "bias"),
    meta_fields=("activation",),
)
@dataclass
class SentenceTransformerDenseLayer:
    """Single dense layer used by Sentence-Transformers projector."""

    weight: jax.Array
    bias: jax.Array | None
    activation: str | None = None


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("layers",),
    meta_fields=(),
)
@dataclass
class SentenceTransformerProjector:
    """Lightweight sequential projector built from ST Dense layers."""

    layers: tuple[SentenceTransformerDenseLayer, ...]

    def __call__(self, inputs: jax.Array) -> jax.Array:
        outputs = inputs
        for layer in self.layers:
            weight = layer.weight
            bias = layer.bias
            outputs = outputs @ weight.T
            if bias is not None:
                outputs = outputs + bias
            outputs = _apply_st_activation(outputs, layer.activation)
        return outputs


# [padded_num_reqs, dim]
# or [padded_num_reqs, padded_max_num_batchec_token_per_req, dim] for allpool
PoolerOutput = jax.Array


class PoolingType(enum.Enum):
    LAST = "LAST"
    MEAN = "MEAN"
    CLS = "CLS"
    ALL = "ALL"


@dataclass(frozen=True)
class ResolvedPoolingConfig:
    task: str
    pooling_type: PoolingType
    normalize: bool

    @classmethod
    def from_config(
        cls,
        task: str,
        pooler_config: PoolerConfig | None,
    ) -> "ResolvedPoolingConfig":
        pooler_config = pooler_config or PoolerConfig()

        # The encode functionality is currently disabled because we cannot use DispatchPooler
        # as intended. (It was part of ModelForEmbedding, and in newer versions it was renamed to token_embed.)
        # This is because TPU does not support alternating requests between these two tasks, and it is
        # out of scope to change the vllm request handler/API server to separate these requests.
        # Therefore, this is disabled by default—users cannot use token_embed/encode functionality for now.

        if task == "embed":
            default_pooling_type = PoolingType.LAST
            default_normalize = True
        elif task == "encode":
            raise ValueError(f"Unsupported pooling task: {task}")
        else:
            raise ValueError(f"Unsupported pooling task: {task}")

        pooling_type_str = pooler_config.pooling_type or default_pooling_type.name
        pooling_type = PoolingType(pooling_type_str.upper())
        normalize = (
            pooler_config.normalize
            if pooler_config.normalize is not None
            else default_normalize
        )

        return cls(task=task, pooling_type=pooling_type, normalize=normalize)


class PoolingMethod(nnx.Module):
    @staticmethod
    def from_pooling_type(pooling_type: PoolingType) -> "PoolingMethod":
        if pooling_type is PoolingType.ALL:
            raise NotImplementedError("ALL pooling is not implemented yet.")
            # return AllPoolingMethod()
        if pooling_type is PoolingType.MEAN:
            return MeanPoolingMethod()
        if pooling_type is PoolingType.LAST:
            return LastPoolingMethod()
        if pooling_type is PoolingType.CLS:
            return CLSPoolingMethod()
        raise NotImplementedError(f"Unsupported pooling type: {pooling_type}")

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        raise NotImplementedError


class AllPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        pass


class MeanPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        padded_prompt_lens = pooling_metadata.prompt_lens
        padded_start_indices = pooling_metadata.first_token_indices
        padded_end_indices = pooling_metadata.last_token_indices
        cumsum = jnp.cumsum(hidden_states, axis=0, dtype=jnp.float32)

        return (
            cumsum[padded_end_indices]
            - cumsum[padded_start_indices]
            + hidden_states[padded_start_indices]
        ) / padded_prompt_lens[:, None]


class LastPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        return hidden_states[pooling_metadata.last_token_indices]


class CLSPoolingMethod(PoolingMethod):
    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> jax.Array:
        return hidden_states[pooling_metadata.first_token_indices]


class PoolerHead(nnx.Module):
    def __call__(
        self,
        pooled: jax.Array,
        token_embeddings: jax.Array,
        token_mask: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


class EmbeddingPoolerHead(PoolerHead):
    def __init__(
        self,
        default_normalize: bool,
        head_dtype: jnp.dtype,
        projector: SentenceTransformerProjector | None = None,
    ) -> None:
        super().__init__()
        self.default_normalize = default_normalize
        self.head_dtype = head_dtype
        self.projector = projector

    def __call__(
        self,
        pooled: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:
        pooled = pooled.astype(self.head_dtype)

        if self.projector is not None:
            pooled = self.projector(pooled)

        if self.default_normalize:
            pooled = normalize(pooled)

        return pooled


class Pooler(nnx.Module):
    @staticmethod
    def for_encode(pooler_config: PoolerConfig | None) -> "Pooler":
        resolved = ResolvedPoolingConfig.from_config("encode", pooler_config)
        raise NotImplementedError("EncodePooler is currently disabled.")

    @staticmethod
    def for_embed(
        pooler_config: PoolerConfig | None,
        head_dtype: jnp.dtype | None = None,
        st_projector: SentenceTransformerProjector | None = None,
    ) -> "Pooler":
        resolved = ResolvedPoolingConfig.from_config("embed", pooler_config)
        dtype = head_dtype or jnp.float32
        return EmbeddingPooler.from_config(resolved, dtype, st_projector)

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError

    def get_supported_tasks(self) -> set[str]:
        raise NotImplementedError


class EmbeddingPooler(Pooler):
    def __init__(
        self,
        pooling: PoolingMethod,
        head: EmbeddingPoolerHead,
    ) -> None:
        self.pooling = pooling
        self.head = head

    @classmethod
    def from_config(
        cls,
        config: ResolvedPoolingConfig,
        head_dtype: jnp.dtype,
        projector: SentenceTransformerProjector | None,
    ) -> "EmbeddingPooler":
        pooling = PoolingMethod.from_pooling_type(config.pooling_type)
        head = EmbeddingPoolerHead(config.normalize, head_dtype, projector)
        return cls(pooling, head)

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: TPUSupportedPoolingMetadata,
    ) -> PoolerOutput:
        hidden_states = hidden_states.astype(jnp.float32)
        # the output mus be of type torch.tensor, but we cannot convert numpy to torch if the dtype is bf16
        pooled = self.pooling(hidden_states, pooling_metadata)
        return self.head(pooled, pooling_metadata)

    def get_supported_tasks(self) -> set[str]:
        return ("embed",)


def _apply_st_activation(x: jax.Array, activation: str | None) -> jax.Array:
    if activation is None:
        return x

    name = activation.lower()
    if name in ("identity", "linear"):
        return x
    if name == "relu":
        return jax.nn.relu(x)
    if name == "tanh":
        return jnp.tanh(x)
    if name == "sigmoid":
        return jax.nn.sigmoid(x)
    if name in ("silu", "swish"):
        return jax.nn.silu(x)
    if name in ("gelu", "gelu_new"):
        approximate = name == "gelu_new"
        return jax.nn.gelu(x, approximate=approximate)
    if name == "softmax":
        return jax.nn.softmax(x, axis=-1)
    if name in ("leakyrelu", "leaky_relu"):
        return jax.nn.leaky_relu(x)
    return x


def normalize(embeddings: jax.Array) -> jax.Array:
    norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = jnp.maximum(norms, 1e-12)
    normalized = embeddings / norms
    return normalized
