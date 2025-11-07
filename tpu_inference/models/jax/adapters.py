import typing as tp
from typing import TYPE_CHECKING

import io

import torch
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh

from tpu_inference.layers.jax.pool.pooler import (Pooler,
                                                  SentenceTransformerDenseLayer,
                                                  SentenceTransformerProjector)
from tpu_inference.logger import init_logger
from vllm.config import VllmConfig
from vllm.transformers_utils.config import (get_hf_file_bytes,
                                            try_get_dense_modules)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_T = tp.TypeVar("_T", bound=type[nnx.Module])

_GENERATE_SUFFIXES = (
    "ForCausalLM",
    "ForConditionalGeneration",
)

_TORCH_TO_JAX_DTYPE = {
    torch.float16: jnp.float16,
    torch.bfloat16: jnp.bfloat16,
    torch.float32: jnp.float32,
    torch.float64: jnp.float64,
}


def _torch_dtype_to_jax(dtype: torch.dtype | None) -> jnp.dtype:
    if dtype is None:
        return jnp.float32

    mapped = _TORCH_TO_JAX_DTYPE.get(dtype)
    if mapped is None:
        logger.warning(
            "Unsupported projector dtype %s; falling back to float32.", dtype
        )
        return jnp.float32
    return mapped


def _normalize_activation_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().lower()
    return normalized or None


def _load_dense_weights(
    folder: str,
    model_config: "ModelConfig",
) -> tuple[jnp.ndarray, jnp.ndarray | None] | None:
    filenames = ("model.safetensors", "pytorch_model.bin")
    for filename in filenames:
        file_path = f"{folder}/{filename}" if folder else filename
        try:
            file_bytes = get_hf_file_bytes(
                file_path,
                model_config.model,
                model_config.revision,
            )
        except Exception:
            logger.exception("Failed to fetch %s", file_path)
            continue

        if not file_bytes:
            continue

        try:
            if filename.endswith(".safetensors"):
                from safetensors.torch import load as load_safetensors

                state_dict = load_safetensors(file_bytes)
            else:
                state_dict = torch.load(
                    io.BytesIO(file_bytes), map_location="cpu", weights_only=True
                )
        except Exception:
            logger.exception("Failed to load %s", file_path)
            continue

        for weight_key in ("weight", "linear.weight", "dense.weight"):
            if weight_key not in state_dict:
                continue

            weight = state_dict[weight_key].to(torch.float32).cpu().numpy()
            bias_key = weight_key.replace("weight", "bias")
            bias = None
            if bias_key in state_dict:
                bias = state_dict[bias_key].to(torch.float32).cpu().numpy()
            return jnp.asarray(weight), jnp.asarray(bias) if bias is not None else None

    return None


def _load_st_projector(
    model_config: "ModelConfig",
) -> SentenceTransformerProjector | None:
    dense_modules = try_get_dense_modules(
        model_config.model,
        revision=model_config.revision,
    )

    if not dense_modules:
        return None

    dtype = _torch_dtype_to_jax(getattr(model_config, "head_dtype", None))
    layers: list[SentenceTransformerDenseLayer] = []

    for layer_config in dense_modules:
        folder = layer_config.get("folder", "")
        dense_params = _load_dense_weights(folder, model_config)
        if dense_params is None:
            continue
        weight, bias = dense_params
        activation = _normalize_activation_name(
            layer_config.get("activation_function")
        )
        layers.append(
            SentenceTransformerDenseLayer(
                weight=jnp.asarray(weight, dtype=dtype),
                bias=jnp.asarray(bias, dtype=dtype) if bias is not None else None,
                activation=activation,
            )
        )

    if not layers:
        return None

    return SentenceTransformerProjector(tuple(layers))

logger = init_logger(__name__)

class PoolingMixin:
    """
    same as VllmModelForPooling 
    """
    is_pooling_model: tp.ClassVar[tp.Literal[True]] = True

    default_pooling_type: tp.ClassVar[str] = "LAST"
    pooler: Pooler


def _get_pooling_model_name(orig_model_name: str, pooling_suffix: str) -> str:
    model_name = orig_model_name
    for suffix in _GENERATE_SUFFIXES:
        model_name = model_name.removesuffix(suffix)
    return model_name + pooling_suffix


def _create_pooling_model_cls(orig_cls: _T) -> _T:
    class ModelForPooling(orig_cls, PoolingMixin): 
        is_pooling_model = True

        def __init__(
            self,
            vllm_config: VllmConfig,
            rng_key: jax.Array,
            mesh: Mesh,
        ) -> None:
            super().__init__(
                vllm_config=vllm_config,
                rng_key=rng_key,
                mesh=mesh,
            )


            # Pooling models do not require language modeling heads.
            # However, there is a problem: since the pattern for loading weights in nnx
            # is abstract_module -> module, removing the lm_head attribute or leaves from the abstract_module
            # results in an error, I think.
            # This is because, during hf_load_weights, we need to match between the hf_key and nnx_key.

            # for attr in ("model.lm_head"):
            #     if hasattr(self, attr):
            #         delattr(self, attr)

            if getattr(self, "pooler", None) is None:
                self._init_pooler(vllm_config=vllm_config)

        def _init_pooler(self, vllm_config: VllmConfig) -> None: 
            raise NotImplementedError

    return ModelForPooling 


def as_embedding_model(cls: _T) -> _T:

    class ModelForEmbedding(_create_pooling_model_cls(cls)):
        def _init_pooler(self, vllm_config: VllmConfig) -> None:
            pooler_config = vllm_config.model_config.pooler_config
            if pooler_config is None:
                raise ValueError(
                    "Embedding models require `pooler_config` to be set in the model configuration."
                )

            model_config = vllm_config.model_config
            head_dtype = _torch_dtype_to_jax(getattr(model_config, "head_dtype", None))
            projector = _load_st_projector(model_config)

            self.pooler = Pooler.for_embed(
                pooler_config,
                head_dtype=head_dtype,
                st_projector=projector,
            )

    ModelForEmbedding.__name__ = _get_pooling_model_name(
        cls.__name__,
        "ForEmbedding",
    )
    return ModelForEmbedding  # type: ignore[return-value]



def init_pooler_from_vllm_model(
        vllm_model: torch.nn.Module,
        vllm_config: VllmConfig,
        rng_key: PRNGKey, 
        mesh: Mesh,
):
    class DummyModule:
        def __init__(self, vllm_config, rng_key, mesh):
            pass

    for suffix in _GENERATE_SUFFIXES:
        if suffix in vllm_model.__class__.__name__:
            return None

    if "ForEmbedding" in vllm_model.__class__.__name__:
        EmbedModel = as_embedding_model(DummyModule)

        embed_model = EmbedModel(vllm_config=vllm_config, rng_key=rng_key, mesh=mesh,)
        embed_model._init_pooler(vllm_config)
        return embed_model.pooler 
    else:
        raise NotImplementedError(
            f"Pooling initialization for {vllm_model.__class__.__name__} is not implemented."
        )
