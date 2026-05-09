"""
4-bit quantized Llama-3 inference via transformers + bitsandbytes (optional CPU stub).
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DEFAULT_MODEL = os.environ.get(
    "HF_MODEL_ID",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)


class LlamaEngine:
    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = None,
    ) -> None:
        self.model_id = model_id or _DEFAULT_MODEL
        self.device_map = device_map or os.environ.get("LLM_DEVICE_MAP", "auto")
        self._tokenizer = None
        self._model = None
        self._use_stub = os.environ.get("LLM_USE_STUB", "").lower() in ("1", "true", "yes")

    def load(self) -> None:
        if self._use_stub:
            logger.warning("LLM_USE_STUB=1: using deterministic stub generation")
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            logger.warning("CUDA not available; set LLM_USE_STUB=1 for CPU dev or use GPU runtime")
            raise RuntimeError("Quantized Llama inference expects CUDA (e.g. Colab GPU)")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        logger.info("Loaded 4-bit model %s", self.model_id)

    def _stub_generate(self, prompt: str, max_new_tokens: int) -> str:
        return (
            f"[stub-llm] {max_new_tokens} max tokens. "
            f"Preview: {prompt[:200]!r}..."
        )

    def ensure_loaded(self) -> None:
        if self._model is None and not self._use_stub:
            self.load()

    def build_prompt(self, query: str, contexts: list[str]) -> str:
        self.ensure_loaded()
        ctx_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no context retrieved)"
        system_text = (
            "You are a helpful assistant. Answer using the context when it is relevant.\n\n"
            f"Context:\n{ctx_block}"
        )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": query},
        ]
        if self._use_stub or self._tokenizer is None:
            return f"[system]\n{system_text}\n\n[user]\n{query}\n\n[assistant]\n"
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{system_text}\n\nUser: {query}\nAssistant:"

    def run_llm(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        self.ensure_loaded()
        if self._use_stub or self._model is None:
            return self._stub_generate(prompt, max_new_tokens)

        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        gen_kw: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature and temperature > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        else:
            gen_kw["do_sample"] = False

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kw)
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt) :].lstrip()
        return text

    def answer_with_rag(
        self,
        query: str,
        contexts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        prompt = self.build_prompt(query, contexts)
        return self.run_llm(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


_ENGINE: LlamaEngine | None = None


def get_engine() -> LlamaEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = LlamaEngine()
    return _ENGINE


def run_llm(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    return get_engine().run_llm(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


def build_rag_prompt(query: str, contexts: list[str]) -> str:
    return get_engine().build_prompt(query, contexts)


def answer_with_rag(
    query: str,
    contexts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    return get_engine().answer_with_rag(
        query,
        contexts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


__all__ = [
    "LlamaEngine",
    "run_llm",
    "build_rag_prompt",
    "answer_with_rag",
    "get_engine",
]
