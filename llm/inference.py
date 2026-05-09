"""
LLM backends: Hugging Face 4-bit Llama (BitsAndBytes), local Ollama (e.g. Llama 3.2 on Colab), or stub.
"""

from __future__ import annotations

import abc
import logging
import os
import sys

logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def ollama_generate(
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """
    Blocking **HTTP** call to local Ollama **`POST /api/generate`**.
    Prefer **`answer_with_rag`** / **`/api/chat`** when you need a system+RAG turns.
    """
    import requests

    base = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    mod = model or os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "600"))
    r = requests.post(
        f"{base}/api/generate",
        json={
            "model": mod,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_new_tokens, "temperature": temperature},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    text = data.get("response")
    if not text:
        raise RuntimeError(f"Unexpected Ollama /api/generate response: {data}")
    return str(text).strip()


_DEFAULT_MODEL = os.environ.get(
    "HF_MODEL_ID",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)


class BaseLLMEngine(abc.ABC):
    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def ensure_loaded(self) -> None: ...

    @abc.abstractmethod
    def build_prompt(self, query: str, contexts: list[str]) -> str: ...

    @abc.abstractmethod
    def run_llm(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str: ...

    def answer_with_rag(
        self,
        query: str,
        contexts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        prompt = self.build_prompt(query, contexts)
        return self.run_llm(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


class StubLlamaEngine(BaseLLMEngine):
    def load(self) -> None:
        logger.warning("LLM_USE_STUB=1: deterministic stub generation")

    def ensure_loaded(self) -> None:
        pass

    def build_prompt(self, query: str, contexts: list[str]) -> str:
        ctx_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no context retrieved)"
        system_text = (
            "You are a helpful assistant. Answer using the context when it is relevant.\n\n"
            f"Context:\n{ctx_block}"
        )
        return f"[system]\n{system_text}\n\n[user]\n{query}\n\n[assistant]\n"

    def run_llm(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        return (
            f"[stub-llm] {max_new_tokens} max tokens. "
            f"Preview: {prompt[:200]!r}..."
        )


class HuggingFaceLlamaEngine(BaseLLMEngine):
    """4-bit Llama via transformers + bitsandbytes."""

    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = None,
    ) -> None:
        self.model_id = model_id or _DEFAULT_MODEL
        self.device_map = device_map or os.environ.get("LLM_DEVICE_MAP", "auto")
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            logger.warning("CUDA not available; set LLM_USE_STUB=1 or LLM_BACKEND=ollama")
            raise RuntimeError("Hugging Face quantized Llama expects CUDA unless using stub/Ollama")

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

    def ensure_loaded(self) -> None:
        if self._model is None:
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
        assert self._tokenizer is not None
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
        assert self._tokenizer is not None and self._model is not None

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


class OllamaEngine(BaseLLMEngine):
    """Llama via Ollama HTTP API (/api/chat), e.g. `llama3.2:1b` on Colab."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
        self._timeout_sec = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "600"))

    def load(self) -> None:
        try:
            import requests

            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            logger.info("Ollama reachable at %s; model=%s", self.base_url, self.model)
        except Exception as e:
            logger.warning("Ollama probe failed (%s): is `ollama serve` running?", e)

    def ensure_loaded(self) -> None:
        pass

    def _chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        import requests

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        }
        r = requests.post(
            f"{self.base_url}/api/chat",
            json=body,
            timeout=self._timeout_sec,
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        if not content:
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return str(content).strip()

    def build_prompt(self, query: str, contexts: list[str]) -> str:
        ctx_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no context retrieved)"
        return (
            "You are a helpful assistant. Answer using the context when it is relevant.\n\n"
            f"Context:\n{ctx_block}\n\nUser question:\n{query}"
        )

    def run_llm(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        return ollama_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            base_url=self.base_url,
            model=self.model,
        )

    def answer_with_rag(
        self,
        query: str,
        contexts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        ctx_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no context retrieved)"
        system_text = (
            "You are a helpful assistant. Answer using the context when it is relevant.\n\n"
            f"Context:\n{ctx_block}"
        )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": query},
        ]
        return self._chat(messages, max_new_tokens, temperature)


# Back-compat alias used in docs / older references
LlamaEngine = HuggingFaceLlamaEngine

_ENGINE: BaseLLMEngine | None = None


def get_engine() -> BaseLLMEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    stub = os.environ.get("LLM_USE_STUB", "").lower() in ("1", "true", "yes")
    if stub:
        _ENGINE = StubLlamaEngine()
        _ENGINE.load()
        return _ENGINE

    backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
    if backend in ("ollama", "local", "ollama-http"):
        _ENGINE = OllamaEngine()
        _ENGINE.load()
        return _ENGINE

    _ENGINE = HuggingFaceLlamaEngine()
    return _ENGINE


def run_llm(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Default engine inference. With **`LLM_BACKEND=ollama`** (default), this is an Ollama **HTTP** `/api/generate` call via **`ollama_generate`**.
    """
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
    "BaseLLMEngine",
    "StubLlamaEngine",
    "HuggingFaceLlamaEngine",
    "OllamaEngine",
    "LlamaEngine",
    "ollama_generate",
    "run_llm",
    "build_rag_prompt",
    "answer_with_rag",
    "get_engine",
]
