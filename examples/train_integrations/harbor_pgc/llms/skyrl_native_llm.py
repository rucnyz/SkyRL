"""Harbor LLM backend that talks SkyRL's vllm-router /skyrl/v1/generate.

Why: harbor's default LiteLLM backend posts to /v1/chat/completions, but the
new SkyRL inference path (vllm-router) doesn't preserve vllm's OpenAI extras
(prompt_token_ids / completion_token_ids / logprobs) needed for step-wise RL.
The router's /skyrl/v1/generate endpoint does preserve them — but it expects
pre-tokenized token_ids, so we tokenize chat messages on the client side here
and decode the response token_ids back to text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer

from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.models.metric import UsageInfo


# Sentinel max output tokens when no cap is configured.
_DEFAULT_MAX_OUTPUT_TOKENS = 16384


class SkyRLNativeLLM(BaseLLM):
    """Harbor LLM backend that calls SkyRL's vllm-router /skyrl/v1/generate.

    Args:
        model_name: served model name on the vllm engines (e.g. "Qwen3.5-9B").
            Note: harbor convention is "hosted_vllm/<served_name>"; we strip the
            "hosted_vllm/" prefix to recover the bare name vllm expects.
        proxy_url: vllm-router URL (data plane) — e.g. "http://127.0.0.1:54567".
        tokenizer_path: HF model path used to apply the chat template + tokenize.
            Must be the same family/tokenizer the served vllm model was loaded with.
        temperature: sampling temperature.
        model_info: dict with max_input_tokens / max_output_tokens (passed through
            from harbor agent kwargs).
        max_output_tokens: per-call cap (defaults to model_info["max_output_tokens"]).
        llm_kwargs: ignored extra kwargs (harbor passes LiteLLM-specific ones; we
            silently swallow them to keep the trial config schema-compatible).
    """

    def __init__(
        self,
        model_name: str,
        proxy_url: str,
        tokenizer_path: str,
        temperature: float = 1.0,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        model_info: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        timeout_sec: float = 900.0,
        **llm_kwargs: Any,
    ):
        super().__init__()
        # Strip the harbor "hosted_vllm/" prefix — vllm sees just the bare name.
        bare = model_name.removeprefix("hosted_vllm/")
        self._model_name = bare
        self._proxy_url = proxy_url.rstrip("/")
        self._temperature = temperature
        self._collect_rollout_details = collect_rollout_details
        self._session_id = session_id
        self._model_info = model_info or {}
        self._max_output_tokens = (
            max_output_tokens
            or self._model_info.get("max_output_tokens")
            or _DEFAULT_MAX_OUTPUT_TOKENS
        )
        self._max_input_tokens = self._model_info.get("max_input_tokens")
        self._timeout_sec = timeout_sec
        # Sampling params bag (only forward what /skyrl/v1/generate knows about).
        self._extra_sampling = {
            k: llm_kwargs[k]
            for k in ("top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty")
            if k in llm_kwargs and llm_kwargs[k] is not None
        }
        # Build tokenizer eagerly so chat-template errors surface at init, not on
        # the first request mid-trial.
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

    # --- BaseLLM interface ------------------------------------------------

    def get_model_context_limit(self) -> int:
        if not self._max_input_tokens:
            raise RuntimeError(
                "SkyRLNativeLLM needs model_info.max_input_tokens to report a "
                "context limit; pass it via harbor agent.kwargs.model_info."
            )
        return int(self._max_input_tokens)

    def get_model_output_limit(self) -> int | None:
        return int(self._max_output_tokens) if self._max_output_tokens else None

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] = [],
        logging_path: Path | None = None,
        previous_response_id: str | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if response_format is not None:
            # Same fallback LiteLLM does — embed JSON schema in the prompt.
            import json as _json

            try:
                schema = _json.dumps(response_format, indent=2)
            except TypeError:
                schema = _json.dumps(response_format.model_json_schema(), indent=2)
            prompt = (
                "You must respond in the following JSON format.\n\n"
                f"Here is the json schema:\n\n```json\n{schema}\n```\n\n"
                f"Here is the prompt:\n\n{prompt}\n"
            )

        messages = list(message_history) + [{"role": "user", "content": prompt}]

        prompt_token_ids: list[int] = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Cheap client-side context check so we can raise the harbor-typed
        # exception before the server complains.
        if self._max_input_tokens and len(prompt_token_ids) > self._max_input_tokens:
            raise ContextLengthExceededError()

        # Build the request. /skyrl/v1/generate forwards sampling_params straight
        # to vLLM, so the field names must match vLLM's SamplingParams.
        sampling_params: dict[str, Any] = {
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "logprobs": 1 if self._collect_rollout_details else None,
            **self._extra_sampling,
        }
        sampling_params = {k: v for k, v in sampling_params.items() if v is not None}

        payload = {
            "sampling_params": sampling_params,
            "model": self._model_name,
            "token_ids": prompt_token_ids,
        }
        headers = {"Content-Type": "application/json"}
        if self._session_id:
            headers["X-Session-ID"] = self._session_id

        url = f"{self._proxy_url}/skyrl/v1/generate"
        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    if "ContextLengthExceeded" in body or "maximum context length" in body.lower():
                        raise ContextLengthExceededError()
                    raise RuntimeError(
                        f"SkyRL /skyrl/v1/generate {resp.status}: {body[:500]}"
                    )
                response = await resp.json()

        # /skyrl/v1/generate response shape:
        #   {"choices": [{"token_ids": [...], "finish_reason": "...",
        #                 "logprobs": {"content": [{"logprob": ...}, ...]}}], ...}
        choice = response["choices"][0]
        completion_token_ids: list[int] = choice["token_ids"]
        finish_reason = choice.get("finish_reason")

        completion_logprobs: list[float] | None = None
        if self._collect_rollout_details:
            logprobs_obj = choice.get("logprobs") or {}
            content = logprobs_obj.get("content") or []
            if content:
                completion_logprobs = [
                    lp_info["logprob"] for lp_info in content if "logprob" in lp_info
                ]

        # Decode completion token ids back to text. skip_special_tokens lets
        # downstream parsers (json / xml in terminus-2) match the structured
        # body cleanly.
        content_text = self._tokenizer.decode(
            completion_token_ids,
            skip_special_tokens=True,
        )

        # Terminus-2 inspects finish_reason == "length" to raise
        # OutputLengthExceededError — keep parity.
        if finish_reason == "length":
            raise OutputLengthExceededError(
                f"Model {self._model_name} hit max_tokens limit. Response was truncated.",
                truncated_response=content_text,
            )

        usage = UsageInfo(
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(completion_token_ids),
            cache_tokens=0,
            cost_usd=0.0,
        )

        # Filter rollout details to only what the trainer keeps; harbor's chat
        # layer pivots these into rollout_details lists per turn.
        if self._collect_rollout_details:
            return LLMResponse(
                content=content_text,
                model_name=self._model_name,
                usage=usage,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                logprobs=completion_logprobs,
                extra={"stop_reason": finish_reason} if finish_reason else None,
            )
        return LLMResponse(
            content=content_text,
            model_name=self._model_name,
            usage=usage,
        )
