"""Terminus-2 subclass that routes the LLM call to SkyRL's native backend.

Loaded through harbor's official extension point — set
``agent.import_path = "examples.train_integrations.harbor_pgc.agents.skyrl_terminus_2:SkyRLTerminus2"``
in the trial config and harbor's factory imports + instantiates this class
instead of stock Terminus2. No monkey-patching.

The only behavioural change vs upstream Terminus2 is ``_initialize_llm_backend``:
when ``llm_backend == "skyrl"`` we return a ``SkyRLNativeLLM`` that talks
``/skyrl/v1/generate`` (preserves vllm's prompt/completion_token_ids and
logprobs, which the OpenAI /v1/chat/completions path on vllm-router drops).
"""

from __future__ import annotations

from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2

from ..llms.skyrl_native_llm import SkyRLNativeLLM


class SkyRLTerminus2(Terminus2):
    """Terminus-2 wired to talk SkyRL's vllm-router native generate endpoint."""

    @staticmethod
    def _initialize_llm_backend(
        llm_backend,
        model_name,
        temperature,
        collect_rollout_details,
        api_base,
        session_id,
        max_thinking_tokens,
        reasoning_effort,
        model_info,
        use_responses_api,
        llm_kwargs,
    ):
        backend_value = llm_backend.value if hasattr(llm_backend, "value") else llm_backend
        if backend_value != "skyrl":
            return Terminus2._initialize_llm_backend(
                llm_backend,
                model_name,
                temperature,
                collect_rollout_details,
                api_base,
                session_id,
                max_thinking_tokens,
                reasoning_effort,
                model_info,
                use_responses_api,
                llm_kwargs,
            )

        extra: dict[str, Any] = dict(llm_kwargs or {})
        proxy_url = extra.pop("proxy_url", None) or api_base
        tokenizer_path = extra.pop("tokenizer_path", None)
        if not proxy_url:
            raise ValueError(
                "skyrl backend requires either agent.kwargs.proxy_url or api_base."
            )
        if not tokenizer_path:
            raise ValueError(
                "skyrl backend requires agent.kwargs.llm_kwargs.tokenizer_path "
                "(HF model path used to apply the chat template)."
            )
        return SkyRLNativeLLM(
            model_name=model_name,
            proxy_url=proxy_url,
            tokenizer_path=tokenizer_path,
            temperature=temperature,
            collect_rollout_details=collect_rollout_details,
            session_id=session_id,
            model_info=model_info,
            **extra,
        )
