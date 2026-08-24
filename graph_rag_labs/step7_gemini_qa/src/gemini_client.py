"""Gemini client wrapper (using the new ``google-genai`` SDK)."""
from __future__ import annotations

import time
from dataclasses import dataclass

from google import genai
from google.genai import types


@dataclass
class GenerationResult:
    text: str
    model: str
    elapsed_ms: float
    input_tokens: int
    output_tokens: int


class GeminiClient:
    """Thin wrapper around the new ``google-genai`` SDK with retry support."""

    def __init__(self, api_key: str, model: str = "gemini-flash-latest") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        max_retries: int = 4,
    ) -> GenerationResult:
        """Run a chat-style generation with retry on 503/UNAVAILABLE.

        ``messages`` is an OpenAI-style list of {"role", "content"} dicts.
        """
        system = None
        contents = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system = text
            elif role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=text)])
                )
            elif role == "assistant":
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=text)])
                )
            else:
                raise ValueError(f"Unknown role: {role}")

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # 503/UNAVAILABLE / 429 = transient — back off and retry.
                status = getattr(exc, "code", None) or 0
                if status in (503, 429, 500) and attempt < max_retries:
                    wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32s
                    print(f"    [retry] Gemini {status} — waiting {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raise
            elapsed = (time.perf_counter() - t0) * 1000

            text = response.text or ""
            usage = response.usage_metadata or {}
            return GenerationResult(
                text=text,
                model=self.model_name,
                elapsed_ms=elapsed,
                input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            )

        # All retries exhausted.
        raise last_exc  # type: ignore[misc]