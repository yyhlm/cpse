from __future__ import annotations

import base64
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

import httpx
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

from .config import require_api_key
from .models import ModelConfig
from .usage_ledger import ApiUsageLedger, normalized_usage

TransportCallable = Callable[[dict[str, Any]], Any]

# Non-retriable HTTP statuses: request will never succeed by repeating it.
# 413 (Payload Too Large) is deterministic — the same rendered images always
# exceed the gateway's request-size limit — so retrying only wastes the backoff
# budget (15+45+90s) before the identical failure.
_NON_RETRIABLE_STATUS = {400, 401, 403, 404, 413, 422}

# API 协议：Responses API（PDF 直接作为文件输入）或
# Chat Completions（PDF 逐页渲染成图片，用于只支持 /chat/completions 的网关）。
API_PROTOCOL_RESPONSES = "responses"
API_PROTOCOL_CHAT_IMAGES = "chat_completions_pdf_images"
API_PROTOCOL_GEMINI = "gemini_generate_content"

# Sentinel for the per-call ``reasoning_effort`` override on the text paths:
# the default keeps the role config's value; passing ``None`` explicitly omits
# ``reasoning_effort`` from the request so the model reasons naturally. The
# TextGrad optimizer and schema-patch calls (which reuse the judge role client,
# where ``reasoning_effort: "none"`` is set to keep judging cheap and reliable)
# pass ``None`` because those format-following tasks need thinking.
_USE_CONFIG_REASONING = object()
_ACTIVE_OPERATION: ContextVar[str | None] = ContextVar("textgrad_validation_api_operation", default=None)


@contextmanager
def api_operation(name: str):
    """Label nested client calls without changing adapter method signatures."""
    token = _ACTIVE_OPERATION.set(name)
    try:
        yield
    finally:
        _ACTIVE_OPERATION.reset(token)


class _GeminiRetriable(RuntimeError):
    """A transient Gemini failure (429 / 5xx / empty content) worth retrying."""


class _GeminiResponseSchemaRejected(RuntimeError):
    """The gateway rejected a native responseSchema before generation began."""


def _gemini_response_schema(value: Any) -> Any:
    """Convert standard JSON Schema to the subset accepted by this Gemini gateway.

    The native ``responseSchema`` endpoint rejects ``additionalProperties`` and
    JSON-Schema union types. Removing the former and choosing the first concrete
    union type preserves the repository's required-field, object, and array
    constraints without mutating the local validation schema. Local jsonschema
    remains authoritative for the original union semantics.
    """
    if isinstance(value, list):
        return [_gemini_response_schema(item) for item in value]
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if key == "additionalProperties":
                continue
            if key == "type" and isinstance(item, list):
                if "items" in value and "array" in item:
                    converted[key] = "array"
                elif "properties" in value and "object" in item:
                    converted[key] = "object"
                else:
                    converted[key] = next((kind for kind in item if kind != "null"), "string")
                continue
            converted[key] = _gemini_response_schema(item)
        return converted
    return value


def _extract_retry_after(exc: APIStatusError) -> float | None:
    """Try to read retry_after from Cloudflare 524 or standard RateLimitError body."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        raw = body.get("retry_after") or body.get("retry-after")
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
    return None


class ResponsesPdfClient:
    """Minimal Responses API client that sends PDFs directly as input_file."""

    def __init__(self, config: ModelConfig, transport: TransportCallable | None = None, on_retry: Callable[[str], None] | None = None, httpx_transport: Any = None, *, role: str = "unspecified", usage_recorder: ApiUsageLedger | None = None):
        self._config = config
        self._transport = transport
        self._httpx_transport = httpx_transport
        self._client: OpenAI | None = None
        self._on_retry = on_retry
        self._role = role
        self._usage_recorder = usage_recorder
        # Runtime-only capability memory. A Gemini gateway may accept the
        # compact stage-1 schema yet reject the complex stage-2 schema. Keep
        # the rejection scoped so later stage-2 batches do not each spend one
        # deterministic HTTP 400, while stage 1 remains schema-constrained.
        self._gemini_rejected_response_schema_scopes: set[str] = set()

    def configure_usage(self, *, role: str, usage_recorder: ApiUsageLedger) -> None:
        """Enable accounting after construction, preserving legacy factories."""
        self._role = role
        self._usage_recorder = usage_recorder

    def complete_pdf_json(
        self,
        *,
        pdf_path: Path,
        system_prompt: str,
        payload: dict[str, Any],
        json_schema: dict[str, Any] | None = None,
        json_schema_name: str = "pdf_extraction",
        json_schema_capability_scope: str | None = None,
        operation: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a local PDF file: {pdf_path}")
        pdf_bytes = pdf_path.read_bytes()
        if not pdf_bytes:
            raise ValueError(f"PDF is empty: {pdf_path}")
        text = f"REQUEST:\n{_json(payload)}"
        if self._config.api_protocol == API_PROTOCOL_GEMINI:
            return self._complete_pdf_json_gemini(
                pdf_path,
                system_prompt,
                text,
                json_schema=json_schema,
                json_schema_capability_scope=json_schema_capability_scope,
                operation=operation,
            )
        if self._config.api_protocol == API_PROTOCOL_CHAT_IMAGES:
            return self._complete_pdf_json_chat_images(pdf_path, system_prompt, text, operation=operation)
        encoded_pdf = base64.b64encode(pdf_bytes).decode("ascii")
        response_format: dict[str, Any] = {"type": "json_object"}
        if json_schema is not None:
            response_format = {
                "type": "json_schema",
                "name": json_schema_name,
                # Terra's strict mode requires every object property to be
                # required, which conflicts with this repository DSL's real
                # optional-field semantics. Non-strict schema mode still gives
                # the gateway the field/type contract; local jsonschema remains
                # the final strict validator.
                "strict": False,
                "schema": json_schema,
            }
        request: dict[str, Any] = {
            "model": self._config.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{encoded_pdf}",
                        },
                        {"type": "input_text", "text": f"{system_prompt}\n\n{text}" if not self._config.use_instructions else text},
                    ],
                }
            ],
            "temperature": self._config.temperature,
            "stream": True,
            "text": {"format": response_format},
        }
        if self._config.max_output_tokens is not None:
            request["max_output_tokens"] = self._config.max_output_tokens
        if self._config.use_instructions:
            request["instructions"] = system_prompt
        response = self._call_with_retry(request, operation=operation)
        return _response_text(response), _safe_metadata(response)

    def _complete_pdf_json_chat_images(self, pdf_path: Path, system_prompt: str, text: str, *, operation: str | None = None) -> tuple[str, dict[str, Any]]:
        """Chat Completions variant: PDF rendered page-by-page into PNG image parts."""
        images = _render_pdf_to_images(pdf_path)
        content: list[dict[str, Any]] = [
            *({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}} for image in images),
            {"type": "text", "text": f"{system_prompt}\n\n{text}" if not self._config.use_instructions else text},
        ]
        messages: list[dict[str, Any]] = []
        if self._config.use_instructions:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        request: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "stream": True,
            "response_format": {"type": "json_object"},
        }
        if self._config.max_output_tokens is not None:
            request["max_tokens"] = self._config.max_output_tokens
        if self._config.reasoning_effort:
            request["reasoning_effort"] = self._config.reasoning_effort
        response = self._call_with_retry(request, operation=operation)
        _maybe_log_empty_content(self._on_retry, response)
        return _response_text(response), _safe_metadata(response)

    def complete_json(self, *, system_prompt: str, payload: dict[str, Any], reasoning_effort: Any = _USE_CONFIG_REASONING, operation: str | None = None) -> str:
        return self.complete_text(
            system_prompt=system_prompt, prompt=f"REQUEST:\n{_json(payload)}",
            json_mode=True, reasoning_effort=reasoning_effort, operation=operation,
        )

    def complete_text(self, *, system_prompt: str | None, prompt: str, json_mode: bool = False, reasoning_effort: Any = _USE_CONFIG_REASONING, operation: str | None = None) -> str:
        if self._config.api_protocol == API_PROTOCOL_GEMINI:
            return self._complete_text_gemini(system_prompt=system_prompt, prompt=prompt, json_mode=json_mode, operation=operation)
        if self._config.api_protocol == API_PROTOCOL_CHAT_IMAGES:
            return self._complete_text_chat_images(
                system_prompt=system_prompt, prompt=prompt, json_mode=json_mode, reasoning_effort=reasoning_effort,
                operation=operation,
            )
        request: dict[str, Any] = {
            "model": self._config.model,
            "input": prompt if self._config.use_instructions else f"{system_prompt or ''}\n\n{prompt}".strip(),
            "temperature": self._config.temperature,
            "stream": True,
        }
        if json_mode:
            request["text"] = {"format": {"type": "json_object"}}
        if self._config.max_output_tokens is not None:
            request["max_output_tokens"] = self._config.max_output_tokens
        if self._config.use_instructions:
            request["instructions"] = system_prompt or ""
        return _response_text(self._call_with_retry(request, operation=operation))

    def _complete_text_chat_images(self, *, system_prompt: str | None, prompt: str, json_mode: bool, reasoning_effort: Any = _USE_CONFIG_REASONING, operation: str | None = None) -> str:
        messages: list[dict[str, Any]] = []
        if self._config.use_instructions:
            messages.append({"role": "system", "content": system_prompt or ""})
            user_content: Any = prompt
        else:
            user_content = f"{system_prompt or ''}\n\n{prompt}".strip()
        messages.append({"role": "user", "content": user_content})
        request: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "stream": True,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if self._config.max_output_tokens is not None:
            request["max_tokens"] = self._config.max_output_tokens
        effective_reasoning = self._config.reasoning_effort if reasoning_effort is _USE_CONFIG_REASONING else reasoning_effort
        if effective_reasoning:
            request["reasoning_effort"] = effective_reasoning
        response = self._call_with_retry(request, operation=operation)
        _maybe_log_empty_content(self._on_retry, response)
        return _response_text(response)

    # ---- Gemini native generateContent protocol ----

    def _complete_pdf_json_gemini(
        self,
        pdf_path: Path,
        system_prompt: str,
        text: str,
        *,
        json_schema: dict[str, Any] | None = None,
        json_schema_capability_scope: str | None = None,
        operation: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Gemini native path: send the PDF bytes inline as ``application/pdf``."""
        pdf_bytes = pdf_path.read_bytes()
        if not pdf_bytes:
            raise ValueError(f"PDF is empty: {pdf_path}")
        prompt = f"{system_prompt}\n\n{text}" if not self._config.use_instructions else text
        parts: list[dict[str, Any]] = [
            {"inline_data": {"mime_type": "application/pdf", "data": base64.b64encode(pdf_bytes).decode("ascii")}},
            {"text": prompt},
        ]
        schema_disabled = (
            json_schema is not None
            and json_schema_capability_scope is not None
            and json_schema_capability_scope in self._gemini_rejected_response_schema_scopes
        )
        body = self._build_gemini_body(
            parts,
            system_prompt=system_prompt,
            json_mode=True,
            json_schema=None if schema_disabled else json_schema,
        )
        try:
            raw, metadata = self._gemini_call(body, operation=operation)
            if schema_disabled:
                metadata["response_schema_skipped_due_to_capability"] = True
                metadata["response_schema_capability_scope"] = json_schema_capability_scope
            return raw, metadata
        except _GeminiResponseSchemaRejected as exc:
            # Some Gemini gateways accept the compact stage-1 response schema
            # but reject the much larger stage-2 schema. This is a one-time
            # capability fallback, not a transient retry: preserve JSON mode
            # and let the existing local schema validator remain authoritative.
            if json_schema_capability_scope is not None:
                self._gemini_rejected_response_schema_scopes.add(json_schema_capability_scope)
            if self._on_retry:
                capability_note = (
                    f"; disabled for later {json_schema_capability_scope} requests this run"
                    if json_schema_capability_scope is not None
                    else ""
                )
                self._on_retry(
                    "[gemini-schema-fallback] responseSchema rejected (HTTP 400); "
                    f"retrying once without responseSchema{capability_note}"
                )
            fallback_body = self._build_gemini_body(
                parts,
                system_prompt=system_prompt,
                json_mode=True,
            )
            raw, metadata = self._gemini_call(fallback_body, operation=operation)
            metadata["response_schema_fallback"] = True
            metadata["response_schema_fallback_reason"] = str(exc)
            if json_schema_capability_scope is not None:
                metadata["response_schema_capability_scope"] = json_schema_capability_scope
            return raw, metadata

    def _complete_text_gemini(self, *, system_prompt: str | None, prompt: str, json_mode: bool, operation: str | None = None) -> str:
        text = f"{system_prompt or ''}\n\n{prompt}".strip() if not self._config.use_instructions else prompt
        body = self._build_gemini_body([{"text": text}], system_prompt=system_prompt, json_mode=json_mode)
        raw, _metadata = self._gemini_call(body, operation=operation)
        return raw

    def _build_gemini_body(
        self,
        parts: list[dict[str, Any]],
        *,
        system_prompt: str | None,
        json_mode: bool,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": parts}]}
        if self._config.use_instructions and system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        generation: dict[str, Any] = {"temperature": self._config.temperature}
        if self._config.max_output_tokens is not None:
            generation["maxOutputTokens"] = self._config.max_output_tokens
        if json_mode:
            generation["responseMimeType"] = "application/json"
        if json_schema is not None:
            generation["responseSchema"] = _gemini_response_schema(json_schema)
        body["generationConfig"] = generation
        return body

    def _gemini_call(self, body: dict[str, Any], *, operation: str | None = None) -> tuple[str, dict[str, Any]]:
        if not self._config.base_url:
            raise RuntimeError("Gemini protocol requires a base_url.")
        url = f"{self._config.base_url}/models/{self._config.model}:generateContent"
        headers = {"x-goog-api-key": require_api_key(self._config), "Content-Type": "application/json"}
        last_error: Exception | None = None
        logical_call_id = uuid.uuid4().hex
        operation = operation or _ACTIVE_OPERATION.get() or self._role
        for attempt in range(self._config.max_retries + 1):
            started = time.perf_counter()
            try:
                with self._gemini_http_client() as client:
                    response = client.post(url, json=body, headers=headers)
                if response.status_code == 400 and "responseSchema" in (body.get("generationConfig") or {}):
                    raise _GeminiResponseSchemaRejected(f"Gemini rejected responseSchema: {response.text[:240]}")
                raw, metadata = self._parse_gemini_response(response)
                self._record_attempt(
                    logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                    status="success", duration_seconds=time.perf_counter() - started,
                    response={"id": metadata.get("id"), "model": metadata.get("model"), "usage": metadata.get("usage")},
                )
                return raw, metadata
            except _GeminiRetriable as exc:
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            except Exception as exc:
                self._record_attempt(
                    logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                    status=self._failure_status(exc), duration_seconds=time.perf_counter() - started,
                    error=exc, retry_backoff_seconds=0.0,
                )
                raise
            if attempt >= self._config.max_retries:
                self._record_attempt(
                    logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                    status=self._failure_status(last_error), duration_seconds=time.perf_counter() - started,
                    error=last_error, retry_backoff_seconds=0.0,
                )
                break
            # Connection-layer failures (httpx.TransportError, dropped socket,
            # protocol reset) retry almost immediately — the upstream gateway
            # does not need long to recover from a broken connection. Only
            # HTTP 429/5xx (wrapped in _GeminiRetriable) backs off exponentially,
            # since those signal real server-side or rate-limit pressure.
            if isinstance(last_error, httpx.TransportError):
                delay = 2.0
            else:
                delay = min(90.0, 15.0 * (3**attempt))
            if self._on_retry:
                kind = type(last_error).__name__
                self._on_retry(f"[gemini] [retry {attempt + 1}/{self._config.max_retries}] waiting {delay:.0f}s after {kind}")
            self._record_attempt(
                logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                status=self._failure_status(last_error), duration_seconds=time.perf_counter() - started,
                error=last_error, retry_backoff_seconds=float(delay),
            )
            time.sleep(delay)
        raise RuntimeError(f"Gemini request failed after {self._config.max_retries + 1} attempt(s): {last_error}") from last_error

    def _gemini_http_client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {"timeout": self._config.timeout_seconds}
        if self._config.proxy:
            kwargs["proxy"] = self._config.proxy
        if self._httpx_transport is not None:
            kwargs["transport"] = self._httpx_transport
        return httpx.Client(**kwargs)

    def _parse_gemini_response(self, response: httpx.Response) -> tuple[str, dict[str, Any]]:
        status = response.status_code
        if status != 200:
            if status in {408, 429} or status >= 500:
                raise _GeminiRetriable(f"Gemini HTTP {status}: {response.text[:240]}")
            raise RuntimeError(f"Gemini HTTP {status}: {response.text[:240]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise _GeminiRetriable(f"Gemini non-JSON response: {response.text[:200]}") from exc
        candidates = data.get("candidates") or []
        if not candidates:
            raise _GeminiRetriable(f"Gemini returned no candidates: {_json(data)[:240]}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        finish = (candidates[0] or {}).get("finishReason")
        if not text.strip():
            if self._on_retry:
                self._on_retry(f"[gemini-empty] finishReason={finish}; no visible text")
            raise _GeminiRetriable(f"Gemini empty content (finishReason={finish})")
        if finish == "MAX_TOKENS":
            # Output budget exhausted mid-generation: the text is truncated and
            # unreliable (e.g. a half-written JSON patch). Retry only gives the
            # same deterministic cut unless max_output_tokens is raised, but the
            # clear diagnostic beats silently feeding a broken string to a parser.
            if self._on_retry:
                self._on_retry(f"[gemini-truncated] finishReason=MAX_TOKENS; output cut at {len(text)} chars (raise max_output_tokens or shrink the request)")
            raise _GeminiRetriable(f"Gemini output truncated (finishReason=MAX_TOKENS, {len(text)} chars)")
        return text, {
            "id": data.get("responseId"),
            "model": self._config.model,
            "usage": data.get("usageMetadata"),
        }

    def _call_with_retry(self, request: dict[str, Any], *, operation: str | None = None) -> Any:
        last_error: Exception | None = None
        logical_call_id = uuid.uuid4().hex
        operation = operation or _ACTIVE_OPERATION.get() or self._role
        # Per-attempt timeout: slightly more than Cloudflare's 120s proxy limit
        # so a 524 arrives before the SDK timeout fires.
        per_attempt_timeout = min(self._config.timeout_seconds, 130.0)
        for attempt in range(self._config.max_retries + 1):
            started = time.perf_counter()
            try:
                if self._transport is not None:
                    response = self._transport(request)
                elif self._config.api_protocol == API_PROTOCOL_CHAT_IMAGES:
                    response = self._get_client().chat.completions.create(timeout=per_attempt_timeout, **request)
                else:
                    response = self._get_client().responses.create(timeout=per_attempt_timeout, **request)
                result = response
                if request.get("stream"):
                    result = self._consume_stream(response)
                    self._raise_if_empty_chat_stream(result)
                self._record_attempt(
                    logical_call_id=logical_call_id,
                    operation=operation,
                    attempt=attempt + 1,
                    status="success",
                    duration_seconds=time.perf_counter() - started,
                    response=result,
                )
                return result
            except APITimeoutError as exc:
                last_error = exc
            except APIConnectionError as exc:
                last_error = exc
            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                if isinstance(status, int) and status in _NON_RETRIABLE_STATUS:
                    self._record_attempt(
                        logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                        status="http_error", duration_seconds=time.perf_counter() - started,
                        error=exc, retry_backoff_seconds=0.0,
                    )
                    if self._on_retry:
                        self._on_retry(f"[req] non-retriable {status}, aborting")
                    raise
                last_error = exc
            except RuntimeError as exc:
                # Stream failures and transport runtime errors are transient unless
                # they have already been classified as a non-retriable HTTP status.
                if str(exc).startswith("Required API key environment variable is missing:"):
                    raise
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            except APIError as exc:
                # The SDK raises the base APIError for mid-stream service faults
                # (e.g. server_is_overloaded) that do not map to a specific
                # status subclass. These are transient — retry instead of letting
                # a temporary overload crash the whole training run.
                last_error = exc
            except Exception as exc:
                self._record_attempt(
                    logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                    status=type(exc).__name__, duration_seconds=time.perf_counter() - started,
                    error=exc, retry_backoff_seconds=0.0,
                )
                raise
            if attempt >= self._config.max_retries:
                self._record_attempt(
                    logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                    status=self._failure_status(last_error), duration_seconds=time.perf_counter() - started,
                    error=last_error, retry_backoff_seconds=0.0,
                )
                break
            # Compute delay: respect Cloudflare retry_after for 524; connection-layer
            # failures (dropped connection, protocol reset, transport error, runtime
            # stream fault) retry almost immediately — the upstream gateway does not
            # need long to recover from a broken socket, and a long fixed wait only
            # inflates wall-clock. Only an explicit HTTP status (429/5xx) backs off
            # exponentially, since those signal real server-side pressure.
            if isinstance(last_error, (APIConnectionError, httpx.TransportError)) or (
                isinstance(last_error, RuntimeError) and not isinstance(last_error, APIStatusError)
            ):
                delay = 2.0
            else:
                delay = _extract_retry_after(last_error) if isinstance(last_error, APIStatusError) else None
                if delay is None:
                    delay = min(90.0, 15.0 * (3**attempt))
                else:
                    delay = min(max(delay, 30.0), 180.0)
            if self._on_retry:
                status = getattr(last_error, "status_code", None) or type(last_error).__name__
                self._on_retry(f"[retry {attempt + 1}/{self._config.max_retries}] waiting {delay:.0f}s after {status}")
            self._record_attempt(
                logical_call_id=logical_call_id, operation=operation, attempt=attempt + 1,
                status=self._failure_status(last_error), duration_seconds=time.perf_counter() - started,
                error=last_error, retry_backoff_seconds=float(delay),
            )
            time.sleep(delay)
        raise RuntimeError(
            f"Responses request failed after {self._config.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _failure_status(error: Exception | None) -> str:
        if isinstance(error, _GeminiRetriable) and "HTTP" in str(error):
            return "http_error"
        if isinstance(error, APIStatusError):
            return "http_error"
        if isinstance(error, APITimeoutError):
            return "timeout"
        if isinstance(error, (APIConnectionError, httpx.TransportError)):
            return "connection_error"
        if isinstance(error, RuntimeError):
            return "runtime_error"
        return type(error).__name__ if error is not None else "unknown_error"

    def _record_attempt(
        self,
        *,
        logical_call_id: str,
        operation: str,
        attempt: int,
        status: str,
        duration_seconds: float,
        response: Any = None,
        error: Exception | None = None,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        if self._usage_recorder is None:
            return
        metadata = _safe_metadata(response) if response is not None else {}
        raw_usage = metadata.get("usage")
        usage = normalized_usage(raw_usage)
        event = {
            "logical_call_id": logical_call_id,
            "attempt": attempt,
            "role": self._role,
            "operation": operation,
            "model_requested": self._config.model,
            "model_returned": metadata.get("model"),
            "api_protocol": self._config.api_protocol,
            "status": status,
            "duration_seconds": round(float(duration_seconds), 6),
            "retry_backoff_seconds": float(retry_backoff_seconds),
            "usage_available": isinstance(raw_usage, dict) or raw_usage is not None,
            **usage,
        }
        if error is not None:
            event["error_type"] = type(error).__name__
            event["http_status"] = getattr(error, "status_code", None)
        self._usage_recorder.record(event)

    def _raise_if_empty_chat_stream(self, result: Any) -> None:
        """Treat an empty chat-completions stream as a retriable transient failure.

        A reasoning model that exhausts its output budget on ``reasoning`` deltas
        returns a 200-OK stream with zero ``delta.content`` (``finish_reason=length``).
        Without this guard the empty string flows through to JSON parsing and
        hard-fails the whole document; the ``except RuntimeError`` branch in
        ``_call_with_retry`` then gives the gateway another chance. This only
        catches the OCCASIONAL empty response (gateway hiccup, reasoning
        nondeterminism) — a DETERMINISTIC budget exhaustion still needs the
        config-level fix (explicit ``max_output_tokens`` and, for the judge,
        ``include_pdf: false`` to drop the image burden). Only the chat-completions
        path produces ``_empty_content_diagnostic``; the Responses stream path
        already raises on ``response.incomplete`` / ``response.failed``.
        """
        if not isinstance(result, dict) or result.get("output_text"):
            return
        diag = result.get("_empty_content_diagnostic")
        if not isinstance(diag, dict):
            return
        _maybe_log_empty_content(self._on_retry, result)
        finish = diag.get("last_finish_reason")
        raise RuntimeError(
            f"chat-completions stream produced 0 bytes of content "
            f"(finish_reason={finish}, chunks={diag.get('chunk_count')}, "
            f"reasoning={diag.get('saw_reasoning') or diag.get('saw_reasoning_content')}); "
            f"likely a reasoning model that exhausted its output budget"
        )

    def _consume_stream(self, response: Any) -> dict[str, Any]:
        if self._config.api_protocol == API_PROTOCOL_CHAT_IMAGES:
            return _consume_chat_completions_stream(response)
        return _consume_response_stream(response)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            client_kwargs: dict[str, Any] = {
                "api_key": require_api_key(self._config),
                "base_url": self._config.base_url,
                "max_retries": 0,
                "timeout": self._config.timeout_seconds,
            }
            if self._config.proxy:
                client_kwargs["http_client"] = httpx.Client(
                    proxy=self._config.proxy,
                    timeout=self._config.timeout_seconds,
                )
            self._client = OpenAI(**client_kwargs)
        return self._client



class ResponsesTextGradEngine:
    """Minimal lazy TextGrad engine adapter to keep log configuration run-local."""

    def __init__(self, client: ResponsesPdfClient):
        self._client = client
        self.model_string = client._config.model

    def generate(self, prompt: str, system_prompt: str | None = None, **_kwargs: Any) -> str:
        # The TextGrad optimizer-update call must reproduce the tagged
        # ``<new_variable_start>...</new_variable_start>`` format, which needs
        # thinking; explicitly bypass the judge role's ``reasoning_effort: "none"``.
        return self._client.complete_text(system_prompt=system_prompt, prompt=prompt, reasoning_effort=None, operation="textgrad_update")

    def __call__(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        return self.generate(prompt, system_prompt=system_prompt, **kwargs)


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(response, dict):
        text = response.get("output_text")
        if isinstance(text, str):
            return text
    raise RuntimeError("Responses API reply has no output_text.")


def _consume_response_stream(stream: Any) -> dict[str, Any]:
    """Collect streamed text deltas and retain response metadata without printing content."""
    chunks: list[str] = []
    completed: Any | None = None
    for event in stream:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                chunks.append(str(delta))
        elif event_type == "response.completed":
            completed = getattr(event, "response", None)
        elif event_type in {"response.failed", "response.incomplete"}:
            response = getattr(event, "response", None)
            error = getattr(response, "error", None) or getattr(event, "error", None)
            raise RuntimeError(f"Responses stream ended with {event_type}: {error or response or event}")
    return {
        "output_text": "".join(chunks),
        "id": getattr(completed, "id", None),
        "model": getattr(completed, "model", None),
        "usage": getattr(completed, "usage", None),
    }


def _consume_chat_completions_stream(stream: Any) -> dict[str, Any]:
    """Collect chat-completions stream deltas and retain metadata.

    When the model streams zero ``delta.content`` (e.g. a reasoning model that
    exhausts its output budget before emitting a visible answer), capture which
    delta fields the gateway actually populated so the failure is diagnosable
    from ``extraction.metadata.json`` and the ``[chat-empty]`` log line instead
    of silently persisting a 0-byte response.
    """
    chunks: list[str] = []
    response_id: Any = None
    model: Any = None
    usage: Any = None
    chunk_count = 0
    delta_keys: set[str] = set()
    first_delta: dict[str, Any] | None = None
    saw_reasoning_content = False
    saw_reasoning = False
    last_finish_reason: Any = None
    for chunk in stream:
        chunk_count += 1
        choices = getattr(chunk, "choices", None)
        if not choices:
            usage = getattr(chunk, "usage", None)
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        snapshot = _delta_snapshot(delta)
        if snapshot:
            delta_keys.update(snapshot.keys())
            if first_delta is None:
                first_delta = snapshot
        content = getattr(delta, "content", None)
        if content:
            chunks.append(str(content))
        if delta is not None:
            if getattr(delta, "reasoning_content", None) is not None:
                saw_reasoning_content = True
            if getattr(delta, "reasoning", None) is not None:
                saw_reasoning = True
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            last_finish_reason = finish_reason
        if response_id is None:
            response_id = getattr(chunk, "id", None)
        if model is None:
            model = getattr(chunk, "model", None)
    output_text = "".join(chunks)
    result: dict[str, Any] = {
        "output_text": output_text,
        "id": response_id,
        "model": model,
        "usage": usage,
    }
    if not output_text:
        result["_empty_content_diagnostic"] = {
            "chunk_count": chunk_count,
            "first_delta": first_delta or {},
            "delta_keys_seen": sorted(delta_keys),
            "saw_reasoning_content": saw_reasoning_content,
            "saw_reasoning": saw_reasoning,
            "last_finish_reason": _json_safe(last_finish_reason),
        }
    return result


def _delta_snapshot(delta: Any) -> dict[str, Any]:
    """Capture the public field→value map of a streamed chat-completions delta.

    Values are truncated so a large ``reasoning_content`` block does not bloat
    the diagnostic artifact. Used to confirm which field a gateway actually
    populated when ``delta.content`` is empty.
    """
    if delta is None:
        return {}
    if isinstance(delta, dict):
        items: list[tuple[str, Any]] = list(delta.items())
    else:
        model_dump = getattr(delta, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(mode="json")
                items = list(dumped.items()) if isinstance(dumped, dict) else []
            except Exception:
                items = []
        else:
            as_dict = getattr(delta, "__dict__", None)
            items = list(as_dict.items()) if isinstance(as_dict, dict) else []
    snapshot: dict[str, Any] = {}
    for key, value in items:
        if str(key).startswith("_"):
            continue
        snapshot[str(key)] = _truncate_for_diagnostic(value)
    return snapshot


def _truncate_for_diagnostic(value: Any, *, limit: int = 200) -> Any:
    """Shorten long strings/nested values so the diagnostic stays compact."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, list):
        return [_truncate_for_diagnostic(v, limit=limit) for v in value[:3]]
    if isinstance(value, dict):
        return {str(k): _truncate_for_diagnostic(v, limit=limit) for k, v in list(value.items())[:5]}
    return _json_safe(value)


def _maybe_log_empty_content(on_log: Callable[[str], None] | None, response: Any) -> None:
    """Emit a one-line ``[chat-empty]`` diagnostic when a stream produced no content."""
    if on_log is None or not isinstance(response, dict):
        return
    diag = response.get("_empty_content_diagnostic")
    if not diag:
        return
    on_log(
        "[chat-empty] captured 0 bytes of delta.content "
        f"(chunks={diag.get('chunk_count')} "
        f"delta_keys={diag.get('delta_keys_seen')} "
        f"first_delta={diag.get('first_delta')} "
        f"reasoning_content={diag.get('saw_reasoning_content')} "
        f"reasoning={diag.get('saw_reasoning')} "
        f"finish_reason={diag.get('last_finish_reason')}); "
        "the gateway streamed no visible answer — likely a reasoning model that exhausted its output budget"
    )


def _render_pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[str]:
    """Rasterize every PDF page to a base64 PNG for chat-completions image input.

    Lazy-imports pymupdf so the Responses API path carries no extra dependency.
    150 dpi (down from 200) keeps every page's PNG below the payload ceiling of
    gateways like SenseNova: at 200 dpi a many-page chemistry paper can exceed
    the request-size limit and return a non-retriable 413 before the model is
    ever called. 150 is the standard legibility/payload trade-off for document
    image input; if 413 still recurs on very long PDFs, lower further or cap
    pages. NOTE: dpi is NOT part of the run fingerprint, so do not change it
    mid-run-id — pick one value and keep it for the whole run.
    """
    import fitz  # pymupdf

    images: list[str] = []
    document = fitz.open(pdf_path)
    try:
        for page in document:
            pix = page.get_pixmap(dpi=dpi)
            images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    finally:
        document.close()
    return images


def _safe_metadata(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        kept = {
            key: _json_safe(response.get(key))
            for key in ("id", "model", "usage")
            if key in response
        }
        diagnostic = response.get("_empty_content_diagnostic")
        if diagnostic is not None:
            kept["_empty_content_diagnostic"] = diagnostic
        return kept
    return {
        key: _json_safe(getattr(response, key))
        for key in ("id", "model", "usage")
        if getattr(response, key, None) is not None
    }


def _json_safe(value: Any) -> Any:
    """Convert SDK response models (notably ResponseUsage) to plain JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(child) for child in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        return {str(key): _json_safe(child) for key, child in as_dict.items() if not key.startswith("_")}
    return str(value)


def _json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)
