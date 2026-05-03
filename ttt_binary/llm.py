"""Thin LLM-call helpers for the skill-chained pipeline.

All three roles (generator, critic, solver) use the same gpt-oss-120b model
via Vertex AI's MaaS endpoint, so the existing repo-wide auth + retry
pattern carries over from Stage1/distinct_llm_prompting.py.

Public surface:
- ``call_vertex(prompt, *, model=DEFAULT_MODEL, system=None, temperature=0.7)``
- Compatibility shims ``call_anthropic`` and ``call_openai`` that forward to
  ``call_vertex`` so any caller written against the old API still works.
"""
from __future__ import annotations

import os
import random
import re
import threading
import time
from typing import Iterable


DEFAULT_MODEL = "openai/gpt-oss-120b-maas"

_vertex_client = None              # cached (OpenAI-shaped) client
_vertex_lock = threading.Lock()    # guards refresh + client creation


# ---------------------------------------------------------------------------
# Vertex MaaS client (shared with Stage1/distinct_llm_prompting.py)
# ---------------------------------------------------------------------------

def _get_vertex_access_token() -> str:
    from google.auth import default
    from google.auth.transport.requests import Request
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token


def _build_vertex_base_url() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT environment variable.\n"
            "  export GOOGLE_CLOUD_PROJECT='your-project-id'"
        )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if location == "global":
        location = "us-central1"
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/endpoints/openapi"
    )


def _is_auth_error(exc: BaseException) -> bool:
    """Detect cases where re-issuing the ADC token might help. Includes 403
    PERMISSION_DENIED because Vertex MaaS sometimes returns 403 (not 401) on
    a stale token."""
    msg = str(exc)
    return (
        "401" in msg
        or "403" in msg
        or "UNAUTHENTICATED" in msg
        or "PERMISSION_DENIED" in msg
        or "invalid authentication credentials" in msg.lower()
    )


def _refresh_vertex_token(*, reason: str) -> None:
    """Mint a fresh ADC token and install it on the cached client. Caller
    MUST hold _vertex_lock."""
    from openai import OpenAI
    new_token = _get_vertex_access_token()
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = OpenAI(api_key=new_token, base_url=_build_vertex_base_url())
    else:
        # The OpenAI SDK reads .api_key on every request, so mutating it is
        # safe and avoids tearing down in-flight connections.
        _vertex_client.api_key = new_token
    print(f"  [vertex info] refreshed ADC token ({reason})", flush=True)


def _get_vertex():
    """Return the cached Vertex client; lazy-init on first call. Token
    refresh happens reactively in `call_vertex` when an auth error fires."""
    global _vertex_client
    if _vertex_client is None:
        with _vertex_lock:
            if _vertex_client is None:
                _refresh_vertex_token(reason="first init")
    return _vertex_client


def call_vertex(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.7,
    max_retries: int = 8,
) -> str:
    """Single user-turn call to gpt-oss-120b via Vertex MaaS.

    Used by all three roles (generator, critic, solver) — same backend, just
    different prompts/temperatures. Retries on transients with jittered
    exponential backoff; on 401 we re-read the ADC token in place.
    """
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    last_err: Exception | None = None
    for attempt in range(max_retries):
        # Re-fetch the client each iteration: this is the cheap path (only a
        # dict lookup unless the TTL expired), and ensures a recently-refreshed
        # token from another thread is picked up here too.
        client = _get_vertex()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature,
            )
            if isinstance(resp, str):
                raise ValueError(f"vertex returned raw string: {resp[:200]!r}")
            if not getattr(resp, "choices", None):
                raise ValueError("response has no choices")
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("empty response content")
            return content
        except Exception as e:
            last_err = e
            if _is_auth_error(e):
                # Reactive refresh as a backstop. Held under the lock so
                # parallel workers don't all refresh at once.
                try:
                    with _vertex_lock:
                        _refresh_vertex_token(reason=f"reactive {type(e).__name__}")
                except Exception as refresh_err:
                    print(f"  [vertex warn] token refresh failed: "
                          f"{type(refresh_err).__name__}: {str(refresh_err)[:120]}",
                          flush=True)
            delay = min(2 ** attempt, 60) + random.uniform(0, 1)
            print(f"  [vertex warn] attempt {attempt+1}/{max_retries} "
                  f"failed ({type(e).__name__}: {str(e)[:140]}); retry in {delay:.1f}s",
                  flush=True)
            time.sleep(delay)
    raise RuntimeError(f"vertex call failed after {max_retries} retries: {last_err!r}")


# Compatibility shims — every prior call site keeps working but routes through
# Vertex MaaS regardless of the model name passed in. Stage 3 / Stage 4 call
# these by name and ignore the SDK distinction.

def call_anthropic(prompt: str, *, model: str = DEFAULT_MODEL, system: str | None = None,
                   temperature: float = 0.7, max_tokens: int = 16000,
                   cache_system: bool = True, max_retries: int = 8) -> str:
    return call_vertex(prompt, model=DEFAULT_MODEL, system=system,
                       temperature=temperature, max_retries=max_retries)


def call_openai(prompt: str, *, model: str = DEFAULT_MODEL, system: str | None = None,
                temperature: float = 0.7, max_retries: int = 8) -> str:
    return call_vertex(prompt, model=DEFAULT_MODEL, system=system,
                       temperature=temperature, max_retries=max_retries)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_loose(text: str):
    """Parse JSON from a model response that may include code fences or prose."""
    import json
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    # Try direct parse first.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find the first balanced top-level JSON object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == opener:
                depth += 1
            elif text[j] == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    raise ValueError(f"could not parse JSON from response: {text[:300]!r}")
