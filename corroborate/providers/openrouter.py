from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from corroborate.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
# Most seconds one request may spend sleeping between retries.
MAX_RETRY_WAIT = 20.0


class UsageMeter:
    """Thread-safe running totals of API usage across calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cost = 0.0

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.reasoning_tokens += reasoning_tokens
            self.cost += cost

    def summary(self) -> str:
        return (
            f"{self.calls} calls, {self.prompt_tokens} prompt + "
            f"{self.completion_tokens} completion tokens "
            f"({self.reasoning_tokens} reasoning), ${self.cost:.4f}"
        )


def _retry_delay(resp: httpx.Response | None, attempt: int) -> float:
    """Honor a numeric Retry-After header, else back off exponentially.

    A 429 backs off from 5s, not 1s: upstream rate-limit windows outlast a
    couple of seconds. The jitter keeps concurrent stories from retrying in
    lockstep and hitting the limit together again.
    """
    if resp is not None:
        retry_after = resp.headers.get("retry-after", "")
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass
        if resp.status_code == 429:
            return min(5.0 * 2 ** attempt, 60.0) + random.uniform(0, 2)
    return float(2 ** attempt)


def ceiling_for_cheapest(
    endpoints: list[dict],
    count: int,
    quantizations: list[str],
    required_params: list[str],
    min_throughput: int | None,
) -> tuple[float, float, list[str]] | None:
    """Price ceiling (USD per 1M prompt, completion) admitting the `count`
    cheapest providers able to serve the request, and their names.

    Able means: an allowed quantization, every request parameter supported,
    and (when OpenRouter reports it) median throughput at the floor. Slow
    providers are left out of the count, not the route: one cheaper than
    the ceiling still receives requests. None if no provider qualifies.
    """
    usable = []
    for ep in endpoints:
        if ep.get("quantization") not in quantizations:
            continue
        if not set(required_params) <= set(ep.get("supported_parameters") or []):
            continue
        p50 = (ep.get("throughput_last_30m") or {}).get("p50")
        if min_throughput and p50 is not None and p50 < min_throughput:
            continue
        pricing = ep.get("pricing") or {}
        try:
            prompt = float(pricing["prompt"]) * 1e6
            completion = float(pricing["completion"]) * 1e6
        except (KeyError, TypeError, ValueError):
            continue
        usable.append((prompt, completion, ep.get("provider_name", "?")))

    if not usable:
        return None
    cheapest = sorted(usable)[:count]
    # A hair over the highest price, so float noise can't exclude it.
    return (
        max(p for p, _, _ in cheapest) * 1.001,
        max(c for _, c, _ in cheapest) * 1.001,
        [name for _, _, name in cheapest],
    )


def price_ceiling(
    api_key: str,
    model: str,
    count: int,
    quantizations: list[str],
    required_params: list[str],
    min_throughput: int | None,
) -> tuple[float, float] | None:
    """Look up the model's providers and return ceiling_for_cheapest's
    prices, or None (with a warning) so the caller uses its static one."""
    try:
        resp = httpx.get(
            ENDPOINTS_URL.format(model=model),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        endpoints = resp.json()["data"]["endpoints"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
        logger.warning(
            "  Price lookup for %s failed (%s: %s); using the configured ceiling",
            model, e.__class__.__name__, e,
        )
        return None

    found = ceiling_for_cheapest(
        endpoints, count, quantizations, required_params, min_throughput,
    )
    if found is None:
        logger.warning(
            "  No provider for %s meets the routing filters; "
            "using the configured ceiling", model,
        )
        return None
    prompt, completion, names = found
    if len(names) < count:
        logger.warning(
            "  Only %d provider(s) for %s meet the routing filters: %s",
            len(names), model, ", ".join(names),
        )
    logger.info(
        "  Routing %s: ceiling $%.3f / $%.3f per 1M (cheapest: %s)",
        model, prompt, completion, ", ".join(names),
    )
    return prompt, completion


def _post_with_retry(
    client: httpx.Client,
    url: str,
    headers: dict,
    body: dict,
    max_retries: int,
    label: str,
    max_wait: float = MAX_RETRY_WAIT,
) -> httpx.Response:
    """POST with backoff on transport errors, 5xx, and 429.

    Retrying stops once the next wait would push the total past max_wait
    seconds: a story whose request keeps being refused must not hold up the
    whole run. Raises RuntimeError if every attempt fails. Other status
    codes are returned unretried for the caller to handle.
    """
    last_error: Exception | None = None
    waited = 0.0
    attempts = 0

    for attempt in range(max_retries):
        attempts += 1
        start = time.time()
        try:
            resp = client.post(url, headers=headers, json=body)
        except httpx.TransportError as e:
            last_error = e
            logger.warning(
                "  %s request failed (attempt %d/%d, %.1fs): %s: %s",
                label, attempt + 1, max_retries, time.time() - start,
                e.__class__.__name__, e,
            )
            delay = _retry_delay(None, attempt)
            if waited + delay > max_wait:
                break
            waited += delay
            time.sleep(delay)
            continue

        if resp.status_code >= 500 or resp.status_code == 429:
            last_error = RuntimeError(f"HTTP {resp.status_code}")
            # OpenRouter's error body names the upstream provider and reason.
            logger.warning(
                "  %s server error (attempt %d/%d): HTTP %d %s",
                label, attempt + 1, max_retries, resp.status_code,
                resp.text[:300].replace("\n", " "),
            )
            delay = _retry_delay(resp, attempt)
            if waited + delay > max_wait:
                break
            waited += delay
            time.sleep(delay)
            continue

        return resp

    raise RuntimeError(
        f"{label} request failed after {attempts} attempts: {last_error}"
    )


class OpenRouterEmbedding(EmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model: str,
        batch_size: int = 50,
        max_concurrency: int = 8,
    ):
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        # Shared across every caller, so concurrent clusters together stay
        # within max_concurrency in-flight requests.
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self.usage = UsageMeter()

    def cost_per_million_tokens(self) -> float:
        return 0.010

    def embed(
        self, texts: list[str], max_retries: int = 3
    ) -> tuple[np.ndarray, int]:
        batches = [
            texts[i:i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        if not batches:
            return np.array([]), 0

        workers = min(self._max_concurrency, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(
                lambda job: self._embed_batch(*job, len(batches), max_retries),
                enumerate(batches),
            ))

        all_embeddings = [vec for vectors, _ in results for vec in vectors]
        total_tokens = sum(tokens for _, tokens in results)
        return np.array(all_embeddings), total_tokens

    def _embed_batch(
        self,
        batch_idx: int,
        batch: list[str],
        num_batches: int,
        max_retries: int,
    ) -> tuple[list[list[float]], int]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {"model": self._model, "input": batch, "encoding_format": "float"}

        with self._slots:
            start = time.time()
            with httpx.Client(timeout=60.0) as client:
                resp = _post_with_retry(
                    client, EMBEDDING_URL, headers, body, max_retries, "Embedding"
                )
            elapsed = time.time() - start

        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API returned HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )

        data = resp.json()
        # Vectors must line up with their input texts: order by the
        # response's index, and refuse a short batch rather than misalign.
        items = sorted(data["data"], key=lambda item: item.get("index", 0))
        if len(items) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(items)} vectors "
                f"for {len(batch)} texts"
            )

        usage = data.get("usage", {})
        batch_tokens = usage.get("total_tokens", 0)
        self.usage.record(
            batch_tokens,
            cost=usage.get("cost")
            or batch_tokens * self.cost_per_million_tokens() / 1_000_000,
        )

        logger.info(
            "  Batch %d/%d (%d texts) — %.1fs, %d tokens",
            batch_idx + 1, num_batches, len(batch), elapsed, batch_tokens,
        )
        return [item["embedding"] for item in items], batch_tokens


class OpenRouterLLM(LLMProvider):

    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning: bool = True,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        provider_routing: dict | None = None,
        max_concurrency: int = 16,
        timeout: float = 180.0,
        fallback: OpenRouterLLM | None = None,
        fallback_after: int = 2,
    ):
        self._api_key = api_key
        self._model = model
        self._reasoning = reasoning
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._provider_routing = provider_routing
        self._timeout = timeout
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self.usage = UsageMeter()
        self._fallback = fallback
        self._fallback_after = fallback_after
        if fallback is not None:
            # The fallback's calls count toward this role's usage.
            fallback.usage = self.usage

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 5,
        *,
        json_mode: bool = False,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._reasoning_effort:
            body["reasoning"] = {"effort": self._reasoning_effort}
        elif self._reasoning:
            body["reasoning"] = {"enabled": True}
        if self._max_tokens:
            body["max_tokens"] = self._max_tokens
        if self._provider_routing:
            body["provider"] = self._provider_routing
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        attempts = self._fallback_after if self._fallback else max_retries
        try:
            with self._slots:
                start = time.time()
                with httpx.Client(timeout=self._timeout) as client:
                    resp = _post_with_retry(
                        client, CHAT_URL, headers, body, attempts, "LLM"
                    )
                elapsed = time.time() - start
            if resp.status_code != 200:
                raise RuntimeError(
                    f"LLM API returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
        except RuntimeError as e:
            if self._fallback is None:
                raise
            logger.warning(
                "  %s failed (%s); routing to %s",
                self._model, e, self._fallback._model,
            )
            return self._fallback.complete(
                system_prompt, user_prompt, max_retries, json_mode=json_mode,
            )

        data = resp.json()
        if not data.get("choices"):
            raise RuntimeError(f"LLM API returned no choices: {data.get('error')}")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens") or 0
        cost = usage.get("cost") or 0.0
        served_model = data.get("model", self._model)
        served_provider = data.get("provider", "?")
        self.usage.record(prompt_tokens, completion_tokens, reasoning_tokens, cost)

        logger.info(
            "  LLM response — %.1fs, %d prompt + %d completion tokens "
            "(%d reasoning), $%.5f [%s via %s]",
            elapsed, prompt_tokens, completion_tokens, reasoning_tokens,
            cost, served_model, served_provider,
        )

        # Never fall back to the reasoning text: a truncated or empty answer
        # is a failure, and the caller decides what to skip.
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        content = (choice.get("message", {}).get("content") or "").strip()
        if finish_reason == "length":
            raise RuntimeError(
                f"LLM output truncated at max_tokens={self._max_tokens} "
                f"({completion_tokens} completion, {reasoning_tokens} reasoning)"
            )
        if not content:
            raise RuntimeError(
                f"LLM returned empty content (finish_reason={finish_reason})"
            )

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost": cost,
            "model": served_model,
            "provider": served_provider,
        }
