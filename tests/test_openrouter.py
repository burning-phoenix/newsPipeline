import unittest
from unittest.mock import Mock, patch

from corroborate.providers.openrouter import (
    OpenRouterLLM,
    _post_with_retry,
    _retry_delay,
    ceiling_for_cheapest,
)


def _response(status: int, retry_after: str | None = None) -> Mock:
    resp = Mock()
    resp.status_code = status
    resp.headers = {"retry-after": retry_after} if retry_after else {}
    return resp


class RetryDelayTest(unittest.TestCase):

    def test_429_backs_off_from_five_seconds_with_jitter(self):
        for attempt, base in [(0, 5), (1, 10), (2, 20)]:
            delay = _retry_delay(_response(429), attempt)
            self.assertTrue(base <= delay <= base + 2)

    def test_429_delay_is_capped(self):
        self.assertLessEqual(_retry_delay(_response(429), 10), 62)

    def test_retry_after_header_wins(self):
        self.assertEqual(_retry_delay(_response(429, "7"), 0), 7.0)

    def test_server_errors_keep_the_short_backoff(self):
        self.assertEqual(_retry_delay(_response(503), 2), 4.0)
        self.assertEqual(_retry_delay(None, 1), 2.0)


class RetryBudgetTest(unittest.TestCase):

    def _always_429(self):
        client = Mock()
        client.post.return_value = _response(429)
        client.post.return_value.text = "rate limited"
        return client

    def test_gives_up_before_the_wait_budget_is_exceeded(self):
        slept = []
        client = self._always_429()
        with patch("corroborate.providers.openrouter.time.sleep", slept.append):
            with self.assertRaises(RuntimeError):
                _post_with_retry(client, "u", {}, {}, 5, "LLM", max_wait=20.0)
        self.assertLessEqual(sum(slept), 20.0)
        self.assertLess(client.post.call_count, 5)

    def test_succeeds_after_a_retry(self):
        ok = _response(200)
        limited = _response(429)
        limited.text = ""
        client = Mock()
        client.post.side_effect = [limited, ok]
        with patch("corroborate.providers.openrouter.time.sleep"):
            self.assertIs(_post_with_retry(client, "u", {}, {}, 5, "LLM"), ok)


def _endpoint(name, prompt, completion, quant="fp8", params=("max_tokens", "reasoning", "response_format"), p50=100):
    return {
        "provider_name": name,
        "quantization": quant,
        "supported_parameters": list(params),
        "throughput_last_30m": {"p50": p50},
        "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)},
    }


class CeilingForCheapestTest(unittest.TestCase):

    PARAMS = ["max_tokens", "reasoning", "response_format"]

    def _ceiling(self, endpoints, count=3):
        return ceiling_for_cheapest(endpoints, count, ["fp8"], self.PARAMS, 30)

    def test_admits_the_three_cheapest(self):
        prompt, completion, names = self._ceiling([
            _endpoint("A", 0.06, 0.18), _endpoint("B", 0.13, 0.28),
            _endpoint("C", 0.132, 0.396), _endpoint("D", 0.14, 0.28),
        ])
        self.assertEqual(names, ["A", "B", "C"])
        self.assertAlmostEqual(prompt, 0.132, places=3)
        self.assertAlmostEqual(completion, 0.396, places=3)
        self.assertGreaterEqual(prompt, 0.132)

    def test_skips_providers_that_cannot_serve_the_request(self):
        _, _, names = self._ceiling([
            _endpoint("fp4", 0.01, 0.01, quant="fp4"),
            _endpoint("no-json", 0.02, 0.02, params=("max_tokens", "reasoning")),
            _endpoint("slow", 0.03, 0.03, p50=10),
            _endpoint("A", 0.06, 0.18), _endpoint("B", 0.13, 0.28),
        ])
        self.assertEqual(names, ["A", "B"])

    def test_none_when_nothing_qualifies(self):
        self.assertIsNone(self._ceiling([_endpoint("fp4", 0.01, 0.01, quant="fp4")]))


class ModelFallbackTest(unittest.TestCase):

    def _ok(self, model):
        resp = _response(200)
        resp.json.return_value = {
            "model": model, "provider": "P",
            "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.001},
        }
        return resp

    def test_routes_to_fallback_after_two_failures(self):
        fallback = OpenRouterLLM("k", "fallback/model")
        primary = OpenRouterLLM("k", "primary/model", fallback=fallback)
        calls = []

        def post(client, url, headers, body, attempts, label, **_):
            calls.append((body["model"], attempts))
            if body["model"] == "primary/model":
                raise RuntimeError("LLM request failed after 2 attempts: HTTP 429")
            return self._ok(body["model"])

        with patch("corroborate.providers.openrouter._post_with_retry", post):
            result = primary.complete("s", "u")
        self.assertEqual(calls[0], ("primary/model", 2))
        self.assertEqual(calls[1][0], "fallback/model")
        self.assertEqual(result["model"], "fallback/model")
        self.assertEqual(primary.usage.cost, 0.001)

    def test_without_fallback_the_error_propagates(self):
        llm = OpenRouterLLM("k", "only/model")
        with patch("corroborate.providers.openrouter._post_with_retry",
                   side_effect=RuntimeError("HTTP 429")):
            with self.assertRaises(RuntimeError):
                llm.complete("s", "u")
