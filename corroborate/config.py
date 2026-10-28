from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class ModelSettings:
    """One LLM role: which model, how hard it thinks, and where it may run.

    The price ceiling is relative: at startup it is set to admit the
    `ceiling_providers` cheapest providers that can serve the request
    (quantization, parameters, throughput). A fixed ceiling broke when the
    cheapest provider stopped serving a model: one provider was left under
    it, and its rate limit failed every call. Within the ceiling, requests
    are load-balanced toward the cheapest, and fall back to the others.

    max_price_* (USD per million tokens) are used only if that lookup fails.
    """
    model: str
    reasoning_effort: str = "low"
    max_tokens: int = 4000
    # The request asks for a JSON object; providers must support it.
    json_mode: bool = False
    ceiling_providers: int = 3
    # GLM 5.3 Flash: Z.AI's own endpoint is $0.15 / $0.50.
    max_price_prompt: float = 0.16
    max_price_completion: float = 0.55
    quantizations: list[str] = field(default_factory=lambda: ["fp8"])
    # Takes over the request after this model fails twice.
    fallback: ModelSettings | None = None
    # Cheapest is not always usable: the cheapest V4 Flash provider ran at
    # ~21 tok/s. Providers below this median tok/s go to the back of the line.
    # Kept under the cheap fp8 providers' range (GMICloud for GLM hovers
    # around 50, DeepInfra for V4 Flash around 40): at 50 their median
    # dipping just under sent every call to a provider with twice the price.
    min_throughput: int | None = 30

    def required_parameters(self) -> list[str]:
        """Request parameters every provider in the route must support."""
        params = ["max_tokens", "reasoning"]
        if self.json_mode:
            params.append("response_format")
        return params

    def provider_routing(
        self, max_price: tuple[float, float] | None = None
    ) -> dict:
        """OpenRouter routing; max_price is (prompt, completion) per 1M.

        No "sort": OpenRouter's default load-balances across the providers
        under the ceiling, weighted toward the cheapest, instead of sending
        every request to the single cheapest one.
        """
        prompt, completion = max_price or (
            self.max_price_prompt, self.max_price_completion,
        )
        routing = {
            "quantizations": self.quantizations,
            "require_parameters": True,
            "allow_fallbacks": True,
            "max_price": {"prompt": prompt, "completion": completion},
        }
        if self.min_throughput:
            routing["preferred_min_throughput"] = {"p50": self.min_throughput}
        return routing


FALLBACK_MODEL = "minimax/minimax-m3.1-flash"


def _fallback(**overrides) -> ModelSettings:
    """MiniMax M3.1 Flash, for a role whose own model fails twice.

    Static ceiling (used only if the price lookup fails): its regular
    price, MiniMax's own endpoint, is $0.15 / $0.60.
    """
    return ModelSettings(
        FALLBACK_MODEL, max_price_prompt=0.16, max_price_completion=0.65,
        **overrides,
    )


@dataclass
class PipelineConfig:
    feeds_path: str = "feeds.yaml"
    prompts_dir: str = "prompts"
    embedding_model: str = "qwen/qwen3-embedding-8b"
    # Cheap model for the input-heavy step: reading every article of a
    # story and returning structured notes nobody reads directly. Reasoning
    # off: at "low" or even "minimal" V4 Flash reasoned past 6000 tokens
    # and was truncated; with "none" it returns the notes in seconds.
    # Static ceiling: the third-cheapest fp8 provider with JSON mode,
    # StreamLake, on 2026-10-27.
    analysis_llm: ModelSettings = field(default_factory=lambda: ModelSettings(
        "deepseek/deepseek-v4-flash-0731", reasoning_effort="none",
        max_tokens=4000, json_mode=True,
        max_price_prompt=0.14, max_price_completion=0.40,
        fallback=_fallback(reasoning_effort="none", max_tokens=4000, json_mode=True),
    ))
    # Better prose model for everything the reader sees.
    writer_llm: ModelSettings = field(default_factory=lambda: ModelSettings(
        "z-ai/glm-5.3-flash",
        fallback=_fallback(),
    ))
    # GLM cannot disable reasoning; "minimal" is the closest to off.
    overview_llm: ModelSettings = field(default_factory=lambda: ModelSettings(
        "z-ai/glm-5.3-flash", reasoning_effort="minimal", max_tokens=1000,
        fallback=_fallback(reasoning_effort="minimal", max_tokens=1000),
    ))
    embedding_batch_size: int = 50
    embedding_concurrency: int = 8
    min_combined_words: int = 15
    min_full_text_words: int = 50
    # Words of each article the analyst reads. The biggest story (28 articles)
    # sent a request the cheap analysis provider kept refusing with 429.
    analysis_max_article_words: int = 800
    min_sentence_words: int = 8
    max_age_hours: int = 24
    claim_similarity_threshold: float = 0.75
    claim_min_sources: int = 3
    # The writer gets the most-corroborated claims only: a section is
    # 150-300 words, and big stories produce 20-30 claims.
    max_claims_per_story: int = 8
    top_clusters: int = 3
    all_clusters: bool = False
    max_workers: int = 16
    feed_workers: int = 16
    feed_timeout: float = 15.0
    clustering_thresholds: list[float] = field(
        default_factory=lambda: [0.35, 0.30, 0.25, 0.20, 0.15]
    )
    api_key: str = ""
    categories: list[str] = field(default_factory=list)
    exclusive: bool = False

    @classmethod
    def load(cls, project_root: Path | None = None) -> PipelineConfig:
        if project_root is None:
            project_root = Path.cwd()
        load_dotenv(project_root / ".env")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        return cls(
            feeds_path=str(project_root / "feeds.yaml"),
            prompts_dir=str(project_root / "prompts"),
            api_key=api_key,
        )

    def load_feeds(self) -> list[dict]:
        with open(self.feeds_path) as f:
            all_feeds = yaml.safe_load(f)["feeds"]

        if not self.categories:
            return all_feeds

        requested = set(self.categories)
        filtered = []
        for feed in all_feeds:
            feed_tags = set(feed.get("tags", []))
            if self.exclusive:
                if requested.issubset(feed_tags):
                    filtered.append(feed)
            else:
                if requested & feed_tags:
                    filtered.append(feed)
        return filtered

    def load_prompt(self, relative_path: str) -> str:
        """Read a prompt file, filling in {min_sources} from config."""
        with open(Path(self.prompts_dir) / relative_path) as f:
            return f.read().replace("{min_sources}", str(self.claim_min_sources))
