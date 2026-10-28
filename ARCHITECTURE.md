# Architecture

Corroborate turns a day of RSS coverage into a short brief that separates what several independent outlets confirm from how each outlet frames it. This document covers how the code is organized, why it is built this way, and how to extend it. For setup and usage, see the [README](README.md).

## Design goals

1. **Facts must be corroborated, not asserted.** A sentence enters the "verified" part of a brief only if 3+ independent outlets say it. That check is done with embeddings in code, not by asking a model.
2. **Models write; code decides.** Selection, verification, attribution and document assembly are deterministic. Models are used only where language is needed: noting framing and differences, and writing prose.
3. **Cheap and fast.** A full run takes a minute or two and costs well under a cent. Expensive models never read raw articles.
4. **Swappable parts.** Every external dependency (embeddings, LLMs, article extraction, clustering) sits behind a small interface, so it can be replaced without touching the stages.

## Flow

```
Feeds (parallel) ──┬─▶ Economics digest (writer) ────────────────────────────┐
                   └─▶ Embed entries → Cluster → Select (≥3 outlets)          │
                         └─▶ per story, all concurrently:                     │
                              fetch → claims [0 claims → drop] → analysis     │
                              → section (+ attribution check)                 │
                                                                              ▼
                         wait for all ─▶ Overview ─▶ Render (code)
```

Only two points wait for everything before them: clustering (it needs every embedding) and the overview (it needs every section). The digest depends only on feed collection. Each story's chain depends only on itself, so stories run in parallel.

| Step | Module | Does | Model? |
|------|--------|------|--------|
| 1. Feed collection | `stages/feed_collection.py` | Fetches all feeds in parallel; keeps the last 24 h; drops short editorial entries and duplicate links | No |
| 2. Embedding | `stages/embedding.py` | Embeds `title + summary` in concurrent batches | Embedding model |
| 3. Clustering | `stages/clustering.py` | Cosine distance, threshold sweep, picks the threshold with the most multi-outlet clusters; ranks by outlet count | No |
| 4. Full-text fetch | `stages/full_text_fetch.py` | Extracts article text; falls back to title + summary (paywalls, video pages) | No |
| 5. Claim detection | `stages/claim_detection.py` | Splits sentences, embeds them, matches across outlets (≥ 0.75 similarity, ≥ 3 outlets), merges near-duplicate claims | Embedding model |
| 6. Story analysis | `stages/story_analysis.py` | A cheap model reads the articles and returns JSON notes (framing per outlet, differences); code validates them | Analysis model |
| 7. Writing | `stages/brief_synthesis.py` | Writes each section from claims + checked notes only; also the economics digest and the overview | Writer model |
| 8. Rendering | `stages/rendering.py` | Heading from the clock, overview, sections, digest, sources line | No |

## Package layout

```
corroborate/
  __main__.py          CLI (python -m corroborate): flags, logging, writes the brief
  pipeline.py          Orchestrator: builds providers, injects them into stages, schedules work
  config.py            PipelineConfig (all tunables) + ModelSettings (one per LLM role)
  models.py            Dataclasses passed between stages
  attribution.py       Builds "(Outlet A, Outlet B and others)" strings; flags unverified ones
  stages/              One class per pipeline step (table above)
  providers/
    base.py            Interfaces: EmbeddingProvider, LLMProvider, ContentExtractor, ClusteringStrategy
    openrouter.py      OpenRouter embeddings + chat: routing, retries, model fallback, usage/cost
    trafilatura_extractor.py
    agglomerative.py
prompts/
  constitution.md      Shared epistemic rules for every model
  story_analyst/soul.md
  brief_synthesizer/soul.md
feeds.yaml             Feed list: outlet, feed name, region, tags, source type
tests/                 stdlib unittest
```

### Data model (`models.py`)

`FeedEntry` (one RSS item) → `StoryCluster` (entries about one story) → `EnrichedCluster` (with `Article` full text) → `Claim` (a sentence plus the outlets confirming it) and `StoryNotes` (framing per outlet, `Discrepancy` list) → `StoryBrief` (the finished markdown section). Stages communicate only through these types.

## How the code is wired

The package follows SOLID principles. Each one maps to something concrete:

- **Single responsibility.** Each stage class does one step. Feed collection knows nothing about clustering; claim detection knows nothing about LLMs.
- **Dependency inversion.** Stages receive abstract providers through their constructors and never create their own clients. `Pipeline` is the only place that picks implementations.
- **Open/closed.** A new embedding backend, LLM API, extractor or clustering algorithm is a new provider class; existing stages don't change.
- **Interface segregation.** Each interface has a single job (`embed`, `complete`, `extract`, `fit`).
- **Liskov substitution.** Implementations honor their interface's contract. For example, `LLMProvider.complete` must raise on an empty or truncated answer rather than return it.

### Concurrency

Stages are per-item and contain no threading. The orchestrator owns concurrency:

- Feeds are fetched by a thread pool inside feed collection.
- Stories run in a story pool (one worker per selected story).
- Article fetches run in a **separate** fetch pool. Story workers block waiting on their fetches, so sharing one pool could deadlock.
- The digest runs on a side thread alongside the editorial path.

Each provider owns its API's concurrency limit with a semaphore, so the number of in-flight requests stays bounded no matter how many stages call it.

### Models and routing

Each LLM role has its own `ModelSettings` in `config.py`:

| Role | Model | Why |
|------|-------|-----|
| Analysis | `deepseek/deepseek-v4-flash-0731`, reasoning off, JSON mode | The input-heavy step (reads every article); its output is notes, not prose |
| Sections, digest | `z-ai/glm-5.3-flash`, reasoning low | Prose quality; sees only claims and notes (~2.5k tokens per story) |
| Overview | `z-ai/glm-5.3-flash`, reasoning minimal | Short summary of the sections |
| Fallback (all roles) | `minimax/minimax-m3.1-flash` | Takes over a request after the role's model fails twice |

All calls go through OpenRouter, with routing built in `ModelSettings.provider_routing`:

- **Filters.** fp8 quantization only, `require_parameters` (every provider must support every parameter the request sends, such as JSON mode), and a soft throughput floor (median ≥ 30 tok/s).
- **Relative price ceiling.** At startup, `price_ceiling` reads the model's provider list and sets the ceiling to admit the **three cheapest providers that pass the filters**. The static `max_price_*` values apply only if that lookup fails. A fixed ceiling broke once already: the cheapest provider stopped serving the analysis model, a single provider was left under the ceiling, and its rate limit failed every analysis call.
- **Load balancing.** There is no `sort`, so OpenRouter spreads requests across the providers under the ceiling, weighted toward the cheapest, and falls back to the others when one refuses.
- **Retries and fallback.** 429s back off from 5 s with jitter. A request never spends more than 20 s sleeping (`MAX_RETRY_WAIT`). After two failed attempts the request goes to the fallback model, whose cost is counted under the same role.
- **Strict answers.** A truncated (`finish_reason: length`) or empty answer raises. Reasoning text is never used as output.

## Design decisions and rationale

**Outlet vs. feed.** `feeds.yaml` separates the media organization (`outlet`) from the specific feed (`name`). Several feeds from one outlet count as one source, so a story covered by three BBC feeds is not "corroborated".

**Paywalled sites are evidence, not claim sources.** Outlets such as the NYT join clustering and can confirm a story exists, but their full text usually can't be fetched. They fall back to title + summary, which rarely matches a full sentence elsewhere, so they seldom originate claims.

**One embedding space.** The same `EmbeddingProvider` instance embeds both articles (for clustering) and sentences (for claims), so every vector lives in the same space and thresholds mean the same thing everywhere.

**Claims are worded by the most central sentence.** Among the matching sentences, the one closest to the others becomes the claim's text, so a claim is always a real sentence from a real article and never a model paraphrase.

**Constitution + soul prompting.** Instead of ad-hoc instructions per call, every model gets `prompts/constitution.md` (shared rules: what counts as verified, no inference, no editorializing), plus a role identity (`soul.md`). Prompts are content, not code, and `{min_sources}` is filled from config so the rules and the code never disagree.

**The analyst is checked by code.** The analysis model's JSON is never trusted as is. `StoryAnalysisStage._check` keeps only outlets that are really in the story. It drops "positions" that describe an omission ("did not name…") or an inference ("implying…"). It drops differences that are just rounding (within 2%). It keeps at most 4 differences, conflicts first and at most 2 "only X reported" items.

**Write once.** Each story section is written once, in final form, from claims and checked notes only. The writer never sees the articles. There is no whole-document rewrite, so verified claims and attributions reach the output unchanged.

**Attribution in code.** Attribution strings are built by `attribution.py` from each claim's confirming outlets. They're ordered by a hash of claim + outlet, which is stable but not alphabetical and not feed order, so no outlet is systematically named first. At most 3 outlets are named, then "and others". The writer copies them verbatim, and any parenthetical naming an outlet that isn't one of those strings is logged.

**Three paragraphs per story.** Every section has verified facts, framing, and where accounts differ. Only a `conflict` (two things that can't both be true) is called a disagreement. Differing figures are given side by side or as a range, and unique details are written as "Only X reported…".

**Degrade, don't fail.** A failed feed logs a warning. A failed article fetch falls back to title + summary. A story with no corroborated claim is dropped. A failed analysis means the section is written from claims alone. A failed story is skipped. One bad source never sinks the run.

**Quiet by default.** The brief goes to stdout, and diagnostics go to stderr only with `--verbose`: per-step progress, per-call model, provider and cost, the routing ceilings, stage timings, and a usage summary.

**Saved briefs are named by date.** `-o` writes `outputs/MON_DAY_YEAR.md`, for example `OCT_03_2026.md`. `-o PATH` uses the given path as is.

### Validated parameters

These defaults in `PipelineConfig` were tested on real runs. Don't change them without data.

| Parameter | Value | Note |
|-----------|-------|------|
| `min_combined_words` | 15 | Drops ~2% of editorial entries (mostly title-only) |
| `clustering_thresholds` | 0.35 → 0.15 | 0.35 usually wins |
| `claim_similarity_threshold` | 0.75 | No observed false positives |
| `claim_min_sources` | 3 | Core principle of the project |
| `min_full_text_words` | 50 | Filters video and stub pages |
| `min_sentence_words` | 8 | Filters trivial sentences |
| `embedding_batch_size` | 50 | Reliable with OpenRouter |

## Extending the pipeline

### Add or change feeds
Edit `feeds.yaml`. Give each feed an `outlet` (the organization, which is what counts as a source), a `name`, a `region` and `tags`. Use the same `outlet` for every feed of one organization. Mark institutional sources (central banks, statistics offices) with `source_type: primary`. They then skip clustering and feed the economics digest, and are exempt from the minimum-word filter. New tags work with `--category` automatically.

### Swap a model or add an LLM role
Change the model string in the role's `ModelSettings` in `config.py`. The routing ceiling adapts on its own. Set `json_mode=True` if the role's requests ask for JSON, so only providers that support it are counted, and set `fallback=` to pick a different fallback model. For a new role, add a `ModelSettings` field to `PipelineConfig`, build it in `Pipeline.__init__` with `_make_llm`, and inject it into the stage that needs it. The usage summary picks it up from `Pipeline._llms`.

### Add a provider (new API, local model, other extractor or clustering)
Subclass the interface in `providers/base.py`:

- `EmbeddingProvider.embed(texts) -> (array, tokens)` and `cost_per_million_tokens()`
- `LLMProvider.complete(system, user, json_mode=...) -> {"content", "prompt_tokens", "completion_tokens", ...}`. It must raise on empty or truncated output.
- `ContentExtractor.extract(url) -> str | None`
- `ClusteringStrategy.fit(distance_matrix, threshold) -> labels`

Then construct it in `Pipeline.__init__`. No stage changes are needed. Keep the provider's own concurrency limit inside the provider, as `OpenRouterLLM` does with its semaphore.

### Add a pipeline step
1. Write a stage class in `stages/` with one public method. Take providers and `PipelineConfig` in the constructor, and don't create threads or clients.
2. If it produces new information, add a dataclass to `models.py` and pass it on explicitly.
3. Call it from `Pipeline`. If it is per story, add it to `_story`, which already runs stories concurrently. If it needs every story's result, add it after the story pool, next to the overview.
4. Add tests under `tests/` for its pure logic. Model calls can be faked with a stub `LLMProvider`.

The planned **syndication detection** step is a good example. Wire copy (AP, Reuters, AFP) republished by several outlets currently counts as independent confirmation. A stage between claim detection and analysis can group near-identical articles (embedding similarity ≥ 0.95) into provenance groups and count each group as one source.

### Change the writing
Edit the prompts, not the code: `prompts/constitution.md` for shared rules, and each role's `soul.md` for identity and output format. Structural rules that must always hold, such as attribution strings, the three paragraphs and outlet validation, are enforced in code, so prompt changes can't break them.

### Change the output
`stages/rendering.py` assembles the document, and `__main__.py` decides where it goes. A different format (HTML, JSON for an app) is a new renderer, with the stages untouched.

## Testing

```bash
python -m unittest discover -s tests      # or: uv run python -m unittest discover -s tests
```

Tests cover the deterministic parts: feed filtering, claim detection logic, attribution, the analyst-notes checks, rendering, output naming, retry and backoff behavior, the relative price ceiling, and model fallback. They make no network calls.

## Known limitations

- **Paywalls.** Paywalled outlets fall back to title + summary, which helps clustering but adds little to claims or framing.
- **Title-only feeds.** Some feeds, including several central banks, publish titles only, so the economics digest can say little about the content of a release.
- **Wire copy.** Syndicated wire copy can inflate corroboration until syndication detection exists.
- **Analysis failure.** When analysis fails for a story, its section has no framing paragraph, and the third paragraph currently says the accounts are consistent even though nothing was compared.
- **Provider churn.** OpenRouter providers come and go and run promotions. The relative ceiling absorbs this, and `--verbose` shows where each call went and what it cost.
