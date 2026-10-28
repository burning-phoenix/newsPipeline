# Corroborate

A pipeline that ingests RSS feeds from global news outlets (currently 45 feeds from 37 outlets, US and Global, expandable), clusters articles into stories, identifies claims confirmed by 3+ independent sources, and generates briefs separating verified facts from editorial framing.

**Fast and cheap:** a full run takes a minute or two and costs well under a cent.

**Want to see what it produces?** Read the sample brief, [`outputs/OCT_03_2026.md`](outputs/OCT_03_2026.md). It is what you should expect after running the pipeline.

For how the pipeline is designed and how to extend it, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

With [uv](https://docs.astral.sh/uv/) (recommended). It installs the exact locked versions the thresholds were tuned against, and fetches Python 3.12 if you don't have it:

```bash
uv sync
uv run python -c "import nltk; nltk.download('punkt_tab')"
```

Then prefix the commands below with `uv run`, for example `uv run python -m corroborate`.

Or with plain pip (Python 3.12+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

Create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Usage

```bash
# Run with all feeds
python -m corroborate

# Show step-by-step progress
python -m corroborate --verbose
```

### Category filtering

Filter feeds by tag to focus on specific coverage areas.

```bash
# Single category
python -m corroborate -c world

# Multiple categories (feeds matching ANY tag)
python -m corroborate -c world politics

# Exclusive mode (feeds matching ALL tags)
python -m corroborate -c world politics --exclusive
```

**Available categories:**

| Tag | Coverage |
|-----|----------|
| `world` | International / global news |
| `us` | United States domestic |
| `europe` | European coverage |
| `asia` | Asia & Asia-Pacific |
| `africa` | African coverage |
| `americas` | Canada, Latin America, Caribbean |
| `politics` | Political reporting |
| `business` | Business & financial news |
| `economics` | Central banks, monetary policy, macro data |
| `investigative` | Investigative journalism |
| `defense` | Defense & security |

### Cluster scope

Only stories covered by 3+ independent outlets are considered, ranked by how many outlets cover them. By default the top 3 are analyzed (full-text fetch, claim detection, analysis, writing). Stories where no claim reaches 3 outlets are dropped. You can change the scope:

```bash
python -m corroborate --top 5        # top 5 stories
python -m corroborate --all          # every story with 3+ outlets
```

### Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Print step-by-step progress to stderr |
| `--category TAG [TAG ...]` | `-c` | Filter feeds by category (default: all) |
| `--exclusive` | | Require ALL categories instead of ANY |
| `--all` | | Analyze every story with 3+ outlets |
| `--top N` | | Number of top stories to analyze (default: 3, ignored with `--all`) |
| `--workers N` | | Concurrent article fetches and LLM calls (default: 16) |
| `--output [PATH]` | `-o` | Also save the brief to a file (see below) |

### Output

Briefs print to stdout. Progress (when `--verbose`) prints to stderr. This means you can pipe briefs to a file while still seeing progress:

```bash
python -m corroborate -c world --verbose > briefs.txt
```

To save the brief for you, use `--output`. With no path it is named by date, `MON_DAY_YEAR.md`, and written to `outputs/` (for example `outputs/OCT_03_2026.md`). A run on the same day overwrites the earlier file. Give a path and it is used exactly as written, with no renaming:

```bash
python -m corroborate -o                  # outputs/OCT_03_2026.md
python -m corroborate -o my_brief.md      # my_brief.md
```

Everything in `outputs/` is git-ignored except the sample brief.

## What the pipeline does

1. **Fetches** all RSS feeds in parallel, filters to the last 24 hours, and drops duplicate links
2. **Embeds** article titles + summaries into vectors
3. **Clusters** articles into stories using cosine similarity

Then, for every selected story at once:

4. **Extracts** full article text
5. **Detects** claims confirmed by 3+ independent outlets via sentence-level embedding similarity
6. **Analyzes** the articles for framing and for where accounts differ: conflicts, differing figures, and details only some outlets report (cheap model, JSON notes checked in code)
7. **Writes** the story's section from the verified claims and those notes (prose model): what happened, how outlets framed it, and where their accounts differ

Alongside the stories, central-bank and institutional releases are written up as an economics section. Finally a short overview is written and the document is assembled in code, with a sources line.

Attributions such as "(BBC News, NPR and others)" are built in code from each claim's confirming outlets, in a stable order that doesn't favor any outlet. The writer copies them verbatim, and any parenthetical that isn't one of them is logged as a warning.

## Models

All calls go through OpenRouter. Each role is set in `ModelSettings` in `corroborate/config.py`:

| Role | Model | Why |
|------|-------|-----|
| Embeddings | `qwen/qwen3-embedding-8b` | Clustering and claim matching |
| Story analysis | `deepseek/deepseek-v4-flash-0731` | Reads every article (the bulk of the tokens); output is notes, not prose |
| Sections, digest, overview | `z-ai/glm-5.3-flash` | Prose quality; sees only claims and notes (~2.5k tokens per story) |

If a model fails twice (rate limit, outage), the request is sent to `minimax/minimax-m3.1-flash` instead. If that fails too, an analysis is skipped and the story is written from its verified claims alone (no framing paragraph). Adding your own provider key under OpenRouter's integration settings gives you your own rate limits.

Requests go to fp8 providers that support the request's parameters. The price ceiling is relative: at startup the pipeline reads each model's provider list and sets the ceiling just high enough to admit the three cheapest of those providers, so it keeps working when a provider leaves or changes price. Requests are spread across them, weighted toward the cheapest. Every call logs its model, provider, tokens and cost, and `--verbose` shows each route's ceiling and ends with a usage and cost summary.

## Tests

```bash
python -m unittest discover -s tests
```

## Feeds

Feeds are configured in `feeds.yaml`. Each feed has an `outlet` (the publication), a `name` (specific feed), and `tags` (categories). Multiple feeds from the same outlet are treated as one source for claim verification. Feeds tagged `source_type: primary` (central banks, institutions) skip clustering and feed the economics section.

45 feeds across 37 outlets covering North America, Europe, Asia, Africa, Latin America, economics/monetary policy, and global investigative sources.

## License

Licensed under the GNU Lesser General Public License v2.1. See [LICENSE](LICENSE).
