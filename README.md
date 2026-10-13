# Hangzhou Mahjong AI

English | [简体中文](README.zh-CN.md)

A Python client for **【杭州麻将对战平台】**, with the fixed **One-Ply Expectation Policy**, three play modes, and recoverable local records.

## Quick start

### 1. Install

Use Python **3.12+** and run the commands from the repository root. Installation builds a C extension: Windows AMD64 uses the declared Zig build dependency; Linux needs a C compiler and matching Python development headers, plus Python's venv support.

**Windows — PowerShell**

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --editable .\heuristic_ai
if (-not (Test-Path -LiteralPath '.\heuristic_ai\config.yaml')) {
    Copy-Item -LiteralPath '.\heuristic_ai\config.example.yaml' -Destination '.\heuristic_ai\config.yaml'
}
```

**Linux — Bash**

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --editable ./heuristic_ai
if [ ! -e ./heuristic_ai/config.yaml ]; then
    cp ./heuristic_ai/config.example.yaml ./heuristic_ai/config.yaml
fi
```

These commands preserve an existing configuration. Use the same virtual environment for installation and execution; build the environment and extension on the target operating system.

### 2. Configure a mode

Edit `heuristic_ai/config.yaml`, using [config.example.yaml](heuristic_ai/config.example.yaml) as the complete template. Replace the `REPLACE_ME_...` values with your own platform connection details, room identifiers, and credentials. The local configuration is ignored by Git.

Keep `schema_version: 3`. **Both `train` and `race` must be complete even when another mode is selected.** Every supplied section is validated, including `match` when present. Only the selected mode is run. The template selects `race` by default.

| Mode | Purpose and required identity | Completion behavior |
|---|---|---|
| `train` | Four distinct player identities in one existing test room; configure its room ID and four unique names/tokens. This is room play, not model training. | Runs one batch and exits. The room must be `registering` or `finished` at startup; the client does not create or continuously reopen rooms. |
| `race` | An already registered tournament participant; configure its room ID and participant token. | Follows ready, stages, and assigned games until an authoritative terminal state or elimination, or until interrupted. Initial registration is done on the platform. |
| `match` | An existing global identity token, supplied as top-level `global_token`. No matching-room ID is configured. | One session by default. With `match.continuous: true`, the next session starts after the current session finishes and its records are published. |

For `match`, set these **top-level fields in the copied template**, keeping its existing sections:

```yaml
mode: match
global_token: REPLACE_ME_GLOBAL_TOKEN
```

Fill the `server.url` of every retained section. `server.insecure: true` skips TLS certificate verification; the template currently uses this value. Relative `competition.output_dir` paths are resolved from the configuration file's directory. The template's `../competition_record` therefore points to the repository's `competition_record/` directory.

### 3. Start and stop

After completing the configuration, use one launch command:

**Windows — PowerShell**

```powershell
& .\.venv\Scripts\python.exe -B .\heuristic_ai\launch.py
```

**Linux — Bash**

```bash
./.venv/bin/python -B ./heuristic_ai/launch.py
```

[launch.py](heuristic_ai/launch.py) reads the `config.yaml` beside it, independent of the working directory. Configuration placeholders are rejected before network access. The mode comes from YAML; there is no `--mode` flag.

To use another complete configuration, replace the script entry with the module entry:

- Windows: `& .\.venv\Scripts\python.exe -B -m heuristic_ai.runtime.runner --config "REPLACE_ME_CONFIG_PATH"`
- Linux: `./.venv/bin/python -B -m heuristic_ai.runtime.runner --config "REPLACE_ME_CONFIG_PATH"`

Progress is written to **stderr**. Press `Ctrl+C` to stop the foreground run and let it close workers and records. Stop the current process before changing modes. Stopping the client does not close the server-side room.

## Technical overview

### Code layout

| Module | Responsibility |
|---|---|
| [config](heuristic_ai/src/heuristic_ai/config/) | Strict YAML loading, schema checks, typed settings, and placeholder rejection. |
| [protocol](heuristic_ai/src/heuristic_ai/protocol/) | Authentication headers, API requests, HTTP transport, retry policy, and snapshot normalization. |
| [runtime](heuristic_ai/src/heuristic_ai/runtime/) | Mode coordination, game workers, I/O processes, action deadlines, supervision, recovery, and progress. |
| [game](heuristic_ai/src/heuristic_ai/game/) | Tiles, observations, legal actions, scoring, hand solving, and public inventory. |
| [strategy](heuristic_ai/src/heuristic_ai/strategy/) | One-Ply Expectation search, static evaluation, precomputation, caches, and native kernels. |
| [artifacts](heuristic_ai/src/heuristic_ai/artifacts/) | Decision records, working shards, validated exports, replay parsing, and recovery cleanup. |

### State, decisions, and submission

The play modes share the same game rules and decision controller:

```text
SSE notifications / state polling
  -> validated snapshot and public inventory
  -> legal action candidates
  -> One-Ply Expectation decision or fallback
  -> action submission and confirmation
  -> decision records and run export
```

[io_process.py](heuristic_ai/src/heuristic_ai/runtime/io_process.py) runs network work in a separate process. HTTP clients and their asynchronous event loop stay in that process, apart from strategy computation. In [httpx_transport.py](heuristic_ai/src/heuristic_ai/protocol/httpx_transport.py), notification connections allow HTTP/2 negotiation; state, action, and control channels use HTTP/1.1.

SSE notifications carry sequence hints, not an authoritative board or action-window timestamp. The client obtains state snapshots before generating actions. Public inventory accounts for visible physical tiles, including claimed discards, to avoid counting one tile twice.

Safe reads have bounded retries. Action submissions are not automatically replayed after a lost acknowledgement, and a successful HTTP response is distinct from confirmation that the intended action was executed. Recovery preserves confirmed or uncertain submissions and the original action budget.

### One-Ply Expectation Policy

The policy identifier is [`one_ply_expectation`](heuristic_ai/src/heuristic_ai/strategy/models.py). Strategy selection is not configurable: the search limit is **20,000 logical nodes**, and caching is enabled.

[one_ply_search.py](heuristic_ai/src/heuristic_ai/strategy/search/one_ply_search.py) evaluates each legal candidate and any mandatory discard after a claim, then projects the player's next draw. Each possible tile is weighted by `max(0, 4 - known_count)`. A leaf takes a legal win when available; otherwise it evaluates discards. Weighted value vectors are compared lexicographically, with static evaluation used to break ties.

The vector includes normalized winning income, nonterminal progress, live-tile weight, fan potential, chain potential, and route count. These are components of a local decision model, not a prediction of final four-player match results.

The model consumes only available public information and the player's own hand. It accounts for intervening draws in wall consumption but does not simulate opponents' hidden hands or actions. Unseen copies may be held by opponents, so their weights are not a known wall distribution.

Precomputation runs the same One-Ply Expectation model on anticipated states. Results are reused only after identity, generation, and input checks. The [C extension](heuristic_ai/src/heuristic_ai/strategy/native/_one_ply_native.c) provides leaf evaluation and JSON helpers; runtime code can use Python implementations when the extension is unavailable.

| Execution source | Meaning |
|---|---|
| `completed_search` | A complete comparison of all root candidates. |
| `prefetch_hit` | Reuse of a validated, complete precomputed decision. |
| `forced_action` | Only one legal candidate is available. |
| `static_fallback` | Static evaluation when the search cannot deliver a usable result. |
| `legal_fallback` | Deterministic legal-action fallback when a richer evaluation cannot be used. |

Unknown public inventory, insufficient history, exhausted budgets, or computation failures can lead to a recorded fallback. A fallback is an internal execution path of the One-Ply Expectation Policy, not another selectable strategy.

### Budgets and recovery

| Setting | Current template/default | Interpretation |
|---|---|---|
| `competition.decision_budget_ms` | `train`: 3000; `race`: 2000; `match`: 3000 | Client decision budget; the available action window can shorten it. |
| Action submission reserve | 300 ms | Internal default for discard and response submission, not a YAML field. |
| `match.matchmaking.max_parallel_games` | 10 | Configurable client capacity, validated from 1 to 16. |
| `match.matchmaking.max_rounds_per_game` | 8 | Configured upper bound sent for matching; the actual room size comes from the platform. |
| `race.supervision` | Enabled; heartbeat timeout 30 s, stall timeout 180 s | Foreground supervisor settings, not server-side action timeouts. |

Search respects both its node limit and remaining action time. Duplicate snapshots, retries, and worker recovery do not grant a fresh long-search budget for the same action opportunity. Configured budgets and table limits are client settings; room rules and game outcomes come from platform responses.

Game recovery is tracked per game and generation. It stops the affected workers, retains submission state and recovery evidence, and coordinates a replacement generation after backoff. Formal play additionally uses a foreground parent supervisor for heartbeat and stall checks. On Windows, the CLI permits only one `race` instance per machine.

### Records and offline recovery

Under the configured output root, the main paths are:

| Path | Contents |
|---|---|
| `.record_work/<mode>/<run-id>/` | Owned working shards used during recording and recovery. |
| `test_records/<run-id>/` | Default `train` and `match` exports. |
| `official_records/<run-id>/` | `race` exports; also `train` when `save_to_official: true`. |
| `runtime_logs/race/<supervision-id>/` | Supervisor logs and run associations. |

Published runs contain `manifest.json` and `decisions.jsonl`; recovery events add `recovery.jsonl`. Train exports also include round replays and state-poll records. Race and match exports do not contain the same complete four-player replay data.

Run state (`completed`, `degraded`, `failed`, or `interrupted`) and export completeness are separate. Credentials and sensitive authentication fields are redacted at serialization boundaries, but records can still contain player names and user/game identifiers.

[artifacts.cli](heuristic_ai/src/heuristic_ai/artifacts/cli.py) provides two separate offline operations. Stop the run before opening its working shards for recovery. Replace the path placeholders with a managed working directory and, for an explicit output, a new destination.

**Merge/export — Windows**

```powershell
& .\.venv\Scripts\python.exe -B -m heuristic_ai.artifacts.cli merge --work-dir "REPLACE_ME_WORK_DIR" --output-dir "REPLACE_ME_OUTPUT_DIR"
```

**Merge/export — Linux**

```bash
./.venv/bin/python -B -m heuristic_ai.artifacts.cli merge --work-dir "REPLACE_ME_WORK_DIR" --output-dir "REPLACE_ME_OUTPUT_DIR"
```

`merge` verifies retained shards and writes a fresh export; `--output-dir` is optional. After export, eligible completed/degraded runs may have their working material cleaned automatically. Failed/interrupted runs retain recovery material.

**Retry cleanup — Windows**

```powershell
& .\.venv\Scripts\python.exe -B -m heuristic_ai.artifacts.cli cleanup --work-dir "REPLACE_ME_WORK_DIR"
```

**Retry cleanup — Linux**

```bash
./.venv/bin/python -B -m heuristic_ai.artifacts.cli cleanup --work-dir "REPLACE_ME_WORK_DIR"
```

`cleanup` retries deletion only after validating the published export and cleanup evidence. Use it for pending cleanup, not as a required step after every merge. Both commands report JSON, perform local filesystem operations, and do not read platform credentials or start network play.
