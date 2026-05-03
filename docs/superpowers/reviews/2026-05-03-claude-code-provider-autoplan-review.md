# /autoplan Review — `anyllm-claude-code` Provider

**Date:** 2026-05-03
**Reviewer mode:** `/autoplan` (CEO + Eng + DX phases). Single-voice (Codex CLI unavailable — `[subagent-only]` tag throughout).
**Artifacts under review:**
- Spec: `docs/superpowers/specs/2026-05-03-claude-code-provider-design.md` (commit `d0b5a7b`)
- Plan: `docs/superpowers/plans/2026-05-03-claude-code-provider.md` (commit `d4f6e32`)
**Restore point:** `~/.gstack/projects/sagikazarmark-anyllm/main-autoplan-restore-20260503-095709.md`

---

## Executive Summary

Three independent reviewers (CEO/strategic, Eng/architecture, DX/developer-experience)
produced overlapping critiques of the implementation plan:

| Phase | Verdict | Headline |
|---|---|---|
| CEO | **RETHINK PREMISE** | Demand unvalidated; CLI surface unstable; scope 2-3x oversized for evidence. |
| Eng | **SHIP WITH FIXES (substantial)** | 27 plan-correctness bugs + unsound `CallGuard::Drop` + dead error-classification path + missing tests on the most failure-prone code. |
| DX | **SHIP WITH FIXES (3 blockers)** | Hidden CLI install prereq; tool-handler-missing silent failure; mock story regression vs sibling. **DX scorecard: 4.4/10.** |

**Cross-phase themes** (concerns flagged independently in all three phases — highest-confidence signals):

1. **Scope is too large for evidence.** All three reviewers recommend a smaller v1.
2. **Tool bridging is the worst cost/risk surface.** Largest implementation cost (Eng), worst silent-failure mode (DX), most fragile interface (CEO). Strong cross-phase signal to drop tools from v1.
3. **Wrapper is fragile against CLI drift.** Top regret scenario (CEO); lockdown env vars and stream-json schema are unverified (Eng); version pin lives only in a code comment (DX).

---

## Phase 1 — CEO Review

### Premises identified (with evaluation)

1. **"Users with a Claude Code subscription want to drive it programmatically through `anyllm`."**
   *Plausible but unmeasured.* Spec asserts demand in §1 without evidence — no GitHub issues cited, no user requests. The maintainer wants this; unclear anyone else does.

2. **"The `claude` CLI's stream-json input/output formats are stable enough to wrap as a portable provider surface."**
   *Load-bearing assumption with no contractual backing.* Anthropic ships Claude Code on a fast cadence and has never committed to stream-json as a public stable interface. Spike (Task 1) is gating, but the *premise* is that the schema, env-var lockdown set, and `--mcp-config` inline-JSON support remain stable across CLI versions.

3. **"Wrapping a subscription CLI 'bends but does not break' Anthropic's ToS, and a README disclaimer is sufficient mitigation."**
   *Aspirational mitigation.* Published crates on crates.io get used however users want. This is a legal/relationship-with-Anthropic premise dressed up as a technical decision.

4. **"A pluggable `Sandbox` trait now is worth designing now even though only `NoSandbox` ships."**
   *YAGNI alarm.* Five sandbox impls deferred to "future work"; trait surface shaped around hypothetical bwrap/firejail/landlock/sandbox-exec needs without a single concrete implementation to validate the abstraction. The `SpawnSpec` design already had to be revised mid-spec — that is a *signal* the abstraction is premature.

5. **"Per-call subprocess startup latency is acceptable."**
   *Acceptable for whom?* Acceptable for an `anyllm` user testing locally. Painful for any iterative use (test loops, evaluation harnesses, agent prototypes). Spec acknowledges in passing and defers persistent sessions to "future work" — but the entire portable-provider framing depends on users *not* noticing the cost gap.

6. **"The `--disallowed-tools` exhaustive list approach is maintainable."**
   *Constant tax.* Anthropic adds tools regularly. Every CLI release potentially silently grants Claude a tool the wrapper didn't disable — found out when a user reports their disk got scribbled on.

### Problem reframings the plan dismisses

**Reframe A — Personal tool, not workspace crate.** Don't ship `anyllm-claude-code` to crates.io. Build as an *example* under `crates/anyllm-examples/` or as a separate repo. Eliminates maintenance contract on a brittle CLI wrapper, removes "looks official" perception that invites production use, avoids putting the project's name behind a ToS-bending artifact.

**Reframe B — User's real problem is "test/prototype against Claude without burning API budget".** If true, the right answer is *not* a CLI wrapper — it's better mock fixtures, a recording/replay layer (VCR-style for `anyllm-anthropic`), or a "bring your own credits" cookbook. CLI wrapper solves billing-source by accident, while incurring 100% of the wrapping cost.

**Reframe C — User wants Claude Code's *agentic* capabilities (planning, tool orchestration, file ops) accessible from Rust.** This design actively defeats it — `--disallowed-tools` strips out everything that makes Claude Code valuable beyond plain Anthropic Messages. Wrapping a $200/mo agent harness to produce a $20/mo chat completion. If this is the real desire, the CLAUDE.md "no agent loops" rule says don't build it here at all.

### 6-month regret scenarios

- **CLI schema drift.** Anthropic renames `result.subtype` to `result.kind` in Claude Code 2.x. Every user gets `Error::UnexpectedResponse`. Bug reports for a crate the maintainer can't fully test without a live subscription. Conformance fixtures lock in a snapshot that diverges from reality.
- **`--disallowed-tools` list incomplete.** Anthropic adds `Browser`, `ComputerUse`, or some new built-in. User runs `anyllm-claude-code` in a CI job with credentials, Claude calls the new built-in, exfiltration follows. README's "personal/testing only" framing is irrelevant for the post-mortem.
- **ToS update.** Anthropic explicitly forbids headless wrapping in a Terms refresh. Crate becomes a liability the maintainer is morally obligated to yank. Yanked crates damage `anyllm`'s reputation as a stable foundation.
- **Anthropic ships an official Claude Code SDK** (likely — Cursor, Cline, etc. are pressuring this) that exposes proper subscription-equivalent endpoints. Crate becomes obsolete the day it's released; everyone who stayed waiting ate the migration cost.
- **22-task / 4000-line investment never amortizes.** Six months in, crate has 47 stars, three contributors trying to reverse-engineer schema changes, maintainer wishing they'd shipped it as an example.

### Alternatives the plan does not seriously engage

- Ship as an example only (not even mentioned).
- Thin shim that returns text only — drop tools (no MCP), drop streaming (no NDJSON parser), drop multi-turn fidelity. ~200 lines instead of 4000. Sufficient for "prototype against my subscription". Rejected implicitly by full ChatProvider parity goal.
- Wait for official SDK.
- Document `anyllm-anthropic` + cookbook recipe. Doesn't help subscription-only users but ships today and might cover 60% of actual demand.
- Build it but don't publish (`publish = false`).

### Competitive risk

- **Anthropic-shipped SDK risk: High.** Claude Code already has public-ish SDK headlines. Within 6 months Anthropic likely ships an official Rust-or-language-agnostic subscription-API-key path. If they do, this crate is obsolete day one.
- **Existing wrappers in other ecosystems** (Python `claude-code-sdk`, TypeScript bindings) already exist. The Rust niche is small enough that a single competing project saturates it.
- **Timing is poor.** Building a wrapper *while* the underlying tool is in heavy active development is the worst window.

### Scope calibration

22 tasks / ~4000 LOC is **2-3x too large** for a v1 of an experimental, ToS-marginal wrapper. Recommended cuts:

- v1 should be: chat one-shot + streaming, no tools, no MCP server, no Sandbox trait. ~600-1000 LOC across 6-8 tasks.
- The MCP server is the highest-cost component AND the largest fragility source.
- The Sandbox trait is pure speculation. Don't need a trait for one impl — need a `cmd_wrapper: Option<Box<dyn Fn>>` field added when first concrete need exists.
- Conformance theater on a partially-conforming provider is its own anti-pattern.

### CLAUDE.md adherence

- **"Building block, not agent framework"**: Honored. Plan strips Claude Code's agentic capabilities via `--disallowed-tools`. (Though see Reframe C — also strips the *reason to use Claude Code over the regular API*.)
- **"Never fake portability"**: Bent. Spec §7 lists `ToolChoice::Required` and `ToolChoice::Specific` as "Approximated — system-prompt nudge". Per CLAUDE.md, this should return `Error::Unsupported`, not approximate.
- **"Prefer typed escape hatches over leaky abstractions"**: Mostly honored via `ChatRequestOptions`. Good.
- **"Do not force adjacent capabilities into chat-shaped APIs"**: Sharpest tension. Claude Code is *not* a chat API — it's an agentic CLI. Plan disables agentic parts to *make* it chat-shaped. Inverse of the rule but arguably worse: forcing a non-chat system into the chat surface by amputating its non-chat capabilities.

### CEO Verdict: **RETHINK PREMISE**

Plan is technically competent and the spike-first approach is responsible. But load-bearing premises (demand real, CLI surface stable, "personal/testing" framing survives publication, Anthropic won't ship an official path within useful lifetime) are unvalidated and at least two are likely wrong. 4000-line scope sized for a strategic provider; artifact is a tactical wrapper. Mismatch will eat maintainer time for years.

Right move: ship 600-line example or `publish = false` crate that proves the integration works, gather six months of evidence on demand and CLI stability, then decide whether to invest in the full provider.

### Top 3 CEO concerns

1. **[CRITICAL]** Premature commitment to brittle wrapper of unstable interface. **Fix:** ship as `publish = false` workspace crate or example until spike has been re-run against three consecutive Claude Code CLI releases without breaking changes.
2. **[HIGH]** `ToolChoice` approximation violates "never fake portability" (CLAUDE.md / spec §7). **Fix:** make `ToolChoice::Required` / `::Specific` return `Error::Unsupported`.
3. **[HIGH]** Scope is 2-3x what evidence justifies; MCP server and Sandbox trait are speculative. **Fix:** cut v1 to chat + streaming only, defer Phase 6 and Task 5 entirely.

---

## Phase 3 — Engineering Review

### Architecture findings

1. **`error.rs` is orphaned.** `classify_subprocess_failure` is defined and tested but never called from `execute_chat`. Stderr `Arc<Mutex<String>>` captured then dropped. **All of spec §10's auth/rate-limit detection is dead code.**

2. **`chat.rs` is a god module.** Contains `ProviderConfig`, `CallGuard` + Drop, `execute_chat` (spawn + stdin pump + stderr drain + stdout stream), `run_chat`, `run_chat_stream`, `idle_timeout_stream`. CLAUDE.md "Conventions" calls out structuring files with primary type / impls / supporting / helpers / tests. Splitting into `chat.rs` (entry points) + `orchestration.rs` (execute_chat + CallGuard) would isolate cleanup-correctness invariants from streaming-vs-non-streaming dispatch.

3. **`Sandbox` trait with one impl is suspect, but justified.** Trait carries `SpawnSpec` rather than a built `Command` so wrappers can prepend args. But `Result<Command>` return is suspicious — `NoSandbox` cannot fail.

4. **`dispatcher.rs` leaks `mcp::ToolHandler` into its public API.** Layering is upside-down: higher-level user-facing `ToolDispatcher` defines its handler shape via wire-level MCP module's internal type. `ToolHandler` typedef belongs in `dispatcher.rs`; `mcp.rs` should consume it.

5. **`wire.rs` types are `pub(crate)` but used in tests as if `pub`.** Task 19 conformance tests do `let events: Vec<OutputEvent> = ...` from outside the crate. Test will not compile.

6. **No abstraction for "subprocess I/O lifecycle".** Stdin-pump task fire-and-forget; stderr-drain task fire-and-forget; both spawned without retained `JoinHandle`. `CallGuard::Drop` doesn't know about them. If they outlive the call, they leak.

### Plan-correctness bugs (won't compile / won't test what they claim)

| # | Bug | Where |
|---|---|---|
| 1 | `ResponseFormat::JsonObject` doesn't exist (variants: `Text`/`Json`/`JsonSchema`) | Task 16 step 1 |
| 2 | `ToolChoice::None` doesn't exist (variants: `Auto`/`Disabled`/`Required`/`Specific`) | spec §7 + plan |
| 3 | `Error::Configuration` doesn't exist | spec §10 |
| 4 | `Error::RateLimit` (singular) doesn't exist; real name is `RateLimited { ... }` | spec §10 |
| 5 | `Usage` is `#[non_exhaustive]` — struct-literal construction forbidden outside crate | Task 8 |
| 6 | `ChatResponse` is `#[non_exhaustive]` — same | Task 8 |
| 7 | `Message::Assistant { ... }` / `User { ... }` struct-literal — `Message` is `#[non_exhaustive]` | Task 9 |
| 8 | `ContentBlock::*` struct literals — `#[non_exhaustive]` | Tasks 8, 9 |
| 9 | `ContentPart::Image { ... }` struct literal — `#[non_exhaustive]` | Task 9 |
| 10 | `ImageSource::Base64 { ... }` struct literal — `#[non_exhaustive]` | Task 9 |
| 11 | `ReasoningEffort::Minimal` doesn't exist (variants: `Low`/`Medium`/`High`) | Task 10 `map_effort` |
| 12 | `request.tool_choice` is `Option<ToolChoice>` — plan treats as `ToolChoice` | Task 14 |
| 13 | `request.tools` is `Option<Vec<Tool>>` — plan treats as `Vec<Tool>` | Task 14 |
| 14 | `wire::OutputEvent` is `pub(crate)` but Task 19 imports from outside crate | Task 19 |
| 15 | `mod mcp;` is private but Task 18 tests use it directly without `__testing` shim | Task 18 |
| 16 | `Tool::description` is `Option<String>` → may serialize to `null` in MCP `tools/list`; Claude's behavior unverified | Task 11 |
| 17 | `McpServer::stop` defined but never called from production `CallGuard::Drop` — only test code uses it | Tasks 11, 14 |
| 18 | `request_timeout` formatted as `Debug` produces inconsistent strings (`300s` vs `5m`) | Task 16 |
| 19 | `futures-executor` workspace dep added in Task 19 step 4 but root workspace inheritance unverified | Task 19 |
| 20 | Plain `(StatusCode::UNAUTHORIZED, "missing or bad bearer token")` is plaintext, not JSON-RPC. Claude's behavior unverified | Task 11 |
| 21 | `idle_timeout_stream` drops in-flight `next()` on timeout, re-polls from scratch — loses `BufReader` state | Task 16 |
| 22 | NDJSON `Vec<u8>` buffer grows unbounded — multi-MB tool_result image base64 → OOM | Task 7 |
| 23 | Stderr ring buffer slices `String` by byte (`g[(len - 4096)..]`) — panics on multibyte UTF-8 boundary | Task 14 |
| 24 | `CallGuard::Drop` spawns `current_thread` runtime *without* I/O driver, calls `child.wait().await` — panics with "no reactor running" | Task 14 |
| 25 | `CallGuard::Drop` cleanup `std::thread::spawn` is fire-and-forget — if Drop runs during runtime shutdown, cleanup thread killed mid-work, leaks subprocess | Task 14 |
| 26 | `Provider::new(token)` resolves `claude` from PATH at runtime, not build-time; inconsistent with `from_env()` and `builder().build()` which validate at build | Task 16 |
| 27 | TDD claim is mostly fictional — most tasks bundle test+impl in one block; Task 14 (orchestration) has zero unit tests at all | Tasks 4, 5, 7, 14, 16 |

### Edge cases & failure modes (severity-tagged)

- **[CRITICAL]** `CallGuard::Drop` cleanup is unsound — will panic at runtime, leak subprocess. (See bug #24/25.)
- **[CRITICAL]** `McpServer::Drop` cannot graceful-shutdown without an awaiter — bound socket may linger past stream drop until OS cleans up.
- **[HIGH]** Stdin pipe backpressure → deadlock. Stdin pump blasts all input then closes; `parse_ndjson` only starts pulling when caller polls. Large input + slow producer + caller not polling yet → deadlock. Fix: drain stdout eagerly.
- **[HIGH]** Stderr drain task `String` slice can panic on non-UTF-8 char boundary.
- **[HIGH]** `idle_timeout_stream` leaks the inner future on timeout — corrupts framing.
- **[HIGH]** MCP server bind succeeds before any error can surface; `axum::serve` failure happens later in spawned task.
- **[HIGH]** MCP plaintext-401 on bad bearer token may cause Claude to retry indefinitely → 100% CPU until per-call timeout.
- **[HIGH]** NDJSON parser unbounded memory on large lines.
- **[MEDIUM]** `ScratchDir::Drop` blocking `remove_dir_all` from inside async runtime — stalls worker thread on large dirs.
- **[MEDIUM]** Spike (Task 1) "if it fails, revisit" insufficient — Tasks 4, 6, 7, 8, 9, 10, 14 hardcode JSON shapes / env-var names that are unverified. Should explicitly gate Phase 3+ on spike outputs.

### Test coverage gaps (CRITICAL gaps shown)

- `CallGuard::Drop` real cleanup on cancellation mid-stream — **not covered**
- Future drop mid-stream releases port + reaps PID — **not covered**
- Stdin pipe deadlock when claude doesn't read — **not covered**
- Stderr pipe full / multibyte UTF-8 boundary — **not covered**
- MCP server bind failure (port exhaustion) — **not covered**
- Concurrent `chat()` calls (independent ports/scratch) — **not covered**
- Subprocess crash before any stream-json event — **not covered** (`fake_claude` always succeeds)
- Subprocess prints partial JSON then exits — **not covered**
- Per-call `request_timeout` triggering — **not covered**
- Stream `idle_timeout` triggering — **not covered**
- OAuth token expiring mid-call — **not covered**
- User tool callback panicking — **not covered**
- Large input (multi-MB image base64) memory — **not covered**
- `ResponseFormat` rejection test — **broken** (uses `JsonObject` which doesn't exist)

### Performance & resource concerns

- Per-call subprocess fork+exec ~50-200 ms (claude is Node-based) + bind/axum boot ~10-25 ms = 100-300 ms cold-start floor before first byte.
- `rand::thread_rng()` in `rand 0.8` is CSPRNG (ChaCha-based, OsRng-seeded) — token is fine. Migration to `rand::rng()` in `rand 0.9+` worth tracking.
- Bearer token in `ps -ef` — acknowledged in spec §6, deferred. Acceptable for v1.
- Mid-stream future drop releases port eventually but sync-only `McpServer::Drop` doesn't await `JoinHandle` — bound socket may linger.
- Memory: long generations / huge tool_results held in memory in non-streaming path — same as every provider.
- Zombie subprocesses under stream-cancellation are the real perf risk (see CallGuard bugs).

### Security concerns

- **[HIGH]** Plaintext-401 on bad bearer token → potential Claude retry storm.
- **[MEDIUM]** Missing env-var lockdown for `NODE_OPTIONS` (env_clear strips it but should be explicitly documented as deliberate). Other unaudited candidates: `SSL_CERT_FILE`, `SSL_CERT_DIR`, `CLAUDE_CODE_DEV_API_BASE_URL`-style internal redirects.
- **[MEDIUM]** User tool callback panic abort axum worker → chat call hangs until `stream_idle_timeout`. No `catch_unwind` / panic-handler middleware.
- **[LOW]** `--mcp-config` exposes bearer token in `ps` (acknowledged).

### TDD discipline audit

- Task 4, 5, 7: "Step 1: Write failing tests" but the code block includes BOTH impl AND tests. Failing test is fictional — test added simultaneously with impl. **Not real TDD.**
- Task 14: Step 4 explicitly says "verify it compiles (no integration test yet)". Most complex piece (cleanup, drains, signals, timeouts) ships with **zero unit tests**. Mock harness in Task 15 doesn't exercise any failure modes.
- Task 16: Closer to TDD, but the test uses `ResponseFormat::JsonObject` which doesn't exist.
- **Overall TDD claim is not honestly delivered.**

### Eng Verdict: **SHIP WITH FIXES (substantial)**

Architecture is largely sound; per-call subprocess + per-call MCP server is defensible. But plan's Rust code is studded with compilation errors against actual `anyllm` API (≥27 distinct bugs); Drop-based cleanup is unsound and will leak processes; test plan covers only happy path while claiming TDD. Required fixes (10) before execution:

1. Replace every `#[non_exhaustive]` struct-literal construction with public constructors / builders.
2. Fix `ToolChoice` / `tools` Option handling in `chat::execute_chat`.
3. Replace `ResponseFormat::JsonObject` / `ToolChoice::None` references; remove `ReasoningEffort::Minimal` arm; spec §10 must say `InvalidRequest` and `RateLimited`.
4. Rewrite `CallGuard::Drop` to not spawn a new tokio runtime: use `kill_on_drop(true)` at spawn + SIGTERM via `nix` synchronously.
5. Make `McpServer::Drop` wait for bound port release, or use `JoinHandle::abort()` and document.
6. Wire `error::classify_subprocess_failure` to actual exit-code / stderr path inside `execute_chat`.
7. Add unit tests for: stdin pipe full / claude crash before result event / cancellation mid-stream / future drop reaps PID / idle_timeout fires.
8. Make `wire::OutputEvent` and `mcp` pub-crate-and-test-visible via `pub mod __testing` re-export, applied to the conformance test paths.
9. Cap NDJSON line size in parser (e.g. 10 MiB hard limit) and surface `Error::UnexpectedResponse` past it.
10. Fix `String` byte-slice in stderr ring buffer to slice on char boundary or store as `VecDeque<u8>`.

### Top 5 Eng concerns

1. **[CRITICAL]** `CallGuard::Drop` cleanup unsound (Task 14). Spawns `current_thread` runtime *without* I/O driver, calls `child.wait().await` on it — panics at runtime, leaking subprocess. **Fix:** spawn subprocess with `kill_on_drop(true)` and SIGTERM via `nix::sys::signal::kill` synchronously in Drop; let kernel reap.
2. **[CRITICAL]** Plan's Rust code does not compile against `anyllm`'s `#[non_exhaustive]` types (Tasks 6, 8, 9, 10, 14, 16). **Fix:** rewrite every construction site to use public `::new` / builder methods.
3. **[HIGH]** Error classification path is dead code (Tasks 4, 14). Stderr captured but never read on subprocess exit; `classify_subprocess_failure` never called from production. Spec §10 unimplemented. **Fix:** in `execute_chat`, capture `child.wait()` after stdout stream ends; on non-zero exit, call `classify_subprocess_failure(exit_code, &stderr_capture.lock().await)` and surface as terminal error.
4. **[HIGH]** TDD claim mostly fictional (Tasks 4, 5, 7, 14, 16). Most failure-prone code (subprocess + pipes + cancellation) ships with zero unit tests. **Fix:** add Task 14b "drain edge-case unit tests" with cases for (claude exits 0 with no events / exits 1 with stderr / hangs / writes invalid JSON / future dropped mid-stream); make `fake_claude` configurable.
5. **[HIGH]** Spec/plan API drift across 6+ invented identifiers (`ResponseFormat::JsonObject`, `ToolChoice::None`, `ReasoningEffort::Minimal`, `Error::Configuration`, `Error::RateLimit`, `tool_choice` vs `Option<ToolChoice>`). **Fix:** sweep plan against actual `anyllm` sources before any task is dispatched.

---

## Phase 3.5 — Developer Experience Review

### Developer journey map

| Stage | Works | Hurts |
|---|---|---|
| 1. Discovery (parent README) | Honest framing | ToS note buried; reader doesn't yet know whether to click |
| 2. Install | Standard `cargo add` | **Hidden prereq:** `claude` CLI install never mentioned. Rust dev with no Node toolchain hits a wall. |
| 3. First setup (`claude setup-token`) | Documented in Quick Start | Two-step ceremony vs sibling's "paste API key"; interactive browser flow not scriptable for CI |
| 4. First example | Compiles | Different invocation pattern from sibling examples (`claude_code_chat.rs` doesn't follow `load_provider_for_example`) |
| 5. First success | Output prints | Latency high; if `ANTHROPIC_API_KEY` also set, precedence undocumented |
| 6. Adding tools | Pattern is clear | Dispatcher split is novel; mismatch fails silently — handler-less tool returns "JSON-RPC InvalidRequest" *to Claude*, not user. Mismatch with portable trait. |
| 7. First error | Variants are correct | Auth detection is string-match heuristic; when it misses, user gets `Error::Provider { message: "claude exited Some(2)" }` with no fix hint |
| 8. Going to production | README "What this is not" is honest | If they read it. Decision-grade caveat appears below Quick Start, not above. |
| 9. Upgrade | Crate keeps working as long as schema unchanged | "matches CLI version X" only in code comment, not README. Schema drift surfaces as `Error::UnexpectedResponse` with truncated line, no version-mismatch detection. |

### Time-to-Hello-World

- `anyllm-claude-code` TTHW: **~8 steps, 5–15 minutes** (dominated by `claude` CLI install if not present + browser-based OAuth)
- `anyllm-anthropic` TTHW: **3 steps, 1–2 minutes**
- **Delta: ~3× the steps + a hard prerequisite the README skips.**

### Error message audit

| Site | Grade | Issue |
|---|---|---|
| `claude` not on PATH | **B+** | Missing install pointer for Rust-only devs |
| Empty token via `from_env` | **B** | No "run `claude setup-token`" breadcrumb |
| Empty token via builder | **C** | Wrong variant (`InvalidRequest` vs `Auth`); inconsistent with `from_env` |
| MCP bind fail | **D** | Unactionable; doesn't suggest custom Sandbox or port-collision diagnosis |
| Scratch dir setup | **C** | Doesn't include the path |
| Subprocess unclassified non-zero exit | **C-** | Exit code + 4KB stderr tail with no fix hint; "`Some(2)`" leaks `Option` to humans |
| Auth heuristic miss | **F** | Falls through to generic `Provider`; no `verbose_errors` flag or stderr lead display |
| Total timeout | **C** | Doesn't suggest `request_timeout` knob; `format!("{:?}", duration)` inconsistent (`300s` vs `5m`) |
| Idle timeout | **C+** | Better, still no `with_idle_timeout` mention |
| stream-json parse fail | **B** | Useful; should explicitly say "your `claude` is newer than the crate's tested version" |
| `ResponseFormat::Json*` | **A-** | Gold standard: problem + cause + fix. Use as model for the rest. |
| Tool with no handler | **F** | Error never reaches user; surfaces as Claude saying "I tried but it failed". Must surface as `Error::InvalidRequest("tool '<name>' declared but no handler registered")` *before* spawning. |

### API ergonomics findings

- **[HIGH]** `with_tools` / `tools` (builder) / `ChatRequest::tools` — three near-identical method names on adjacent types. Will conflate.
- **[HIGH]** No portable tool surface — `ToolDispatcher` is provider-specific. Plan does not propose lifting to core. Either lift or document divergence explicitly.
- **[HIGH]** Naming inconsistency with sibling: `api_key`/`ANTHROPIC_API_KEY` vs `oauth_token`/`CLAUDE_CODE_OAUTH_TOKEN`; `claude_path` vs `CLAUDE_CODE_BIN`. Pick one verb shape.
- **[MEDIUM]** `Sandbox`, `SandboxPaths`, `SpawnSpec` all `pub` for v1 with only `NoSandbox` — pollutes rustdoc with no payoff. Make `pub(crate)` until second impl.
- **[MEDIUM]** No per-request timeout override. `request_timeout` is on Provider builder only.
- **[MEDIUM]** No tests for `with_chat_capabilities` / `with_tools` ordering — fragile against future Inner-field additions.
- **[MEDIUM]** Custom env vars out of scope. `ANTHROPIC_BETAS`, `HTTP_PROXY`, custom CA bundles all unreachable. `builder.extra_env(k, v)` is a 5-line change with massive DX upside.
- **[LOW]** `ChatRequestOptions::allowed_tools` re-enables built-ins by name — footgun without sandbox. README must warn.
- **[LOW]** `Provider::new(token)` resolves `claude` lazily; `from_env` and `builder().build()` validate at build. Inconsistent.

### Mock & test story

Plan ships `mock` as a feature flag, but it gates a `pub mod mock` whose only public function is `fake_claude_path()` — which **panics** if `cargo build --bin fake_claude` hasn't been run separately first. Problems:

1. **Requires manual `cargo build` step before `cargo test`.** `cargo test` should "just work".
2. **No way to inject fake response inline.** Want to test "what does my code do when Claude returns 3 tool calls"? Write a `.jsonl` file on disk, set `FAKE_CLAUDE_SCRIPT=path` env. Compare to `anyllm`'s built-in `MockProvider::build(|b| b.text("..."))` — Rust-native, in-memory, zero filesystem.
3. **Doesn't compose with normal test setup.** Right answer: behind `mock` feature, expose Rust-native `MockClaudeCodeProvider` that bypasses subprocess entirely; OR document `anyllm::MockProvider` as canonical and demote `fake_claude_path` to `#[doc(hidden)]`.

**Verdict: test story for downstream users is not defensible.** Sibling provider users get `anyllm::MockProvider` for free.

### DX scorecard

| Dimension | Score | Notes |
|---|---|---|
| Getting started < 5 min | **4/10** | `claude setup-token` + CLI install pushes past 5 min; README skips install |
| API/CLI naming guessable | **5/10** | `with_tools` / `tools` / `register` / `oauth_token` vs `api_key` / `claude_path` vs `CLAUDE_CODE_BIN` |
| Error messages actionable | **5/10** | One A-grade error; rest C–F. Subprocess + tool-handler-missing worst |
| Docs findable & complete | **6/10** | Honest, structured. Hidden prereq. No troubleshooting. Spec linked but not on docs.rs |
| Upgrade path safe | **4/10** | Version pin in code comment only. No version-mismatch detection |
| Dev environment friction-free | **3/10** | Requires CLI install + browser OAuth + NPM/Homebrew. Hostile to fresh CI runners |
| Mock/test story | **3/10** | Requires manual binary build; no in-process Rust mock |
| Escape hatches present | **5/10** | Sandbox half-baked v1; no custom env vars; no per-request timeout |

**Composite: ~4.4/10.** Acceptable v0.1.x exploratory ship; not yet sibling-quality.

### DX Verdict: **SHIP WITH FIXES (3 blockers)**

Architectural plan is sound; maintainer is honest about scope. But DX layer is underbaked relative to engineering rigor. Three changes block clean ship:

1. **README must mention CLI installation explicitly** (1 paragraph; blocks every fresh user).
2. **Tool-handler-missing must error in `chat()` before spawning** (10-line validation; prevents worst silent-failure mode).
3. **Mock story must be either Rust-native or explicitly delegated to `anyllm::MockProvider`** (1 module rewrite or 1 doc paragraph).

Naming inconsistencies, public `Sandbox`/`SpawnSpec` for single-impl v1, missing `extra_env` escape hatch should land in v0.2 — none individually fatal but together make crate feel less polished than sibling.

---

## Cross-Phase Decision Categories

### Single-voice "User Challenges" — recommend changing the user's stated direction

These would be User Challenges in dual-voice mode. Single-voice means one reviewer pushed back; user decides whether they're right.

**Flag 1 (CEO): Don't publish to crates.io.** Ship as `publish = false` workspace crate or example until 3 consecutive Claude Code CLI releases pass spike without breaking changes.

**Flag 2 (CEO + Eng): Drop tools / MCP server from v1.** Cut Phase 6, Tasks 11–12, 17, 18; keep chat + streaming + reasoning + image-input.

**Flag 3 (CEO + Eng): Make `ToolChoice::Required` / `::Specific` return `Error::Unsupported` instead of approximating.** Direct CLAUDE.md non-compliance ("never fake portability").

### Taste decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | `claude_path` builder vs `bin_path` builder | `bin_path` + `CLAUDE_CODE_BIN` (consistent verb pair) |
| 2 | `oauth_token` vs `api_key` builder | Keep `oauth_token` (semantically distinct); document parallel |
| 3 | `Sandbox`/`SandboxPaths`/`SpawnSpec` `pub` vs `pub(crate)` | `pub(crate)` for v1; lift when first concrete impl lands |
| 4 | Mock harness — `fake_claude` binary vs Rust-native | Delete `mock` feature; document `anyllm::MockProvider`; keep `fake_claude` `#[doc(hidden)]` for crate's own integration tests |

---

## Decisions Outstanding

The user was presented five paths at the final approval gate; they chose to save research and findings before deciding. Recap:

- **A.** Heavy revision before any implementation. Address Flags 1–3, 4 taste decisions, AND the Eng plan-correctness bugs. Re-write spec + plan, re-run /autoplan if desired.
- **B.** Targeted fixes only — proceed with smaller v1. Cut tools/MCP/Sandbox from v1, fix Eng plan-correctness bugs, fix `CallGuard::Drop` soundness, fix README CLI-install prereq. Defer publishing decision and `ToolChoice` debate. Ship chat-only v0.1, observe for 6 months, revisit.
- **C.** Engineering fixes only, ignore strategic critique. Fix the 27+ plan bugs and `CallGuard::Drop`. Ship the full 22-task plan. Plan still needs substantial revision before subagents can run.
- **D.** Ignore everything; proceed with current plan. Dispatch implementation subagents now. Each will hit compilation errors and patch around them. High burn rate.
- **E.** Stop here, do nothing further. /autoplan was an exercise; review value captured offline.

---

## Files

- This review: `docs/superpowers/reviews/2026-05-03-claude-code-provider-autoplan-review.md`
- Spec under review: `docs/superpowers/specs/2026-05-03-claude-code-provider-design.md`
- Plan under review: `docs/superpowers/plans/2026-05-03-claude-code-provider.md`
- Restore point: `~/.gstack/projects/sagikazarmark-anyllm/main-autoplan-restore-20260503-095709.md`
