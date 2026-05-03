# Design — `anyllm-claude-code` provider

**Status:** Draft, ready for implementation planning
**Date:** 2026-05-03
**Scope:** New workspace crate that exposes the Claude Code CLI as an `anyllm::ChatProvider`.

## 1. Motivation

`anyllm-anthropic` already covers the Anthropic Messages API (regular API
billing). Users with a **Claude Code subscription** have no portable way to
drive that subscription from `anyllm` today. This crate fills that gap by
running the `claude` CLI under the hood and surfacing it as a regular
`ChatProvider`.

The provider is explicitly framed for personal and testing use, not
production:

- Per-call subprocess startup adds latency a normal HTTP provider does not pay.
- The Claude Code subscription's ToS contemplates interactive CLI use; wrapping
  it as a headless provider is technically feasible but bends that contract.
  This is called out in the README.

The provider implements `ChatProvider` only. It does not implement
`EmbeddingProvider`.

## 2. Goals and non-goals

### Goals

- Provide a `ChatProvider` that maps cleanly onto `anyllm`'s portable surface
  for the cases that map at all.
- Preserve full multi-turn fidelity, including assistant tool_use blocks,
  tool_result blocks (text), reasoning blocks, and image input.
- Bridge `ChatRequest.tools` into Claude via an in-process HTTP MCP server.
- Keep configuration and filesystem state isolated from the user's normal
  `claude` environment to make behavior reproducible across users.
- Establish a pluggable `Sandbox` trait so future hard-isolation impls
  (bwrap, firejail, landlock, sandbox-exec) can drop in without touching
  the Provider.

### Non-goals (v1)

- Hard process-level isolation. v1 ships only a `NoSandbox` impl plus
  Provider-level soft scoping. `BwrapSandbox` and friends are deferred.
- Network egress enforcement. The combination of "all built-in tools
  disabled" plus `--strict-mcp-config` plus FS scoping is judged "good
  enough in practice"; a real netfilter is future work.
- Embeddings. Claude Code has no embedding endpoint.
- Image generation in assistant output. Code-mode Claude does not produce it.
- Native JSON-Schema-constrained output. Routed through `ExtractingProvider`
  if the `extract` feature is on; otherwise `Error::Unsupported`.
- OAuth refresh-token plumbing. v1 reads a static `CLAUDE_CODE_OAUTH_TOKEN`;
  expiry surfaces as `Error::Auth` and the user re-runs `claude setup-token`.
- macOS keychain bridge for auth. Env var only.
- Persistent-session execution model.

## 3. Public API surface

The crate mirrors `anyllm-anthropic`'s shape. Core types in `lib.rs`:

```rust
pub struct Provider { /* Arc<Inner> */ }

impl Provider {
    pub fn new(oauth_token: impl Into<String>) -> Result<Self>;
    pub fn from_env() -> Result<Self>;             // CLAUDE_CODE_OAUTH_TOKEN
    pub fn builder() -> ProviderBuilder;
    pub fn with_chat_capabilities(self, resolver: impl ChatCapabilityResolver) -> Self;
}

pub struct ProviderBuilder {
    pub fn oauth_token(self, t: impl Into<String>) -> Self;
    pub fn claude_path(self, p: impl Into<PathBuf>) -> Self;
    pub fn sandbox(self, s: impl Sandbox + 'static) -> Self;
    pub fn request_timeout(self, d: Duration) -> Self;       // default 5 min
    pub fn stream_idle_timeout(self, d: Duration) -> Self;   // default 60 s
    pub fn build(self) -> Result<Provider>;
}

impl ChatProvider for Provider { /* ... */ }

pub struct ChatRequestOptions { /* typed provider-specific request extras */ }

pub trait Sandbox: Send + Sync {
    fn wrap(&self, cmd: tokio::process::Command, paths: &SandboxPaths)
        -> Result<tokio::process::Command>;
}

pub struct NoSandbox;
pub struct SandboxPaths {
    pub scratch_dir: PathBuf,
    pub fake_home: PathBuf,
}
```

Identity:

- `provider_name()` returns `"claude-code"`.
- Capability answers come from `with_chat_capabilities` resolver if installed,
  then fall back to a built-in matrix (see §7).

## 4. Architecture

Per `chat()` call the Provider stands up four short-lived components and
tears them down on every exit path:

```
                    ┌─────────────────────────┐
ChatRequest ──────► │  Provider (Rust)        │
                    │   - render stream-json  │
                    │   - prepare scratch dir │
                    │   - mint MCP token      │
                    └────────┬────────────────┘
                             │
                  ┌──────────┴──────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐     ┌────────────────────┐
        │ MCP HTTP server  │     │  Sandbox.wrap()    │
        │ (in-process)     │     │  → tokio::Command  │
        │ exposes request  │     └─────────┬──────────┘
        │  tools as MCP    │               │
        └────────┬─────────┘               ▼
                 │                ┌────────────────────┐
                 │                │  claude subprocess │
                 │                │  --input-format    │
                 │                │  stream-json       │
                 │ JSON-RPC       │  --output-format   │
                 ◄────────────────┤  stream-json       │
                                  └─────────┬──────────┘
                                            │ stream-json events
                                            ▼
                                   ┌─────────────────┐
                                   │ Stream parser   │
                                   │ → StreamEvent   │
                                   └─────────┬───────┘
                                             ▼
                                    ChatResponse / ChatStream
```

Per-call resources:

- A scratch directory under `std::env::temp_dir()` (RW for the call).
- A fake `$HOME` inside the scratch dir (empty, RW for the call).
- A bound `127.0.0.1:0` ephemeral port hosting an HTTP MCP server.
- A 256-bit random bearer token authenticating MCP requests.
- A `claude` subprocess.

There is no shared mutable state between concurrent `chat()` calls. Two
concurrent calls on the same `Provider` get fully independent scratch dirs,
ports, tokens, MCP servers, and subprocesses.

## 5. Per-call execution flow

1. Resolve the `claude` binary path:
   - `ProviderBuilder::claude_path(...)` if set, else
   - `CLAUDE_CODE_BIN` env var if set, else
   - first `claude` on `PATH`, else
   - `Error::Configuration("claude binary not found; set CLAUDE_CODE_BIN
     or builder.claude_path(...)")`.
2. Create the per-call scratch dir and the empty fake `$HOME` inside it.
3. Mint a 256-bit bearer token via the system RNG.
4. Bind the MCP HTTP server on `127.0.0.1:0`. Register exactly the tools
   in `request.tools`. Authenticate every request against the bearer token.
5. Render the request:
   - `messages` → newline-delimited stream-json events on stdin (multi-turn,
     including assistant turns, `tool_use` blocks, `tool_result` blocks
     with text content, image parts).
   - `system` (one or many `SystemOptions`) → concatenated with `\n\n`
     and passed via `--system-prompt` (replace, not append).
   - `tools` → exposed via the MCP server. No inline tool definitions to
     `claude`. Tools appear to Claude as `mcp__anyllm__<name>`; the prefix
     is stripped when mapping back to anyllm `ToolCall`s.
   - `tool_choice` → see §7.
   - `reasoning` → `CLAUDE_CODE_EFFORT_LEVEL` env (`low`/`medium`/`high`/
     `xhigh`/`max`) and/or `CLAUDE_CODE_DISABLE_THINKING=1`.
6. Build a `tokio::process::Command` with the lockdown env (§8) plus the
   `CLAUDE_CODE_OAUTH_TOKEN`.
7. Hand the `Command` to `Sandbox::wrap(cmd, &SandboxPaths { scratch_dir,
   fake_home })`. `NoSandbox` returns it unchanged.
8. Spawn. Pipe stream-json events to stdin. Read newline-delimited
   stream-json events from stdout.
9. For each output event, emit a `StreamEvent` (or accumulate into
   `ChatResponse` for the non-streaming path).
10. On stream-json `result` event: finalize `ChatResponse` (usage, finish
    reason, model, id) and return it.
11. On *any* exit path (success, error, future drop):
    - SIGTERM the subprocess; wait up to 5 s.
    - SIGKILL if still alive; reap.
    - Shut down the MCP server.
    - Delete the scratch dir.

    Cleanup runs synchronously in the `Drop` impl of an internal guard so
    it survives task cancellation.

## 6. Subprocess invocation

Always-on CLI args:

```
claude
  -p
  --input-format  stream-json
  --output-format stream-json
  --model         <ChatRequest.model>
  --system-prompt <concatenated system messages>
  --strict-mcp-config
  --mcp-config    '{"mcpServers":{"anyllm":{"type":"http",
                     "url":"http://127.0.0.1:<port>/mcp",
                     "headers":{"Authorization":"Bearer <token>"}}}}'
  --disallowed-tools Bash,Read,Edit,Write,MultiEdit,Glob,Grep,
                     WebFetch,WebSearch,Task,TodoWrite,NotebookEdit,...
```

The `--disallowed-tools` list is the exhaustive set of Claude Code's
built-in tools at the targeted CLI version. We maintain it in code and
document the version of the CLI it was built against.

`--mcp-config` is passed as inline JSON, not a file path. The bearer
token is briefly visible in `ps -ef` output to the same user; this is
acceptable for a one-shot, localhost-bound, request-lifetime token.
Switching to a temp-file-in-scratch-dir is a trivial follow-up if the
`ps` exposure ever becomes a concern.

## 7. Capability mapping

Capability answers reported via `chat_capability(model, capability)`,
unless a `ChatCapabilityResolver` is installed.

| `anyllm` feature | Status | Implementation note |
|---|---|---|
| Chat (one-shot + streaming) | Supported | Native; stream-json on both stdin and stdout |
| `NativeStreaming` | Supported | Real incremental delivery from stream-json |
| Multi-turn including tool re-entry | Supported | Stream-json input carries assistant `tool_use` and prior `tool_result` |
| `ToolCalls` | Supported | Via per-call HTTP MCP server |
| `ParallelToolCalls` | Supported | Stream-json emits multiple `tool_use` blocks naturally |
| `ToolChoice::Auto` | Supported | Default; no flag needed |
| `ToolChoice::None` | Supported | Pass empty MCP tool list |
| `ToolChoice::Required` | Approximated | System-prompt nudge; capability reported as `Unknown` |
| `ToolChoice::Specific(name)` | Approximated | System-prompt nudge naming the tool; capability reported as `Unknown` |
| `ReasoningConfig` (request-side) | Supported | Maps to `CLAUDE_CODE_EFFORT_LEVEL` / `CLAUDE_CODE_DISABLE_THINKING` |
| `ReasoningOutput` | Supported | stream-json `thinking` events → `ContentBlock::Reasoning` |
| `ReasoningReplay` | Supported | Prior `Reasoning` blocks re-encoded into stream-json input |
| `ImageInput` | Supported | `ImageSource::Base64` and `::Url` re-encoded into stream-json input |
| `ImageDetail` | Unsupported | Hint silently dropped |
| `ImageOutput` | Unsupported | Code-mode Claude does not generate images |
| `ImageReplay` | Unsupported | Follows from `ImageOutput: Unsupported` |
| `StructuredOutput` (`ResponseFormat::Json*`) | Unsupported natively | If the `extract` feature is on, callers route through `ExtractingProvider`; the Provider itself returns `Error::Unsupported` when given a non-text `ResponseFormat` |
| `cache_control` on system prompts | Silently dropped | Claude Code manages its own prompt cache |
| Image content inside `ToolResult` blocks | Pending validation spike | Returns `Error::Unsupported` if stream-json input rejects them |
| Embeddings | Not implemented | Crate does not implement `EmbeddingProvider` |
| Usage reporting | Supported | From stream-json `result` event; populates `Usage` including cache fields |
| Finish reason | Best-effort | `success`→`Stop`, `error_max_turns`→`Length`, others → `Other(...)`; tool-call termination → `ToolCalls` |

## 8. Env vars set per call

Always-on additions to the spawned process's environment:

```
CLAUDE_CODE_OAUTH_TOKEN=<from builder/env>
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
CLAUDE_CODE_SKIP_PROMPT_HISTORY=1
CLAUDE_CODE_DISABLE_CLAUDE_MDS=1
CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1
CLAUDE_CODE_DISABLE_CRON=1
CLAUDE_CODE_AUTO_CONNECT_IDE=false
CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1
CLAUDE_CODE_DISABLE_POLICY_SKILLS=1
CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1
CLAUDE_CODE_SIMPLE=1
CLAUDE_CODE_TMPDIR=<scratch>/tmp
CLAUDE_CODE_PLUGIN_CACHE_DIR=<scratch>/plugins
CLAUDE_CODE_DEBUG_LOGS_DIR=<scratch>/debug
HOME=<scratch>/home
```

The Provider does not pass through the user's existing environment by
default; the spawned process gets exactly this set plus a minimal
baseline (`PATH`, `LANG`, `TZ`). A builder option to extend the env is
out of scope for v1.

## 9. Sandbox abstraction

The trait carries the per-call scratch dir and fake `$HOME` so impls like
`BwrapSandbox` can bind-mount them. The intent is sketched below; the
exact trait signature is refined during implementation planning. In
particular, the input may be a richer `SpawnSpec { program, args, env }`
struct rather than a `tokio::process::Command`, because impls that need
to inject a wrapper command (`bwrap`, `firejail`) must move the original
program into the wrapper's argument list.

```rust
// Sketch — exact signature finalized in the implementation plan.
pub trait Sandbox: Send + Sync {
    fn wrap(&self, spec: SpawnSpec) -> Result<tokio::process::Command>;
}

pub struct SpawnSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
    pub paths: SandboxPaths,
}

pub struct SandboxPaths {
    pub scratch_dir: PathBuf,
    pub fake_home: PathBuf,
}

pub struct NoSandbox;

impl Sandbox for NoSandbox {
    fn wrap(&self, spec: SpawnSpec) -> Result<tokio::process::Command> {
        let mut cmd = tokio::process::Command::new(spec.program);
        cmd.args(spec.args);
        cmd.env_clear();
        for (k, v) in spec.env { cmd.env(k, v); }
        Ok(cmd)
    }
}
```

The Provider always does soft scoping (sets `$HOME`, `CLAUDE_CODE_TMPDIR`,
`CLAUDE_CODE_PLUGIN_CACHE_DIR`, `CLAUDE_CODE_DEBUG_LOGS_DIR` to the scratch
dir, scrubs the inherited env). The `Sandbox` trait is a hook for *additional*
process-level isolation on top of soft scoping.

Future impls (deferred, noted in README and §13):

- `BwrapSandbox` (Linux): wraps `claude` with `bwrap --ro-bind /usr /usr
  --tmpfs /tmp --bind <scratch> <scratch> --bind <fake_home> $HOME
  --proc /proc --dev /dev --unshare-all --share-net --die-with-parent --`.
- `FirejailSandbox` (Linux).
- `LandlockSandbox` (Linux).
- `SandboxExecSandbox` (macOS).

## 10. Error mapping

| Condition | Variant |
|---|---|
| `claude` binary not found | `Error::Configuration` |
| Empty / missing `CLAUDE_CODE_OAUTH_TOKEN` at build time | `Error::Auth` (from `from_env`) or `Error::InvalidRequest` (from builder) |
| MCP server bind failure | `Error::Provider` |
| Subprocess spawn failure | `Error::Provider` |
| Per-call timeout | `Error::Timeout` (default 5 min, builder-configurable) |
| Stream idle timeout | `Error::Timeout` (default 60 s, builder-configurable) |
| stream-json parse failure | `Error::UnexpectedResponse` (offending line truncated and included) |
| Auth failure (string-matched on stderr / exit) | `Error::Auth` |
| Rate limit (string-matched on stream-json or stderr) | `Error::RateLimit` |
| Other non-zero exit before `result` event | `Error::Provider` (with stderr tail attached) |
| User tool closure returns `Err` | Surfaced to Claude as `tool_result { is_error: true }`; not an anyllm `Error` |

Stderr is captured into a bounded ring buffer (last 4 KiB), logged via
`tracing` at `debug` always, and attached to error context on non-zero
exit.

## 11. Lifecycle and concurrency

- Per-call timeout default 5 min (longer than the Anthropic crate's 2 min
  to allow for subprocess startup plus first-token latency); builder can
  override.
- Stream idle timeout default 60 s; if no stream-json event arrives for
  that long the call is killed and returns `Error::Timeout`.
- Cancellation: dropping the future runs the cleanup path described in
  §5 step 11. The internal guard's `Drop` impl owns the cleanup so it
  fires whether the future completes, errors, or is cancelled.
- Concurrent `chat()` calls on the same `Provider` are fully independent
  (own scratch dir, own port, own MCP server, own subprocess). No shared
  state to lock. Memory and CPU scale linearly with concurrency.
- A `tracing::span!("claude_code.chat", model, ...)` wraps each call so it
  composes cleanly with the existing `TracingChatProvider` wrapper.

## 12. Crate layout, features, platforms

New crate `crates/anyllm-claude-code` with a workspace dep entry in
`Cargo.toml`. Files:

- `lib.rs` — `Provider`, `ProviderBuilder`, env-var lookup, capability
  matrix, `Sandbox` trait, `NoSandbox`.
- `chat.rs` — `ChatProvider` impl: render request → spawn → drain.
- `streaming.rs` — stream-json parser, `StreamEvent` mapping.
- `wire.rs` — stream-json input/output type definitions
  (serde-driven, narrowly scoped to what we actually emit/consume).
- `mcp.rs` — in-process HTTP MCP server hosting per-call tool dispatch.
- `sandbox.rs` — `Sandbox` trait, `NoSandbox`, `SandboxPaths`.
- `options.rs` — `ChatRequestOptions` typed extras.
- `error.rs` — error mapping helpers.
- `conformance_tests.rs` — `anyllm-conformance` wired against the mock
  subprocess harness; included via `#[cfg(test)] mod conformance_tests;`
  in `lib.rs`, mirroring the Anthropic crate.

Feature flags:

| Feature | Default | What it gates |
|---|---|---|
| `extract` | on | Passthrough to `anyllm/extract` |
| `http-tracing` | off | `reqwest_middleware` + `reqwest_tracing` for the MCP HTTP server |
| `mock` | off | Mock subprocess + mock MCP harness for tests/examples |

Platform support:

| Platform | Tier | Sandbox available |
|---|---|---|
| Linux | 1 — full CI | `NoSandbox` (default). `BwrapSandbox` etc. are deferred future work. |
| macOS | 2 — CI compiles + smoke tests | `NoSandbox` only |
| Windows | 3 — best effort | `NoSandbox` only |

Conformance: `anyllm-conformance` runs against a mock-subprocess harness
in CI (no real `claude` binary required). Integration tests against a
real `claude` are opt-in (`#[ignore]` by default), gated on `claude`
being on `PATH` and `CLAUDE_CODE_OAUTH_TOKEN` being set.

Examples in `crates/anyllm-examples/examples/`:

- `claude_code_chat.rs`
- `claude_code_stream.rs`
- `claude_code_tools.rs`

## 13. Validation spike (gating)

Before building the full provider, a one-day spike confirms the
assumptions the design rests on:

1. `claude --input-format stream-json --output-format stream-json`
   accepts multi-turn input including assistant `tool_use` blocks and
   prior `tool_result` blocks.
2. `--mcp-config` accepts inline JSON in the targeted CLI version.
3. Image content survives a round-trip through stream-json input.
4. `CLAUDE_CODE_OAUTH_TOKEN` works as documented and overrides any
   keychain-stored credentials.
5. The full lockdown env-var set (§8) is honored: no inadvertent file
   writes outside the scratch dir, no telemetry traffic, no auto-update
   calls, no plugin/marketplace fetches.

If any check fails, the relevant section is revisited before
implementation continues.

## 14. Future work (explicitly deferred)

- Hard-isolation `Sandbox` impls: `BwrapSandbox`, `FirejailSandbox`,
  `LandlockSandbox` (Linux); `SandboxExecSandbox` (macOS).
- Network egress filtering (proxy or netfilter sandbox).
- OAuth refresh-token plumbing inside the Provider.
- macOS keychain bridge for auth.
- Persistent-session execution model if subprocess startup cost ever
  becomes a real complaint.
- Embedding endpoint (only if Claude Code adds one).
- Builder option to extend the inherited env beyond the minimal
  baseline.
- Switching `--mcp-config` to a temp file (eliminating the brief
  bearer-token exposure in `ps -ef`).
