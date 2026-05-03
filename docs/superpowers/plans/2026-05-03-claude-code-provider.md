# anyllm-claude-code Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `anyllm-claude-code` workspace crate that exposes the Claude Code CLI as an `anyllm::ChatProvider`, allowing callers to drive their Claude Code subscription through the portable `anyllm` interface.

**Architecture:** Per-call one-shot subprocess execution model. Each `chat()` call: binds an in-process HTTP MCP server on `127.0.0.1:0` exposing the request's tools, spawns `claude -p --input-format stream-json --output-format stream-json --mcp-config '<inline json>' --strict-mcp-config --disallowed-tools <all built-ins>`, pipes the rendered conversation in on stdin, drains stream-json events from stdout into anyllm `StreamEvent`s / `ChatResponse`. Cleanup (kill subprocess, shut down MCP server, delete scratch dir) runs in a `Drop` guard so it survives task cancellation. Sandboxing is pluggable via a `Sandbox` trait; v1 ships only `NoSandbox`.

**Tech Stack:** Rust 2024, tokio (process + net), axum 0.8 (MCP HTTP server), reqwest (only for the MCP feature flag wiring — no outbound HTTP from this crate's hot path), serde / serde_json, futures-util, the existing `anyllm` core types, `anyllm-conformance` for behavioral assertions.

**Spec:** `docs/superpowers/specs/2026-05-03-claude-code-provider-design.md`

---

## Phase 0 — Validation spike (gating)

The spec rests on assumptions about Claude CLI behavior that we have not personally verified. Run this spike first; if any check fails, fix the spec (and this plan) before continuing.

### Task 1: Validation spike

**Files:**
- Create: `docs/superpowers/notes/2026-05-03-claude-code-spike.md`
- Reference: `docs/superpowers/specs/2026-05-03-claude-code-provider-design.md` §13

- [ ] **Step 1: Verify `claude` CLI is on PATH and authenticated**

```bash
claude --version
claude setup-token --help     # confirm the OAuth-token setup flow exists
```

Expected: `claude --version` prints a version (record it in the notes file). If `claude` is missing, install it before continuing.

- [ ] **Step 2: Verify `--mcp-config` accepts inline JSON**

```bash
claude -p "say hi" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --output-format stream-json --max-turns 1
```

Expected: Claude responds normally; no parse error on the inline JSON. If it errors, the spec's §6 inline-JSON assumption fails — fall back to writing the config to a per-call temp file inside the scratch dir. Update spec §6 and Task 14 below accordingly.

- [ ] **Step 3: Verify `--input-format stream-json` accepts multi-turn**

Build a fixture file that contains a multi-turn conversation with: user text, assistant text reply, user follow-up, then have Claude respond. Pipe via stdin:

```bash
cat > /tmp/spike-input.jsonl <<'EOF'
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"What's 2+2?"}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"4"}]}}
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"And 3+3?"}]}}
EOF
claude -p --input-format stream-json --output-format stream-json --max-turns 1 < /tmp/spike-input.jsonl
```

Expected: Claude answers "6" (or equivalent), demonstrating it consumed the prior assistant turn rather than treating the whole thing as a single user message. **Capture the exact event shape Claude emits on stdout — the schema for `wire.rs` will mirror it.**

- [ ] **Step 4: Verify multi-turn with prior tool_use / tool_result blocks round-trips**

Construct a fixture where the first assistant turn contains a `tool_use` block and the next user turn contains the matching `tool_result`:

```bash
cat > /tmp/spike-tools-input.jsonl <<'EOF'
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"What's the weather?"}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{"location":"SF"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"Sunny, 72F"}]}}
EOF
claude -p --input-format stream-json --output-format stream-json --max-turns 1 < /tmp/spike-tools-input.jsonl
```

Expected: Claude treats the prior `tool_use` and `tool_result` as historical context and responds about the weather. If Claude errors or rejects the input shape, document the actual accepted shape and update the spec's "multi-turn including tool re-entry: Supported" claim.

- [ ] **Step 5: Verify `CLAUDE_CODE_OAUTH_TOKEN` overrides keychain**

```bash
TOKEN=$(claude setup-token | grep -oE 'eyJ[A-Za-z0-9._-]+')   # adjust extraction to actual output
unset ANTHROPIC_API_KEY
CLAUDE_CODE_OAUTH_TOKEN="$TOKEN" claude -p "say hi" --max-turns 1 --output-format stream-json
```

Expected: Authenticates successfully. Run again with `CLAUDE_CODE_OAUTH_TOKEN=invalid` and confirm the resulting error pattern (exit code, stderr contents) — record it for the §10 string-match auth detection.

- [ ] **Step 6: Verify the lockdown env-var set is honored**

Spawn `claude` with the spec §8 env-var set, an isolated `$HOME=/tmp/spike-home`, and `--strict-mcp-config --mcp-config '{"mcpServers":{}}'`. Use `inotifywait -mr` or `strace -fe trace=openat,connect` (Linux) to observe filesystem and network activity:

```bash
mkdir -p /tmp/spike-home /tmp/spike-tmp
strace -fe trace=openat,connect -o /tmp/spike-trace -- env -i \
    PATH=/usr/bin:/bin HOME=/tmp/spike-home \
    CLAUDE_CODE_OAUTH_TOKEN="$TOKEN" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    CLAUDE_CODE_SKIP_PROMPT_HISTORY=1 \
    CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 \
    CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 \
    CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 \
    CLAUDE_CODE_DISABLE_CRON=1 \
    CLAUDE_CODE_AUTO_CONNECT_IDE=false \
    CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL=1 \
    CLAUDE_CODE_DISABLE_POLICY_SKILLS=1 \
    CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1 \
    CLAUDE_CODE_SIMPLE=1 \
    CLAUDE_CODE_TMPDIR=/tmp/spike-tmp \
    claude -p "say hi" --max-turns 1 --output-format stream-json --strict-mcp-config --mcp-config '{"mcpServers":{}}'
grep -E 'openat.*\.claude|connect.*(?!api\.anthropic\.com)' /tmp/spike-trace | head -20
```

Expected: No file accesses outside `/tmp/spike-home`, `/tmp/spike-tmp`, `/usr`, `/etc`. No network connections except to `api.anthropic.com`. Document any unexpected accesses — they may need additional env-var lockdown or a noted limitation.

- [ ] **Step 7: Verify image content survives stream-json input**

Encode a 1×1 transparent PNG as base64, embed it in a user message, send it through:

```bash
cat > /tmp/spike-image.jsonl <<'EOF'
{"type":"user","message":{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII="}},{"type":"text","text":"What color is this pixel?"}]}}
EOF
claude -p --input-format stream-json --output-format stream-json --max-turns 1 < /tmp/spike-image.jsonl
```

Expected: Claude responds about the pixel color (or admits the image is too small). If Claude rejects the input shape, image input becomes `Unsupported`; update the capability matrix.

- [ ] **Step 8: Document findings**

Write `docs/superpowers/notes/2026-05-03-claude-code-spike.md` with:
- Claude CLI version tested
- For each step: passed/failed + the exact event/error shapes observed
- A "Schema Reference" section documenting the exact stream-json input and output event shapes Claude accepts/emits (this is the source-of-truth for `wire.rs` in Task 7)
- A "Spec Deltas" section listing any spec sections that need to be updated based on findings (and apply those updates)

- [ ] **Step 9: Commit**

```bash
git add docs/superpowers/notes/2026-05-03-claude-code-spike.md docs/superpowers/specs/
git commit -m "docs(spike): validate claude-code provider assumptions

Document the stream-json schema and lockdown behavior verified against
the live Claude CLI before implementation begins."
```

---

## Phase 1 — Crate scaffolding

### Task 2: Create the empty crate skeleton

**Files:**
- Create: `crates/anyllm-claude-code/Cargo.toml`
- Create: `crates/anyllm-claude-code/src/lib.rs`
- Modify: `Cargo.toml` (root) — add workspace dep entry

- [ ] **Step 1: Create `crates/anyllm-claude-code/Cargo.toml`**

```toml
[package]
name = "anyllm-claude-code"
description = "Claude Code CLI provider for anyllm — drive your Claude Code subscription through the anyllm portable interface"
version = { workspace = true }
edition = { workspace = true }
rust-version = { workspace = true }
license = { workspace = true }
repository = { workspace = true }
homepage = { workspace = true }
keywords = { workspace = true }
categories = { workspace = true }

[dependencies]
anyllm = { workspace = true }
axum = "0.8"
futures-core = { workspace = true }
futures-util = { workspace = true }
rand = "0.8"
reqwest-middleware = { workspace = true, optional = true }
reqwest-tracing = { workspace = true, optional = true }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true, features = ["fs", "io-util", "macros", "net", "process", "rt", "signal", "sync", "time"] }

[dev-dependencies]
anyllm-conformance = { workspace = true }
tempfile = "3"
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }

[features]
default = ["extract"]
extract = ["anyllm/extract"]
http-tracing = ["dep:reqwest-middleware", "dep:reqwest-tracing"]
mock = []

[package.metadata.docs.rs]
all-features = true
```

Note: `rand` is needed for the bearer token; `tempfile` (dev-dep) is for scratch-dir tests.

- [ ] **Step 2: Create `crates/anyllm-claude-code/src/lib.rs`**

```rust
#![warn(missing_docs)]
//! Claude Code CLI provider for `anyllm`.
//!
//! Wraps the `claude` CLI as a regular [`anyllm::ChatProvider`], allowing
//! callers to drive their Claude Code subscription through the portable
//! `anyllm` interface.
//!
//! See the [design spec](https://github.com/sagikazarmark/anyllm/blob/main/docs/superpowers/specs/2026-05-03-claude-code-provider-design.md)
//! for the architecture and capability matrix.

// Modules added in later tasks.
```

- [ ] **Step 3: Add workspace dep entry**

Edit `Cargo.toml` (root) — insert into the `[workspace.dependencies]` table, alphabetized:

```toml
anyllm-claude-code = { version = "0.1.1", path = "crates/anyllm-claude-code" }
```

- [ ] **Step 4: Verify it builds**

```bash
cargo build -p anyllm-claude-code
```

Expected: Compiles with no warnings.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/anyllm-claude-code
git commit -m "feat(claude-code): scaffold anyllm-claude-code crate"
```

### Task 3: Wire the crate into the README providers table

**Files:**
- Modify: `README.md` (root) — both the workspace-crates table and the providers table

- [ ] **Step 1: Add row to "Workspace Crates" table**

In `README.md`, in the "Workspace Crates" table, insert after the `anyllm-cloudflare-worker` row:

```markdown
| [`anyllm-claude-code`](crates/anyllm-claude-code) | Provider adapter | Wraps the `claude` CLI for use with a Claude Code subscription; per-call subprocess + in-process MCP server for tools |
```

- [ ] **Step 2: Add row to "Providers" table**

In the "Providers" table, insert after the Cloudflare Workers AI row:

```markdown
| Claude Code (subscription) | [`anyllm-claude-code`](crates/anyllm-claude-code) | ✓ | n/a | Wraps the `claude` CLI; uses your Claude Code subscription via `CLAUDE_CODE_OAUTH_TOKEN`. Personal/testing use; see crate README for ToS notes |
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): list anyllm-claude-code in providers table"
```

---

## Phase 2 — Errors and sandbox

### Task 4: Error mapping helpers

**Files:**
- Create: `crates/anyllm-claude-code/src/error.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod error;`)

- [ ] **Step 1: Write the failing tests**

Append to `crates/anyllm-claude-code/src/error.rs`:

```rust
//! Error mapping helpers for the Claude Code provider.
//!
//! Translates subprocess failure modes (non-zero exit, stderr messages,
//! parse failures) into [`anyllm::Error`] variants per the design spec
//! §10. Detection of auth and rate-limit failures is best-effort string
//! matching against `claude` stderr.

use anyllm::Error;

/// Convert a subprocess non-zero exit into an [`Error`].
///
/// Inspects the trailing stderr (already truncated to a bounded ring
/// buffer) for known auth- and rate-limit signatures. Falls through to
/// [`Error::Provider`] otherwise.
pub(crate) fn classify_subprocess_failure(
    exit_code: Option<i32>,
    stderr_tail: &str,
) -> Error {
    let stderr_lower = stderr_tail.to_ascii_lowercase();
    if is_auth_failure(&stderr_lower) {
        return Error::Auth(format!(
            "claude reported authentication failure (exit {:?}): {}",
            exit_code,
            truncate(stderr_tail, 256)
        ));
    }
    if is_rate_limit(&stderr_lower) {
        return Error::RateLimited {
            message: truncate(stderr_tail, 256),
            retry_after: None,
            request_id: None,
        };
    }
    Error::Provider {
        status: None,
        message: format!("claude exited {:?}", exit_code),
        body: Some(truncate(stderr_tail, 4096)),
        request_id: None,
    }
}

fn is_auth_failure(stderr_lower: &str) -> bool {
    stderr_lower.contains("invalid api key")
        || stderr_lower.contains("invalid token")
        || stderr_lower.contains("not authenticated")
        || stderr_lower.contains("authentication failed")
        || stderr_lower.contains("oauth_token") && stderr_lower.contains("invalid")
        || stderr_lower.contains("401")
}

fn is_rate_limit(stderr_lower: &str) -> bool {
    stderr_lower.contains("rate limit")
        || stderr_lower.contains("usage limit")
        || stderr_lower.contains("quota") && stderr_lower.contains("exceed")
        || stderr_lower.contains("429")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out = s[..max].to_string();
        out.push_str("...[truncated]");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_auth_failure() {
        let err = classify_subprocess_failure(Some(2), "Error: invalid OAuth token");
        assert!(matches!(err, Error::Auth(_)));
    }

    #[test]
    fn classifies_rate_limit() {
        let err = classify_subprocess_failure(Some(2), "Error: usage limit exceeded for today");
        match err {
            Error::RateLimited { message, .. } => assert!(message.contains("usage limit")),
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn falls_through_to_provider() {
        let err = classify_subprocess_failure(Some(1), "some other failure mode");
        match err {
            Error::Provider { message, body, .. } => {
                assert!(message.contains("exited"));
                assert!(body.unwrap().contains("some other failure"));
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn truncate_caps_long_tails() {
        let long = "a".repeat(10_000);
        let err = classify_subprocess_failure(Some(1), &long);
        match err {
            Error::Provider { body: Some(body), .. } => {
                assert!(body.len() < 5000);
                assert!(body.ends_with("...[truncated]"));
            }
            other => panic!("expected Provider with body, got {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Add `mod error;` to `lib.rs`**

In `crates/anyllm-claude-code/src/lib.rs`, after the docs:

```rust
mod error;

pub(crate) use error::classify_subprocess_failure;
```

- [ ] **Step 3: Run tests, verify they pass**

```bash
cargo test -p anyllm-claude-code error::tests
```

Expected: All four tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/anyllm-claude-code/src
git commit -m "feat(claude-code): error classification helpers"
```

### Task 5: Sandbox trait and NoSandbox impl

**Files:**
- Create: `crates/anyllm-claude-code/src/sandbox.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `pub mod sandbox;` re-exports)

- [ ] **Step 1: Write failing tests**

Create `crates/anyllm-claude-code/src/sandbox.rs`:

```rust
//! Pluggable process-isolation hook for the Claude Code provider.
//!
//! v1 ships only [`NoSandbox`]; the trait exists so future impls
//! ([`BwrapSandbox`], `FirejailSandbox`, etc.) can drop in without
//! touching the [`crate::Provider`].

use std::ffi::OsString;
use std::path::PathBuf;

use anyllm::Result;

/// Per-call paths a [`Sandbox`] impl may need to bind-mount.
#[derive(Debug, Clone)]
pub struct SandboxPaths {
    /// Per-call RW scratch directory.
    pub scratch_dir: PathBuf,
    /// Per-call empty fake `$HOME` (lives inside `scratch_dir`).
    pub fake_home: PathBuf,
}

/// Fully-prepared spawn description handed to a [`Sandbox`] impl.
///
/// Carrying program/args/env separately (rather than a built
/// [`tokio::process::Command`]) lets impls like a future `BwrapSandbox`
/// move the original program into the wrapper's argument list.
#[derive(Debug, Clone)]
pub struct SpawnSpec {
    /// Path to the program to execute (e.g. resolved `claude`).
    pub program: PathBuf,
    /// Arguments for the program, in order.
    pub args: Vec<OsString>,
    /// Environment variables for the spawned process. Replaces (does not
    /// extend) the parent process's environment.
    pub env: Vec<(OsString, OsString)>,
    /// Per-call paths the sandbox may need to expose.
    pub paths: SandboxPaths,
}

/// Process-isolation hook.
///
/// Implementations construct the [`tokio::process::Command`] the
/// [`crate::Provider`] will spawn. The default [`NoSandbox`] just builds
/// a `Command` from the spec verbatim; future impls wrap it (e.g.
/// `bwrap --ro-bind ... -- <program> <args>`).
pub trait Sandbox: Send + Sync {
    /// Build the [`tokio::process::Command`] to spawn for this call.
    fn build_command(&self, spec: SpawnSpec) -> Result<tokio::process::Command>;
}

/// No-op sandbox: builds the command verbatim from the spec.
///
/// Filesystem isolation in this mode is purely the soft scoping the
/// [`crate::Provider`] already does (per-call `$HOME`, scoped temp/plugin/
/// debug dirs).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSandbox;

impl Sandbox for NoSandbox {
    fn build_command(&self, spec: SpawnSpec) -> Result<tokio::process::Command> {
        let mut cmd = tokio::process::Command::new(spec.program);
        cmd.args(spec.args);
        cmd.env_clear();
        for (k, v) in spec.env {
            cmd.env(k, v);
        }
        Ok(cmd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn paths() -> SandboxPaths {
        SandboxPaths {
            scratch_dir: PathBuf::from("/tmp/scratch"),
            fake_home: PathBuf::from("/tmp/scratch/home"),
        }
    }

    fn spec() -> SpawnSpec {
        SpawnSpec {
            program: PathBuf::from("/usr/bin/echo"),
            args: vec!["hi".into()],
            env: vec![("FOO".into(), "bar".into())],
            paths: paths(),
        }
    }

    #[test]
    fn no_sandbox_builds_command_verbatim() {
        let cmd = NoSandbox.build_command(spec()).unwrap();
        let std_cmd = cmd.as_std();
        assert_eq!(std_cmd.get_program(), "/usr/bin/echo");
        let args: Vec<&std::ffi::OsStr> = std_cmd.get_args().collect();
        assert_eq!(args, vec![std::ffi::OsStr::new("hi")]);
        let envs: Vec<(&std::ffi::OsStr, Option<&std::ffi::OsStr>)> =
            std_cmd.get_envs().collect();
        assert_eq!(
            envs,
            vec![(std::ffi::OsStr::new("FOO"), Some(std::ffi::OsStr::new("bar")))]
        );
    }

    #[test]
    fn no_sandbox_clears_inherited_env() {
        // env_clear is the contract: parent env is not leaked.
        let cmd = NoSandbox.build_command(spec()).unwrap();
        let std_cmd = cmd.as_std();
        // Only the FOO var we passed should be present.
        assert_eq!(std_cmd.get_envs().count(), 1);
    }
}
```

- [ ] **Step 2: Add module to `lib.rs`**

```rust
pub mod sandbox;

pub use sandbox::{NoSandbox, Sandbox, SandboxPaths, SpawnSpec};
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p anyllm-claude-code sandbox::tests
```

Expected: Both tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/anyllm-claude-code/src
git commit -m "feat(claude-code): pluggable Sandbox trait with NoSandbox impl"
```

---

## Phase 3 — Wire types

The exact field names below assume the spike confirmed the schema. If the spike found differences, edit these structs to match before continuing.

### Task 6: stream-json output event types

**Files:**
- Create: `crates/anyllm-claude-code/src/wire.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod wire;`)

- [ ] **Step 1: Write the failing test**

Create `crates/anyllm-claude-code/src/wire.rs`:

```rust
//! Serde types for the stream-json input and output formats consumed and
//! emitted by `claude --input-format stream-json --output-format stream-json`.
//!
//! The shapes mirror what the spike (Task 1) verified against the live
//! CLI. Where Claude's vocabulary diverges from anyllm's, the divergence
//! is normalized in the dedicated mapping modules ([`crate::streaming`]
//! for output, [`crate::chat`] for input rendering), not here.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------- Output events (stdout: claude → us) ----------

/// One event of the stream-json output format, as printed line-by-line
/// by `claude --output-format stream-json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OutputEvent {
    /// Mid-stream assistant message chunk (one or more content blocks).
    Assistant {
        message: AssistantMessage,
    },
    /// Tool result emitted by the user/system after Claude called a tool.
    User {
        message: UserMessage,
    },
    /// System notice (start of session, model switch, etc.).
    System {
        #[serde(flatten)]
        rest: Value,
    },
    /// Terminal event: usage, finish reason, model, ID.
    Result(ResultEvent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AssistantMessage {
    pub id: Option<String>,
    pub model: Option<String>,
    pub content: Vec<OutputContentBlock>,
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<UsageBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UserMessage {
    pub content: Vec<OutputContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OutputContentBlock {
    Text { text: String },
    Thinking { thinking: String, signature: Option<String> },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        is_error: Option<bool>,
        content: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UsageBlock {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResultEvent {
    pub subtype: String,
    pub session_id: Option<String>,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub num_turns: Option<u32>,
    #[serde(default)]
    pub usage: Option<UsageBlock>,
    #[serde(default)]
    pub result: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

// ---------- Input events (stdin: us → claude) ----------

/// One event of the stream-json input format, written line-by-line to
/// `claude --input-format stream-json` on stdin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum InputEvent {
    User { message: InputUserMessage },
    Assistant { message: InputAssistantMessage },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InputUserMessage {
    pub role: String, // "user"
    pub content: Vec<InputContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InputAssistantMessage {
    pub role: String, // "assistant"
    pub content: Vec<InputContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum InputContentBlock {
    Text { text: String },
    Thinking { thinking: String, #[serde(skip_serializing_if = "Option::is_none")] signature: Option<String> },
    Image { source: ImageSource },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        content: Value, // string or array of content blocks
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_assistant_text_event() {
        let line = r#"{"type":"assistant","message":{"id":"msg_1","model":"claude-sonnet-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":null,"usage":{"input_tokens":4,"output_tokens":1}}}"#;
        let evt: OutputEvent = serde_json::from_str(line).unwrap();
        match evt {
            OutputEvent::Assistant { message } => {
                assert_eq!(message.id.as_deref(), Some("msg_1"));
                assert_eq!(message.content.len(), 1);
                assert!(matches!(message.content[0], OutputContentBlock::Text { ref text } if text == "hi"));
                assert_eq!(message.usage.unwrap().input_tokens, Some(4));
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn parses_tool_use_event() {
        let line = r#"{"type":"assistant","message":{"id":null,"model":null,"content":[{"type":"tool_use","id":"toolu_1","name":"search","input":{"q":"rust"}}],"stop_reason":"tool_use"}}"#;
        let evt: OutputEvent = serde_json::from_str(line).unwrap();
        match evt {
            OutputEvent::Assistant { message } => {
                assert!(matches!(
                    message.content[0],
                    OutputContentBlock::ToolUse { ref id, ref name, .. }
                        if id == "toolu_1" && name == "search"
                ));
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn parses_result_event() {
        let line = r#"{"type":"result","subtype":"success","session_id":"s1","is_error":false,"duration_ms":900,"num_turns":1,"usage":{"input_tokens":10,"output_tokens":3},"result":"hi"}"#;
        let evt: OutputEvent = serde_json::from_str(line).unwrap();
        match evt {
            OutputEvent::Result(r) => {
                assert_eq!(r.subtype, "success");
                assert_eq!(r.is_error, false);
                assert_eq!(r.usage.unwrap().output_tokens, Some(3));
            }
            other => panic!("expected Result, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_input_user_message_with_text() {
        let evt = InputEvent::User {
            message: InputUserMessage {
                role: "user".into(),
                content: vec![InputContentBlock::Text { text: "hi".into() }],
            },
        };
        let json = serde_json::to_string(&evt).unwrap();
        let back: InputEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            back,
            InputEvent::User { message } if message.content.len() == 1
        ));
    }

    #[test]
    fn round_trips_input_with_image_base64() {
        let evt = InputEvent::User {
            message: InputUserMessage {
                role: "user".into(),
                content: vec![InputContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: "aGVsbG8=".into(),
                    },
                }],
            },
        };
        let json = serde_json::to_string(&evt).unwrap();
        assert!(json.contains(r#""media_type":"image/png""#));
        assert!(json.contains(r#""data":"aGVsbG8=""#));
        let _back: InputEvent = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn round_trips_input_with_tool_result() {
        let evt = InputEvent::User {
            message: InputUserMessage {
                role: "user".into(),
                content: vec![InputContentBlock::ToolResult {
                    tool_use_id: "toolu_1".into(),
                    is_error: Some(false),
                    content: Value::String("Sunny, 72F".into()),
                }],
            },
        };
        let json = serde_json::to_string(&evt).unwrap();
        let back: InputEvent = serde_json::from_str(&json).unwrap();
        match back {
            InputEvent::User { message } => match &message.content[0] {
                InputContentBlock::ToolResult { tool_use_id, is_error, content } => {
                    assert_eq!(tool_use_id, "toolu_1");
                    assert_eq!(*is_error, Some(false));
                    assert_eq!(content, &Value::String("Sunny, 72F".into()));
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Add `mod wire;` to `lib.rs`**

```rust
mod wire;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p anyllm-claude-code wire::tests
```

Expected: All five tests pass. If any fail because the spike documented different field names, update the structs to match the spike's findings *before* continuing.

- [ ] **Step 4: Commit**

```bash
git add crates/anyllm-claude-code/src
git commit -m "feat(claude-code): stream-json wire types for input and output"
```

---

## Phase 4 — Streaming parser and StreamEvent mapping

### Task 7: NDJSON parser

**Files:**
- Create: `crates/anyllm-claude-code/src/streaming.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod streaming;`)

- [ ] **Step 1: Write failing tests**

Create `crates/anyllm-claude-code/src/streaming.rs`:

```rust
//! Stream-json output parser and mapping into [`anyllm::StreamEvent`].
//!
//! The parser is line-delimited NDJSON: each newline-terminated chunk on
//! `claude` stdout is one [`crate::wire::OutputEvent`].

use anyllm::{Error, Result, StreamEvent};
use futures_core::Stream;
use futures_util::StreamExt;
use std::pin::Pin;

use crate::wire::OutputEvent;

/// Parse a byte stream of NDJSON output events into typed [`OutputEvent`]s.
///
/// Carries lossless framing: an event is emitted only after a complete
/// line (terminated by `\n`) has been received. The final partial line
/// (no trailing newline) is parsed at end-of-stream.
pub(crate) fn parse_ndjson<S>(byte_stream: S) -> impl Stream<Item = Result<OutputEvent>>
where
    S: Stream<Item = std::io::Result<bytes::Bytes>> + Send + 'static,
{
    use futures_util::stream;

    stream::unfold(
        (Box::pin(byte_stream), Vec::<u8>::new(), false),
        |(mut s, mut buf, mut done)| async move {
            loop {
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = &line[..line.len() - 1]; // strip \n
                    if line.is_empty() || line.iter().all(|&b| b == b'\r' || b == b' ' || b == b'\t') {
                        continue;
                    }
                    let parsed = serde_json::from_slice::<OutputEvent>(line).map_err(|e| {
                        Error::UnexpectedResponse(format!(
                            "failed to parse stream-json line: {} (line: {})",
                            e,
                            String::from_utf8_lossy(line).chars().take(200).collect::<String>()
                        ))
                    });
                    return Some((parsed, (s, buf, done)));
                }
                if done {
                    if !buf.is_empty() && !buf.iter().all(|&b| b == b'\r' || b == b' ' || b == b'\t') {
                        let parsed = serde_json::from_slice::<OutputEvent>(&buf).map_err(|e| {
                            Error::UnexpectedResponse(format!(
                                "failed to parse trailing stream-json: {}",
                                e
                            ))
                        });
                        buf.clear();
                        return Some((parsed, (s, buf, done)));
                    }
                    return None;
                }
                match s.as_mut().next().await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(e)) => return Some((Err(Error::Stream(e.to_string())), (s, buf, true))),
                    None => done = true,
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures_util::stream;

    fn bytes_stream(chunks: Vec<&'static str>) -> impl Stream<Item = std::io::Result<Bytes>> {
        stream::iter(chunks.into_iter().map(|s| Ok(Bytes::from(s))))
    }

    #[tokio::test]
    async fn parses_two_complete_lines() {
        let s = bytes_stream(vec![
            "{\"type\":\"assistant\",\"message\":{\"id\":null,\"model\":null,\"content\":[{\"type\":\"text\",\"text\":\"a\"}],\"stop_reason\":null}}\n",
            "{\"type\":\"result\",\"subtype\":\"success\",\"session_id\":null,\"is_error\":false}\n",
        ]);
        let parsed: Vec<_> = parse_ndjson(s).collect().await;
        assert_eq!(parsed.len(), 2);
        assert!(matches!(parsed[0].as_ref().unwrap(), OutputEvent::Assistant { .. }));
        assert!(matches!(parsed[1].as_ref().unwrap(), OutputEvent::Result(_)));
    }

    #[tokio::test]
    async fn handles_split_lines() {
        let s = bytes_stream(vec![
            "{\"type\":\"assistant\",\"messa",
            "ge\":{\"id\":null,\"model\":null,\"content\":[{\"type\":\"text\",\"text\":\"a\"}],\"stop_reason\":null}}\n",
        ]);
        let parsed: Vec<_> = parse_ndjson(s).collect().await;
        assert_eq!(parsed.len(), 1);
        assert!(matches!(parsed[0].as_ref().unwrap(), OutputEvent::Assistant { .. }));
    }

    #[tokio::test]
    async fn parses_trailing_line_without_newline() {
        let s = bytes_stream(vec![
            "{\"type\":\"result\",\"subtype\":\"success\",\"session_id\":null,\"is_error\":false}",
        ]);
        let parsed: Vec<_> = parse_ndjson(s).collect().await;
        assert_eq!(parsed.len(), 1);
        assert!(matches!(parsed[0].as_ref().unwrap(), OutputEvent::Result(_)));
    }

    #[tokio::test]
    async fn skips_empty_lines() {
        let s = bytes_stream(vec![
            "\n\n",
            "{\"type\":\"result\",\"subtype\":\"success\",\"session_id\":null,\"is_error\":false}\n",
        ]);
        let parsed: Vec<_> = parse_ndjson(s).collect().await;
        assert_eq!(parsed.len(), 1);
    }

    #[tokio::test]
    async fn surfaces_parse_error_with_context() {
        let s = bytes_stream(vec!["{not json}\n"]);
        let parsed: Vec<_> = parse_ndjson(s).collect().await;
        assert_eq!(parsed.len(), 1);
        match &parsed[0] {
            Err(Error::UnexpectedResponse(msg)) => assert!(msg.contains("not json")),
            other => panic!("expected UnexpectedResponse, got {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Add `bytes` to dependencies**

In `crates/anyllm-claude-code/Cargo.toml`, add to `[dependencies]`:

```toml
bytes = "1"
```

- [ ] **Step 3: Add `mod streaming;` to `lib.rs`**

```rust
mod streaming;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p anyllm-claude-code streaming::tests
```

Expected: All five tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/anyllm-claude-code
git commit -m "feat(claude-code): NDJSON stream-json parser"
```

### Task 8: Map OutputEvents to anyllm StreamEvents

**Files:**
- Modify: `crates/anyllm-claude-code/src/streaming.rs`

- [ ] **Step 1: Write failing tests**

Append to `crates/anyllm-claude-code/src/streaming.rs`:

```rust
// ---------- OutputEvent → anyllm::StreamEvent mapping ----------

use anyllm::{ContentBlock, FinishReason, StreamBlockType, Usage, UsageMetadataMode};
use crate::wire::{OutputContentBlock, ResultEvent, UsageBlock};

/// Convert one [`OutputEvent`] into zero or more [`StreamEvent`]s.
///
/// `next_block_index` is incremented per content block emitted so
/// block-start / delta / stop events form a coherent indexed stream.
pub(crate) struct StreamEventMapper {
    next_block_index: usize,
    response_started: bool,
}

impl StreamEventMapper {
    pub(crate) fn new() -> Self {
        Self { next_block_index: 0, response_started: false }
    }

    /// Translate one output event into the corresponding stream events.
    pub(crate) fn map(&mut self, event: OutputEvent) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        match event {
            OutputEvent::System { .. } | OutputEvent::User { .. } => {
                // Session/model-switch system notices and tool_result
                // echoes have no portable analogue.
            }
            OutputEvent::Assistant { message } => {
                if !self.response_started {
                    self.response_started = true;
                    out.push(StreamEvent::ResponseStart {
                        id: message.id.clone(),
                        model: message.model.clone(),
                    });
                }
                for block in message.content {
                    let idx = self.next_block_index;
                    self.next_block_index += 1;
                    match block {
                        OutputContentBlock::Text { text } => {
                            out.push(StreamEvent::BlockStart {
                                index: idx,
                                block_type: StreamBlockType::Text,
                                id: None,
                                name: None,
                                type_name: None,
                                data: None,
                            });
                            if !text.is_empty() {
                                out.push(StreamEvent::TextDelta { index: idx, text });
                            }
                            out.push(StreamEvent::BlockStop { index: idx });
                        }
                        OutputContentBlock::Thinking { thinking, signature } => {
                            out.push(StreamEvent::BlockStart {
                                index: idx,
                                block_type: StreamBlockType::Reasoning,
                                id: None,
                                name: None,
                                type_name: None,
                                data: None,
                            });
                            if !thinking.is_empty() {
                                out.push(StreamEvent::ReasoningDelta {
                                    index: idx,
                                    text: thinking,
                                    signature,
                                });
                            }
                            out.push(StreamEvent::BlockStop { index: idx });
                        }
                        OutputContentBlock::ToolUse { id, name, input } => {
                            let tool_name = strip_mcp_prefix(&name).to_string();
                            out.push(StreamEvent::BlockStart {
                                index: idx,
                                block_type: StreamBlockType::ToolCall,
                                id: Some(id),
                                name: Some(tool_name),
                                type_name: None,
                                data: None,
                            });
                            out.push(StreamEvent::ToolCallDelta {
                                index: idx,
                                arguments: serde_json::to_string(&input)
                                    .unwrap_or_else(|_| String::from("{}")),
                            });
                            out.push(StreamEvent::BlockStop { index: idx });
                        }
                        OutputContentBlock::ToolResult { .. } => {
                            // Not emitted by Claude in assistant messages.
                        }
                    }
                }
            }
            OutputEvent::Result(r) => {
                let finish = map_finish_reason(&r);
                let usage = r.usage.as_ref().map(usage_block_to_anyllm);
                out.push(StreamEvent::ResponseMetadata {
                    finish_reason: Some(finish),
                    usage,
                    usage_mode: UsageMetadataMode::Snapshot,
                    id: None,
                    model: None,
                    metadata: Default::default(),
                });
                out.push(StreamEvent::ResponseStop);
            }
        }
        out
    }
}

fn strip_mcp_prefix(name: &str) -> &str {
    name.strip_prefix("mcp__anyllm__").unwrap_or(name)
}

pub(crate) fn map_finish_reason(r: &ResultEvent) -> FinishReason {
    if r.is_error {
        return FinishReason::Other(format!("error:{}", r.subtype));
    }
    match r.subtype.as_str() {
        "success" => FinishReason::Stop,
        "error_max_turns" => FinishReason::Length,
        other => FinishReason::Other(other.to_string()),
    }
}

pub(crate) fn usage_block_to_anyllm(u: &UsageBlock) -> Usage {
    Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
        cache_read_input_tokens: u.cache_read_input_tokens,
        ..Default::default()
    }
}

/// Collect a stream of [`OutputEvent`]s into a final [`anyllm::ChatResponse`].
///
/// Used by the non-streaming `chat()` path. Returns the response only
/// after a [`OutputEvent::Result`] has been seen; if the stream ends
/// before that, returns [`Error::UnexpectedResponse`].
pub(crate) async fn collect_into_response<S>(stream: S) -> Result<anyllm::ChatResponse>
where
    S: Stream<Item = Result<OutputEvent>>,
{
    use anyllm::{ChatResponse, ResponseMetadata};
    futures_util::pin_mut!(stream);

    let mut content: Vec<ContentBlock> = Vec::new();
    let mut finish: Option<FinishReason> = None;
    let mut usage: Option<Usage> = None;
    let mut model: Option<String> = None;
    let mut id: Option<String> = None;

    while let Some(event) = stream.next().await {
        match event? {
            OutputEvent::System { .. } | OutputEvent::User { .. } => {}
            OutputEvent::Assistant { message } => {
                if model.is_none() { model = message.model.clone(); }
                if id.is_none() { id = message.id.clone(); }
                for block in message.content {
                    match block {
                        OutputContentBlock::Text { text } => {
                            content.push(ContentBlock::Text { text })
                        }
                        OutputContentBlock::Thinking { thinking, signature } => {
                            content.push(ContentBlock::Reasoning {
                                text: thinking,
                                signature,
                            })
                        }
                        OutputContentBlock::ToolUse { id, name, input } => {
                            content.push(ContentBlock::ToolCall {
                                id,
                                name: strip_mcp_prefix(&name).to_string(),
                                arguments: serde_json::to_string(&input)
                                    .unwrap_or_else(|_| String::from("{}")),
                            })
                        }
                        OutputContentBlock::ToolResult { .. } => {}
                    }
                }
            }
            OutputEvent::Result(r) => {
                finish = Some(map_finish_reason(&r));
                if r.usage.is_some() {
                    usage = r.usage.as_ref().map(usage_block_to_anyllm);
                }
            }
        }
    }

    if finish.is_none() {
        return Err(Error::UnexpectedResponse(
            "stream ended before claude emitted a result event".into(),
        ));
    }

    Ok(ChatResponse {
        content,
        finish_reason: finish,
        usage,
        model,
        id,
        metadata: ResponseMetadata::new(),
    })
}

#[cfg(test)]
mod mapping_tests {
    use super::*;
    use crate::wire::*;

    #[test]
    fn maps_assistant_text_to_blockstart_delta_blockstop() {
        let mut m = StreamEventMapper::new();
        let evt = OutputEvent::Assistant {
            message: AssistantMessage {
                id: Some("m1".into()),
                model: Some("claude-sonnet-4-6".into()),
                content: vec![OutputContentBlock::Text { text: "hello".into() }],
                stop_reason: None,
                usage: None,
            },
        };
        let events = m.map(evt);
        assert!(matches!(events[0], StreamEvent::ResponseStart { .. }));
        assert!(matches!(events[1], StreamEvent::BlockStart { .. }));
        assert!(matches!(events[2], StreamEvent::TextDelta { .. }));
        assert!(matches!(events[3], StreamEvent::BlockStop { index: 0 }));
    }

    #[test]
    fn strips_mcp_prefix_from_tool_call_name() {
        let mut m = StreamEventMapper::new();
        let evt = OutputEvent::Assistant {
            message: AssistantMessage {
                id: None,
                model: None,
                content: vec![OutputContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "mcp__anyllm__get_weather".into(),
                    input: serde_json::json!({"loc": "SF"}),
                }],
                stop_reason: Some("tool_use".into()),
                usage: None,
            },
        };
        let events = m.map(evt);
        match events.iter().find(|e| matches!(e, StreamEvent::BlockStart { .. })) {
            Some(StreamEvent::BlockStart {
                block_type: StreamBlockType::ToolCall,
                name: Some(n),
                ..
            }) => assert_eq!(n, "get_weather"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn maps_result_to_metadata_then_stop_with_usage() {
        let mut m = StreamEventMapper::new();
        let evt = OutputEvent::Result(ResultEvent {
            subtype: "success".into(),
            session_id: None,
            is_error: false,
            duration_ms: None,
            num_turns: None,
            usage: Some(UsageBlock {
                input_tokens: Some(7),
                output_tokens: Some(2),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }),
            result: None,
            error: None,
        });
        let events = m.map(evt);
        assert_eq!(events.len(), 2);
        match &events[0] {
            StreamEvent::ResponseMetadata { finish_reason, usage, .. } => {
                assert_eq!(*finish_reason, Some(FinishReason::Stop));
                assert_eq!(usage.as_ref().unwrap().input_tokens, Some(7));
            }
            other => panic!("unexpected metadata event: {other:?}"),
        }
        assert!(matches!(events[1], StreamEvent::ResponseStop));
    }

    #[test]
    fn maps_max_turns_to_finish_length() {
        let mut m = StreamEventMapper::new();
        let evt = OutputEvent::Result(ResultEvent {
            subtype: "error_max_turns".into(),
            session_id: None,
            is_error: false,
            duration_ms: None,
            num_turns: None,
            usage: None,
            result: None,
            error: None,
        });
        let events = m.map(evt);
        match &events[0] {
            StreamEvent::ResponseMetadata { finish_reason, .. } => {
                assert_eq!(*finish_reason, Some(FinishReason::Length));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn collect_into_response_assembles_text_and_usage() {
        let events = vec![
            Ok(OutputEvent::Assistant {
                message: AssistantMessage {
                    id: Some("m1".into()),
                    model: Some("claude-sonnet-4-6".into()),
                    content: vec![OutputContentBlock::Text { text: "hi".into() }],
                    stop_reason: None,
                    usage: None,
                },
            }),
            Ok(OutputEvent::Result(ResultEvent {
                subtype: "success".into(),
                session_id: None,
                is_error: false,
                duration_ms: None,
                num_turns: Some(1),
                usage: Some(UsageBlock {
                    input_tokens: Some(7),
                    output_tokens: Some(2),
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                }),
                result: Some("hi".into()),
                error: None,
            })),
        ];
        let stream = futures_util::stream::iter(events);
        let resp = collect_into_response(stream).await.unwrap();
        assert_eq!(resp.text(), Some("hi".into()));
        assert_eq!(resp.usage.unwrap().output_tokens, Some(2));
        assert_eq!(resp.model.as_deref(), Some("claude-sonnet-4-6"));
        assert_eq!(resp.id.as_deref(), Some("m1"));
        assert_eq!(resp.finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn collect_errors_when_no_result_event() {
        let events = vec![Ok(OutputEvent::Assistant {
            message: AssistantMessage {
                id: None, model: None,
                content: vec![OutputContentBlock::Text { text: "x".into() }],
                stop_reason: None, usage: None,
            },
        })];
        let stream = futures_util::stream::iter(events);
        let err = collect_into_response(stream).await.unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse(_)));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p anyllm-claude-code streaming
```

Expected: All mapping tests plus the earlier parser tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/anyllm-claude-code/src/streaming.rs
git commit -m "feat(claude-code): map stream-json events to anyllm StreamEvent"
```

---

## Phase 5 — Request rendering (anyllm → stream-json input + CLI args)

### Task 9: Render messages to stream-json input lines

**Files:**
- Create: `crates/anyllm-claude-code/src/render.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod render;`)

- [ ] **Step 1: Write failing tests**

Create `crates/anyllm-claude-code/src/render.rs`:

```rust
//! Render an [`anyllm::ChatRequest`] into the stream-json input lines and
//! CLI arguments expected by `claude --input-format stream-json`.

use anyllm::{
    ChatRequest, ContentBlock, ContentPart, ImageSource as AnyllmImageSource, Message,
    Result, ToolResultContent, UserContent,
};

use crate::wire::{
    ImageSource, InputAssistantMessage, InputContentBlock, InputEvent, InputUserMessage,
};

/// Render the request's `messages` into one stream-json input event per
/// turn. Each event will be JSON-serialized and newline-terminated when
/// piped to claude's stdin.
pub(crate) fn render_messages(req: &ChatRequest) -> Result<Vec<InputEvent>> {
    let mut out = Vec::new();
    for msg in &req.messages {
        out.push(render_message(msg)?);
    }
    Ok(out)
}

fn render_message(msg: &Message) -> Result<InputEvent> {
    match msg {
        Message::User { content, .. } => Ok(InputEvent::User {
            message: InputUserMessage {
                role: "user".into(),
                content: render_user_content(content)?,
            },
        }),
        Message::Assistant { content, .. } => Ok(InputEvent::Assistant {
            message: InputAssistantMessage {
                role: "assistant".into(),
                content: render_assistant_content(content)?,
            },
        }),
        Message::Tool { tool_call_id, content, is_error, .. } => {
            // Tool results are sent inside the *next* user turn in
            // Anthropic's wire format. The anyllm Message::Tool variant
            // preserves the historical "tool" role; we collapse it into
            // a user message containing a single tool_result block here.
            Ok(InputEvent::User {
                message: InputUserMessage {
                    role: "user".into(),
                    content: vec![InputContentBlock::ToolResult {
                        tool_use_id: tool_call_id.clone(),
                        is_error: *is_error,
                        content: tool_result_content_to_json(content),
                    }],
                },
            })
        }
    }
}

fn render_user_content(content: &UserContent) -> Result<Vec<InputContentBlock>> {
    match content {
        UserContent::Text(text) => Ok(vec![InputContentBlock::Text { text: text.clone() }]),
        UserContent::Parts(parts) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        out.push(InputContentBlock::Text { text: text.clone() })
                    }
                    ContentPart::Image { source, .. } => {
                        // `detail` is silently dropped (capability matrix: ImageDetail Unsupported).
                        out.push(InputContentBlock::Image {
                            source: anyllm_image_to_wire(source),
                        })
                    }
                    ContentPart::Other { .. } => {
                        // Provider-specific parts have no portable encoding;
                        // drop silently.
                    }
                }
            }
            Ok(out)
        }
    }
}

fn render_assistant_content(blocks: &[ContentBlock]) -> Result<Vec<InputContentBlock>> {
    let mut out = Vec::with_capacity(blocks.len());
    for b in blocks {
        match b {
            ContentBlock::Text { text } => {
                out.push(InputContentBlock::Text { text: text.clone() })
            }
            ContentBlock::Reasoning { text, signature } => {
                out.push(InputContentBlock::Thinking {
                    thinking: text.clone(),
                    signature: signature.clone(),
                })
            }
            ContentBlock::ToolCall { id, name, arguments } => {
                let input: serde_json::Value = serde_json::from_str(arguments)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                out.push(InputContentBlock::ToolUse {
                    id: id.clone(),
                    // We re-attach the mcp prefix here so Claude recognizes
                    // these as MCP tool calls when re-reading prior turns.
                    name: format!("mcp__anyllm__{name}"),
                    input,
                });
            }
            ContentBlock::Image { .. } => {
                // ImageOutput is Unsupported per the capability matrix;
                // refuse to round-trip a foreign assistant image block.
                return Err(anyllm::Error::Unsupported(
                    "claude-code: assistant ImageBlock cannot be sent back to claude".into(),
                ));
            }
            ContentBlock::Other { .. } => {
                // Non-portable provider blocks are silently dropped on input.
            }
        }
    }
    Ok(out)
}

fn anyllm_image_to_wire(src: &AnyllmImageSource) -> ImageSource {
    match src {
        AnyllmImageSource::Base64 { media_type, data } => ImageSource::Base64 {
            media_type: media_type.clone(),
            data: data.clone(),
        },
        AnyllmImageSource::Url { url } => ImageSource::Url { url: url.clone() },
    }
}

fn tool_result_content_to_json(content: &ToolResultContent) -> serde_json::Value {
    match content {
        ToolResultContent::Text(s) => serde_json::Value::String(s.clone()),
        ToolResultContent::Json(v) => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{ChatRequest, ContentPart, Message, UserContent};

    /// Helper: build an assistant message from raw content blocks. The
    /// exact public constructor on `Message` may differ — check
    /// `crates/anyllm/src/chat/message.rs` and adjust if needed (e.g.
    /// `Message::assistant_blocks(...)` or struct-literal `Message::Assistant
    /// { content, name: None, extensions: None }`).
    fn assistant(blocks: Vec<ContentBlock>) -> Message {
        Message::Assistant { content: blocks, name: None, extensions: None }
    }

    #[test]
    fn renders_single_user_text_message() {
        let req = ChatRequest::new("claude-sonnet-4-6").user("hi");
        let lines = render_messages(&req).unwrap();
        assert_eq!(lines.len(), 1);
        let json = serde_json::to_string(&lines[0]).unwrap();
        assert!(json.contains(r#""type":"user""#));
        assert!(json.contains(r#""text":"hi""#));
    }

    #[test]
    fn renders_assistant_tool_call_with_mcp_prefix() {
        let req = ChatRequest::new("claude-sonnet-4-6")
            .user("get weather")
            .message(assistant(vec![ContentBlock::ToolCall {
                id: "toolu_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"loc":"SF"}"#.into(),
            }]));
        let lines = render_messages(&req).unwrap();
        let assistant_json = serde_json::to_string(&lines[1]).unwrap();
        assert!(assistant_json.contains(r#""name":"mcp__anyllm__get_weather""#));
        assert!(assistant_json.contains(r#""id":"toolu_1""#));
    }

    #[test]
    fn renders_tool_message_as_user_tool_result() {
        let req = ChatRequest::new("claude-sonnet-4-6")
            .user("get weather")
            .message(Message::tool_result("toolu_1", "get_weather", "Sunny, 72F"));
        let lines = render_messages(&req).unwrap();
        let tool_json = serde_json::to_string(&lines[1]).unwrap();
        assert!(tool_json.contains(r#""type":"user""#));
        assert!(tool_json.contains(r#""type":"tool_result""#));
        assert!(tool_json.contains(r#""tool_use_id":"toolu_1""#));
    }

    #[test]
    fn rejects_assistant_image_block() {
        // ContentBlock::Image variant fields differ from this sketch —
        // adjust to match `crates/anyllm/src/chat/content.rs`.
        let req = ChatRequest::new("claude-sonnet-4-6")
            .user("ok")
            .message(assistant(vec![ContentBlock::Image {
                source: AnyllmImageSource::Url { url: "x".into() },
            }]));
        let err = render_messages(&req).unwrap_err();
        assert!(matches!(err, anyllm::Error::Unsupported(_)));
    }

    #[test]
    fn renders_user_image_part_base64() {
        let req = ChatRequest::new("claude-sonnet-4-6").message(Message::User {
            content: UserContent::Parts(vec![ContentPart::Image {
                source: AnyllmImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "aGVsbG8=".into(),
                },
                detail: None,
            }]),
            name: None,
            extensions: None,
        });
        let lines = render_messages(&req).unwrap();
        let json = serde_json::to_string(&lines[0]).unwrap();
        assert!(json.contains(r#""media_type":"image/png""#));
    }
}
```

- [ ] **Step 2: Add `mod render;` to `lib.rs`**

```rust
mod render;
```

- [ ] **Step 3: Run tests, debug message-construction call sites if needed**

```bash
cargo test -p anyllm-claude-code render::tests
```

The exact constructor names (`Message::assistant_with_blocks`, `ContentPart::Image(ImagePartRef { ... })`, `Message::tool(...)`, `Message::User { ... }`) follow the patterns in `crates/anyllm/src/chat/message.rs`. Read that file and adjust the test setup if the public constructors differ; the production code in this task only depends on the public `Message`, `ContentPart`, `ContentBlock`, `ImageSource`, and `ToolResultContent` types.

Expected: All five tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/anyllm-claude-code/src
git commit -m "feat(claude-code): render anyllm messages into stream-json input"
```

### Task 10: Render the CLI invocation (args + env)

**Files:**
- Modify: `crates/anyllm-claude-code/src/render.rs`

- [ ] **Step 1: Append failing tests for argv and env rendering**

Append to `crates/anyllm-claude-code/src/render.rs`:

```rust
// ---------- CLI args + env rendering ----------

use std::ffi::OsString;
use std::path::Path;

use anyllm::{ReasoningConfig, ReasoningEffort, ResponseFormat, SystemPrompt, ToolChoice};

/// The exhaustive list of Claude Code's built-in tools, passed via
/// `--disallowed-tools` to leave only MCP-served tools available.
///
/// Updated to match Claude CLI version recorded in the spike notes.
pub(crate) const DISALLOWED_BUILTIN_TOOLS: &[&str] = &[
    "Bash",
    "BashOutput",
    "Edit",
    "Glob",
    "Grep",
    "KillShell",
    "MultiEdit",
    "NotebookEdit",
    "Read",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
];

/// One-off MCP server descriptor injected into `--mcp-config`.
pub(crate) struct McpEndpoint<'a> {
    pub url: &'a str,
    pub bearer_token: &'a str,
}

/// Build the inline JSON value used as the argument to `--mcp-config`.
pub(crate) fn build_mcp_config_json(ep: &McpEndpoint<'_>) -> String {
    serde_json::json!({
        "mcpServers": {
            "anyllm": {
                "type": "http",
                "url": ep.url,
                "headers": {
                    "Authorization": format!("Bearer {}", ep.bearer_token)
                }
            }
        }
    })
    .to_string()
}

/// Build the argv (excluding `program`) for the claude subprocess.
pub(crate) fn build_argv(req: &ChatRequest, mcp: &McpEndpoint<'_>) -> Vec<OsString> {
    let mut argv: Vec<OsString> = vec![
        "-p".into(),
        "--input-format".into(),
        "stream-json".into(),
        "--output-format".into(),
        "stream-json".into(),
        "--model".into(),
        req.model.clone().into(),
        "--strict-mcp-config".into(),
        "--mcp-config".into(),
        build_mcp_config_json(mcp).into(),
    ];

    let system_concat = render_system_prompt(&req.system);
    if !system_concat.is_empty() {
        argv.push("--system-prompt".into());
        argv.push(system_concat.into());
    }

    argv.push("--disallowed-tools".into());
    argv.push(DISALLOWED_BUILTIN_TOOLS.join(",").into());

    // ToolChoice nudges (Required / Specific) are appended into the
    // system prompt above by render_system_prompt; nothing else to do here.

    // ResponseFormat — anything other than text/None is unsupported natively.
    if let Some(fmt) = req.response_format.as_ref() {
        if !matches!(fmt, ResponseFormat::Text) {
            // Returned as Err by the Provider before we even build argv;
            // see chat::Provider::chat. Defensive no-op here.
        }
    }

    argv
}

/// Concatenate all system messages with `\n\n`. Tool-choice nudges for
/// Required/Specific are appended as additional sentences so Claude sees
/// them in the system prompt.
pub(crate) fn render_system_prompt(system: &[SystemPrompt]) -> String {
    let mut parts: Vec<String> = system.iter().map(|s| s.content.clone()).collect();
    parts.retain(|s| !s.is_empty());
    parts.join("\n\n")
}

pub(crate) fn append_tool_choice_nudge(prompt: &mut String, choice: &ToolChoice) {
    let nudge = match choice {
        ToolChoice::Auto | ToolChoice::None => return,
        ToolChoice::Required => Some(
            "You must call one of the available tools to answer this turn — do not respond in plain text.".to_string(),
        ),
        ToolChoice::Specific { name } => Some(format!(
            "You must call the tool named `{name}` to answer this turn — do not respond in plain text and do not call any other tool."
        )),
    };
    if let Some(n) = nudge {
        if !prompt.is_empty() {
            prompt.push_str("\n\n");
        }
        prompt.push_str(&n);
    }
}

/// Build the per-call env-var set per spec §8.
pub(crate) fn build_env(
    oauth_token: &str,
    scratch_dir: &Path,
    fake_home: &Path,
    reasoning: Option<&ReasoningConfig>,
) -> Vec<(OsString, OsString)> {
    let mut env: Vec<(OsString, OsString)> = vec![
        ("PATH".into(), default_path().into()),
        ("LANG".into(), "C.UTF-8".into()),
        ("TZ".into(), "UTC".into()),
        ("HOME".into(), fake_home.as_os_str().to_owned()),
        ("CLAUDE_CODE_OAUTH_TOKEN".into(), oauth_token.into()),
        ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".into(), "1".into()),
        ("CLAUDE_CODE_SKIP_PROMPT_HISTORY".into(), "1".into()),
        ("CLAUDE_CODE_DISABLE_CLAUDE_MDS".into(), "1".into()),
        ("CLAUDE_CODE_DISABLE_AUTO_MEMORY".into(), "1".into()),
        ("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS".into(), "1".into()),
        ("CLAUDE_CODE_DISABLE_CRON".into(), "1".into()),
        ("CLAUDE_CODE_AUTO_CONNECT_IDE".into(), "false".into()),
        (
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL".into(),
            "1".into(),
        ),
        ("CLAUDE_CODE_DISABLE_POLICY_SKILLS".into(), "1".into()),
        ("CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS".into(), "1".into()),
        ("CLAUDE_CODE_SIMPLE".into(), "1".into()),
        (
            "CLAUDE_CODE_TMPDIR".into(),
            scratch_dir.join("tmp").as_os_str().to_owned(),
        ),
        (
            "CLAUDE_CODE_PLUGIN_CACHE_DIR".into(),
            scratch_dir.join("plugins").as_os_str().to_owned(),
        ),
        (
            "CLAUDE_CODE_DEBUG_LOGS_DIR".into(),
            scratch_dir.join("debug").as_os_str().to_owned(),
        ),
    ];

    if let Some(r) = reasoning {
        if !r.enabled {
            env.push(("CLAUDE_CODE_DISABLE_THINKING".into(), "1".into()));
        } else if let Some(effort) = r.effort {
            env.push((
                "CLAUDE_CODE_EFFORT_LEVEL".into(),
                map_effort(effort).into(),
            ));
        }
    }

    env
}

fn default_path() -> &'static str {
    if cfg!(windows) {
        "C:\\Windows\\System32;C:\\Windows;C:\\Program Files\\nodejs"
    } else {
        "/usr/local/bin:/usr/bin:/bin"
    }
}

fn map_effort(e: ReasoningEffort) -> &'static str {
    match e {
        ReasoningEffort::Minimal => "low",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
    }
}

#[cfg(test)]
mod argv_tests {
    use super::*;
    use std::path::PathBuf;

    fn ep() -> McpEndpoint<'static> {
        McpEndpoint { url: "http://127.0.0.1:54321/mcp", bearer_token: "tok" }
    }

    #[test]
    fn argv_includes_stream_json_and_strict_mcp() {
        let req = ChatRequest::new("claude-sonnet-4-6").user("hi");
        let argv = build_argv(&req, &ep());
        let joined: Vec<String> = argv.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert!(joined.iter().any(|s| s == "--input-format"));
        assert!(joined.iter().any(|s| s == "stream-json"));
        assert!(joined.iter().any(|s| s == "--strict-mcp-config"));
        assert!(joined.iter().any(|s| s == "claude-sonnet-4-6"));
    }

    #[test]
    fn argv_disallows_all_builtin_tools() {
        let req = ChatRequest::new("claude-sonnet-4-6").user("hi");
        let argv = build_argv(&req, &ep());
        let pos = argv.iter().position(|s| s == "--disallowed-tools").unwrap();
        let list = argv[pos + 1].to_string_lossy().into_owned();
        assert!(list.contains("Bash"));
        assert!(list.contains("Read"));
        assert!(list.contains("Write"));
        assert!(list.contains("WebFetch"));
    }

    #[test]
    fn mcp_config_json_includes_bearer_token() {
        let json = build_mcp_config_json(&ep());
        assert!(json.contains(r#""url":"http://127.0.0.1:54321/mcp""#));
        assert!(json.contains(r#""Authorization":"Bearer tok""#));
        // Sanity: parses back as an object.
        let _v: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn env_includes_all_lockdown_vars() {
        let env = build_env("oauth", &PathBuf::from("/tmp/scratch"), &PathBuf::from("/tmp/scratch/home"), None);
        let keys: Vec<String> = env.iter().map(|(k, _)| k.to_string_lossy().into_owned()).collect();
        for required in &[
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS",
            "CLAUDE_CODE_DISABLE_CRON",
            "CLAUDE_CODE_AUTO_CONNECT_IDE",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL",
            "CLAUDE_CODE_DISABLE_POLICY_SKILLS",
            "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS",
            "CLAUDE_CODE_SIMPLE",
            "CLAUDE_CODE_TMPDIR",
            "CLAUDE_CODE_PLUGIN_CACHE_DIR",
            "CLAUDE_CODE_DEBUG_LOGS_DIR",
            "HOME",
        ] {
            assert!(keys.iter().any(|k| k == required), "missing {required}");
        }
    }

    #[test]
    fn env_maps_reasoning_disabled() {
        let env = build_env(
            "oauth",
            &PathBuf::from("/tmp/scratch"),
            &PathBuf::from("/tmp/scratch/home"),
            Some(&ReasoningConfig { enabled: false, budget_tokens: None, effort: None }),
        );
        assert!(env.iter().any(|(k, v)| k == "CLAUDE_CODE_DISABLE_THINKING" && v == "1"));
    }

    #[test]
    fn env_maps_reasoning_effort_high() {
        let env = build_env(
            "oauth",
            &PathBuf::from("/tmp/scratch"),
            &PathBuf::from("/tmp/scratch/home"),
            Some(&ReasoningConfig { enabled: true, budget_tokens: None, effort: Some(ReasoningEffort::High) }),
        );
        assert!(env.iter().any(|(k, v)| k == "CLAUDE_CODE_EFFORT_LEVEL" && v == "high"));
    }

    #[test]
    fn tool_choice_nudge_required_appended() {
        let mut prompt = "You are concise.".to_string();
        append_tool_choice_nudge(&mut prompt, &ToolChoice::Required);
        assert!(prompt.contains("You must call one of the available tools"));
    }

    #[test]
    fn tool_choice_nudge_specific_includes_name() {
        let mut prompt = String::new();
        append_tool_choice_nudge(&mut prompt, &ToolChoice::Specific { name: "search".into() });
        assert!(prompt.contains("`search`"));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p anyllm-claude-code render
```

Expected: All argv and env tests plus the earlier message-rendering tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/anyllm-claude-code/src/render.rs
git commit -m "feat(claude-code): render CLI argv and lockdown env vars"
```

---

## Phase 6 — In-process MCP HTTP server

### Task 11: Bearer-authenticated HTTP MCP server scaffolding

**Files:**
- Create: `crates/anyllm-claude-code/src/mcp.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod mcp;`)

The MCP HTTP transport speaks JSON-RPC 2.0 over POST. v1 implements the minimal subset Claude exercises: `initialize`, `tools/list`, `tools/call`. (If the spike showed Claude calling other methods, add those here.)

- [ ] **Step 1: Write failing test for initialize**

Create `crates/anyllm-claude-code/src/mcp.rs`:

```rust
//! In-process HTTP MCP server bridging anyllm tool definitions into Claude.
//!
//! Spawned per `chat()` call on `127.0.0.1:0`. Authenticated by a one-shot
//! bearer token. JSON-RPC 2.0 over POST; supports the subset of MCP that
//! `claude` exercises: `initialize`, `tools/list`, `tools/call`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use anyllm::{Result, Tool};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Async tool handler. Returns the tool's textual result on success.
pub(crate) type ToolHandler =
    Arc<dyn Fn(Value) -> futures_core::future::BoxFuture<'static, Result<String>> + Send + Sync>;

/// Per-call MCP server: knows the request's tools and the auth token.
pub(crate) struct McpServer {
    pub addr: SocketAddr,
    pub bearer_token: String,
    shutdown: Option<oneshot::Sender<()>>,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl McpServer {
    /// Bind on `127.0.0.1:0`, register the given tool definitions, and
    /// start serving in the background.
    pub(crate) async fn start(
        tools: Vec<Tool>,
        handlers: HashMap<String, ToolHandler>,
        bearer_token: String,
    ) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|e| {
            anyllm::Error::Provider {
                status: None,
                message: format!("MCP server bind failed: {e}"),
                body: None,
                request_id: None,
            }
        })?;
        let addr = listener.local_addr().map_err(|e| anyllm::Error::Provider {
            status: None,
            message: format!("MCP server local_addr failed: {e}"),
            body: None,
            request_id: None,
        })?;

        let state = Arc::new(McpState { tools, handlers, bearer_token: bearer_token.clone() });
        let app = Router::new()
            .route("/mcp", post(handle_jsonrpc))
            .with_state(state);

        let (tx, rx) = oneshot::channel();
        let join = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await;
        });

        Ok(Self {
            addr,
            bearer_token,
            shutdown: Some(tx),
            join: Some(join),
        })
    }

    /// Signal graceful shutdown and await the task.
    pub(crate) async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        // Don't await join in Drop (we may not be in async context); the
        // axum task is best-effort cancelled by graceful_shutdown firing.
    }
}

struct McpState {
    tools: Vec<Tool>,
    handlers: HashMap<String, ToolHandler>,
    bearer_token: String,
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

async fn handle_jsonrpc(
    State(state): State<Arc<McpState>>,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    if !auth_ok(&headers, &state.bearer_token) {
        return (StatusCode::UNAUTHORIZED, "missing or bad bearer token").into_response();
    }
    if req.jsonrpc != "2.0" {
        return rpc_err(req.id, -32600, "invalid Request: jsonrpc must be \"2.0\"").into_response();
    }

    let result = match req.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "anyllm", "version": env!("CARGO_PKG_VERSION") }
        })),
        "tools/list" => Ok(json!({
            "tools": state.tools.iter().map(|t| json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.parameters,
            })).collect::<Vec<_>>()
        })),
        "tools/call" => match call_tool(&state, &req.params).await {
            Ok(text) => Ok(json!({
                "content": [{ "type": "text", "text": text }]
            })),
            Err(e) => Ok(json!({
                "content": [{ "type": "text", "text": format!("error: {e}") }],
                "isError": true,
            })),
        },
        other => Err((-32601, format!("method not found: {other}"))),
    };

    match result {
        Ok(value) => Json(JsonRpcResponse {
            jsonrpc: "2.0",
            id: req.id,
            result: Some(value),
            error: None,
        })
        .into_response(),
        Err((code, message)) => rpc_err(req.id, code, &message).into_response(),
    }
}

async fn call_tool(state: &McpState, params: &Value) -> Result<String> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyllm::Error::InvalidRequest("missing tools/call name".into()))?;
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    let handler = state
        .handlers
        .get(name)
        .ok_or_else(|| anyllm::Error::InvalidRequest(format!("no handler for tool {name}")))?
        .clone();
    handler(arguments).await
}

fn auth_ok(headers: &HeaderMap, expected: &str) -> bool {
    let Some(h) = headers.get(axum::http::header::AUTHORIZATION) else { return false; };
    let Ok(s) = h.to_str() else { return false; };
    s == format!("Bearer {expected}")
}

fn rpc_err(id: Option<Value>, code: i32, message: &str) -> Json<JsonRpcResponse> {
    Json(JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(JsonRpcError { code, message: message.into() }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::Tool;

    fn token() -> String { "test-token-abc".to_string() }

    async fn start_with_one_tool() -> McpServer {
        let tool = Tool::new("ping", json!({"type":"object","properties":{}})).description("Ping");
        let mut handlers: HashMap<String, ToolHandler> = HashMap::new();
        handlers.insert(
            "ping".to_string(),
            Arc::new(|_args| Box::pin(async { Ok::<_, anyllm::Error>("pong".to_string()) })),
        );
        McpServer::start(vec![tool], handlers, token()).await.unwrap()
    }

    async fn rpc(server: &McpServer, body: Value) -> Value {
        let url = format!("http://{}/mcp", server.addr);
        let resp = reqwest::Client::new()
            .post(&url)
            .header("Authorization", format!("Bearer {}", server.bearer_token))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        resp.json().await.unwrap()
    }

    #[tokio::test]
    async fn rejects_missing_bearer_token() {
        let server = start_with_one_tool().await;
        let url = format!("http://{}/mcp", server.addr);
        let resp = reqwest::Client::new()
            .post(&url)
            .json(&json!({"jsonrpc":"2.0","id":1,"method":"initialize"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
        server.stop().await;
    }

    #[tokio::test]
    async fn initialize_returns_capabilities() {
        let server = start_with_one_tool().await;
        let body: Value = rpc(&server, json!({"jsonrpc":"2.0","id":1,"method":"initialize"})).await;
        assert_eq!(body["jsonrpc"], "2.0");
        assert_eq!(body["result"]["protocolVersion"], "2024-11-05");
        server.stop().await;
    }

    #[tokio::test]
    async fn tools_list_returns_registered_tool() {
        let server = start_with_one_tool().await;
        let body = rpc(&server, json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).await;
        let tools = body["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "ping");
    }

    #[tokio::test]
    async fn tools_call_invokes_handler() {
        let server = start_with_one_tool().await;
        let body = rpc(
            &server,
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ping","arguments":{}}}),
        )
        .await;
        let content = &body["result"]["content"][0];
        assert_eq!(content["type"], "text");
        assert_eq!(content["text"], "pong");
        server.stop().await;
    }

    #[tokio::test]
    async fn unknown_method_returns_jsonrpc_error() {
        let server = start_with_one_tool().await;
        let body = rpc(&server, json!({"jsonrpc":"2.0","id":4,"method":"who"})).await;
        assert_eq!(body["error"]["code"], -32601);
        server.stop().await;
    }
}
```

- [ ] **Step 2: Add `reqwest` to dev-deps for MCP tests**

In `crates/anyllm-claude-code/Cargo.toml` `[dev-dependencies]`:

```toml
reqwest = { workspace = true }
```

- [ ] **Step 3: Add `mod mcp;` to `lib.rs`**

```rust
mod mcp;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p anyllm-claude-code mcp::tests
```

Expected: All five MCP tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/anyllm-claude-code
git commit -m "feat(claude-code): in-process bearer-auth HTTP MCP server"
```

### Task 12: Bridge anyllm tool callbacks into MCP handlers

The `anyllm::Tool` type carries the schema only — the actual closure that runs when a tool is called is supplied separately by the caller (typically passed alongside via the runtime). For our provider, the caller supplies a `ToolDispatcher` map at chat-time. Most user code today wraps this in their own dispatch layer; we expose a typed handle so users can register `name → handler` pairs.

**Files:**
- Create: `crates/anyllm-claude-code/src/dispatcher.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `pub mod dispatcher;`)

- [ ] **Step 1: Write failing test**

Create `crates/anyllm-claude-code/src/dispatcher.rs`:

```rust
//! Tool-callback registry for the Claude Code provider.
//!
//! `anyllm::Tool` carries only the schema; the closure that runs when a
//! tool is invoked is supplied here. Callers build a [`ToolDispatcher`]
//! once and pass it into [`crate::ChatRequestOptions::with_dispatcher`]
//! (or the per-call API surface — see Task 17).

use std::collections::HashMap;
use std::sync::Arc;

use anyllm::Result;
use futures_core::future::BoxFuture;
use serde_json::Value;

use crate::mcp::ToolHandler;

/// Maps tool name → async handler closure.
#[derive(Clone, Default)]
pub struct ToolDispatcher {
    pub(crate) handlers: HashMap<String, ToolHandler>,
}

impl ToolDispatcher {
    /// Empty dispatcher. Tools without a registered handler error inside
    /// the MCP server with a JSON-RPC InvalidRequest.
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Register a handler for a tool by name. The handler receives the
    /// JSON arguments as Claude provided them and returns the textual
    /// result that gets streamed back as the tool_result content.
    #[must_use]
    pub fn register<F, Fut>(mut self, name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<String>> + Send + 'static,
    {
        let handler: ToolHandler = Arc::new(move |args| {
            let fut = handler(args);
            Box::pin(fut) as BoxFuture<'static, Result<String>>
        });
        self.handlers.insert(name.into(), handler);
        self
    }

    /// Number of registered handlers.
    #[must_use]
    pub fn len(&self) -> usize { self.handlers.len() }

    /// Whether no handlers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.handlers.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn registers_and_invokes_handler() {
        let d = ToolDispatcher::new().register("hello", |_args| async move {
            Ok::<_, anyllm::Error>("world".to_string())
        });
        let h = d.handlers.get("hello").unwrap().clone();
        let result = h(serde_json::json!({})).await.unwrap();
        assert_eq!(result, "world");
    }

    #[tokio::test]
    async fn handlers_can_use_arguments() {
        let d = ToolDispatcher::new().register("echo", |args| async move {
            Ok::<_, anyllm::Error>(args.get("msg").and_then(|v| v.as_str()).unwrap_or("").to_string())
        });
        let h = d.handlers.get("echo").unwrap().clone();
        let result = h(serde_json::json!({"msg": "hi"})).await.unwrap();
        assert_eq!(result, "hi");
    }
}
```

- [ ] **Step 2: Add module to `lib.rs`**

```rust
pub mod dispatcher;
pub use dispatcher::ToolDispatcher;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p anyllm-claude-code dispatcher
```

Expected: Both tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/anyllm-claude-code/src
git commit -m "feat(claude-code): ToolDispatcher for registering tool callbacks"
```

---

## Phase 7 — Per-call orchestration

### Task 13: Scratch-dir + bearer-token primitives

**Files:**
- Create: `crates/anyllm-claude-code/src/runtime.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod runtime;`)

- [ ] **Step 1: Write failing tests**

Create `crates/anyllm-claude-code/src/runtime.rs`:

```rust
//! Per-call runtime primitives: scratch dir, bearer token, cleanup guard.

use std::path::{Path, PathBuf};
use anyllm::Result;
use rand::RngCore;

/// Owns a per-call scratch directory; deletes it on drop.
pub(crate) struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    /// Create a fresh scratch dir under `std::env::temp_dir()` with the
    /// `tmp/`, `plugins/`, `debug/`, and `home/` subdirs pre-created.
    pub(crate) fn create() -> Result<Self> {
        let mut path = std::env::temp_dir();
        let suffix: u64 = rand::thread_rng().next_u64();
        path.push(format!("anyllm-claude-{:016x}", suffix));
        std::fs::create_dir_all(path.join("tmp")).map_err(io_to_provider)?;
        std::fs::create_dir_all(path.join("plugins")).map_err(io_to_provider)?;
        std::fs::create_dir_all(path.join("debug")).map_err(io_to_provider)?;
        std::fs::create_dir_all(path.join("home")).map_err(io_to_provider)?;
        Ok(Self { path })
    }

    pub(crate) fn root(&self) -> &Path { &self.path }
    pub(crate) fn fake_home(&self) -> PathBuf { self.path.join("home") }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn io_to_provider(e: std::io::Error) -> anyllm::Error {
    anyllm::Error::Provider {
        status: None,
        message: format!("scratch dir setup failed: {e}"),
        body: None,
        request_id: None,
    }
}

/// Generate a 256-bit hex-encoded bearer token using the system RNG.
pub(crate) fn mint_bearer_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_dir_creates_subdirs_and_deletes_on_drop() {
        let path;
        {
            let s = ScratchDir::create().unwrap();
            path = s.root().to_path_buf();
            assert!(path.join("tmp").is_dir());
            assert!(path.join("plugins").is_dir());
            assert!(path.join("debug").is_dir());
            assert!(path.join("home").is_dir());
        }
        assert!(!path.exists(), "scratch dir should be cleaned on drop");
    }

    #[test]
    fn mint_bearer_token_is_64_hex_chars() {
        let t = mint_bearer_token();
        assert_eq!(t.len(), 64);
        assert!(t.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn mint_bearer_token_is_unique() {
        let a = mint_bearer_token();
        let b = mint_bearer_token();
        assert_ne!(a, b);
    }
}
```

- [ ] **Step 2: Add module to `lib.rs`**

```rust
mod runtime;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p anyllm-claude-code runtime
```

Expected: All three tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/anyllm-claude-code/src
git commit -m "feat(claude-code): scratch dir + bearer-token primitives"
```

### Task 14: Per-call orchestration: spawn, drain, cleanup

**Files:**
- Create: `crates/anyllm-claude-code/src/chat.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod chat;`)

This task wires the pieces together but does not yet expose `Provider`. The orchestration function is:

```rust
async fn execute_chat(
    config: &ProviderConfig,
    request: &ChatRequest,
    dispatcher: &ToolDispatcher,
) -> Result<(impl Stream<Item = Result<OutputEvent>>, CleanupGuard)>
```

It returns the parsed event stream plus a cleanup guard that owns the subprocess and MCP server.

- [ ] **Step 1: Write failing integration test using the mock subprocess**

(The mock subprocess is built in Task 15. For now write the test against a `claude` binary that is *expected* to be a fake echo script — we'll replace it with the proper mock harness in Task 15 and re-run.)

Create `crates/anyllm-claude-code/src/chat.rs`:

```rust
//! Per-call orchestration: spawn the claude subprocess, pipe stream-json
//! input, drain stream-json output, run cleanup on every exit path.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyllm::{ChatRequest, ChatResponse, Error, Result, StreamEvent};
use futures_core::Stream;
use futures_util::StreamExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Child;
use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::dispatcher::ToolDispatcher;
use crate::mcp::McpServer;
use crate::render;
use crate::runtime::{mint_bearer_token, ScratchDir};
use crate::sandbox::{Sandbox, SandboxPaths, SpawnSpec};
use crate::streaming::{collect_into_response, parse_ndjson, StreamEventMapper};
use crate::wire::OutputEvent;

/// Internal config bundling the parts an `execute_chat` call needs.
#[derive(Clone)]
pub(crate) struct ProviderConfig {
    pub claude_path: PathBuf,
    pub oauth_token: String,
    pub sandbox: Arc<dyn Sandbox>,
    pub request_timeout: Duration,
    pub stream_idle_timeout: Duration,
}

/// Cleanup guard: owns the subprocess and MCP server; tearing down on Drop.
pub(crate) struct CallGuard {
    child: Option<Child>,
    mcp: Option<McpServer>,
    _scratch: ScratchDir,
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Best-effort: try graceful, then kill.
            #[cfg(unix)]
            {
                if let Some(pid) = child.id() {
                    let _ = nix::sys::signal::kill(
                        nix::unistd::Pid::from_raw(pid as i32),
                        nix::sys::signal::Signal::SIGTERM,
                    );
                }
            }
            // Don't block long in Drop; force-kill after a short wait.
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .expect("cleanup runtime");
                rt.block_on(async {
                    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                });
            });
        }
        if let Some(mcp) = self.mcp.take() {
            // McpServer::Drop fires graceful shutdown; we just let it go.
            drop(mcp);
        }
        // _scratch deleted by its own Drop after this.
    }
}

/// Orchestrate a single chat call.
///
/// Returns a stream of parsed [`OutputEvent`]s plus the [`CallGuard`]
/// that owns the live subprocess and MCP server. Caller decides whether
/// to collect the stream into a [`ChatResponse`] or surface it as a
/// streaming response.
pub(crate) async fn execute_chat(
    config: &ProviderConfig,
    request: &ChatRequest,
    dispatcher: &ToolDispatcher,
) -> Result<(
    impl Stream<Item = Result<OutputEvent>>,
    CallGuard,
)> {
    // Refuse non-text response formats up front.
    if let Some(fmt) = request.response_format.as_ref() {
        if !matches!(fmt, anyllm::ResponseFormat::Text) {
            return Err(Error::Unsupported(
                "claude-code provider does not support non-text response_format natively; \
                 enable the `extract` feature on anyllm and use ExtractingProvider".into(),
            ));
        }
    }

    let scratch = ScratchDir::create()?;
    let token = mint_bearer_token();

    // Build per-call MCP handler map: only tools registered in dispatcher.
    let mcp_tool_defs: Vec<anyllm::Tool> = request.tools.iter().cloned().collect();
    let mcp = McpServer::start(mcp_tool_defs.clone(), dispatcher.handlers.clone(), token.clone()).await?;

    // Render argv + env.
    let mcp_endpoint = render::McpEndpoint {
        url: &format!("http://{}/mcp", mcp.addr),
        bearer_token: &token,
    };
    let mut argv = render::build_argv(request, &mcp_endpoint);

    // Append tool-choice nudge into the system prompt argument if needed.
    if let Some(idx) = argv.iter().position(|s| s == "--system-prompt") {
        if let Some(value) = argv.get_mut(idx + 1) {
            let mut s = value.to_string_lossy().into_owned();
            render::append_tool_choice_nudge(&mut s, &request.tool_choice);
            *value = s.into();
        }
    } else if matches!(
        request.tool_choice,
        anyllm::ToolChoice::Required | anyllm::ToolChoice::Specific { .. }
    ) {
        // No system prompt set yet — inject a nudge-only one.
        let mut s = String::new();
        render::append_tool_choice_nudge(&mut s, &request.tool_choice);
        argv.push("--system-prompt".into());
        argv.push(s.into());
    }

    let env = render::build_env(
        &config.oauth_token,
        scratch.root(),
        &scratch.fake_home(),
        request.reasoning.as_ref(),
    );

    let spec = SpawnSpec {
        program: config.claude_path.clone(),
        args: argv,
        env,
        paths: SandboxPaths {
            scratch_dir: scratch.root().to_path_buf(),
            fake_home: scratch.fake_home(),
        },
    };

    let mut cmd = config.sandbox.build_command(spec)?;
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| Error::Provider {
        status: None,
        message: format!("failed to spawn claude: {e}"),
        body: None,
        request_id: None,
    })?;

    // Pipe rendered stream-json input to stdin.
    let input_lines = render::render_messages(request)?;
    let mut stdin = child.stdin.take().ok_or_else(|| Error::Provider {
        status: None,
        message: "failed to acquire claude stdin".into(),
        body: None,
        request_id: None,
    })?;
    tokio::spawn(async move {
        for evt in input_lines {
            let json = match serde_json::to_string(&evt) {
                Ok(s) => s,
                Err(_) => break,
            };
            if stdin.write_all(json.as_bytes()).await.is_err() { break; }
            if stdin.write_all(b"\n").await.is_err() { break; }
        }
        let _ = stdin.shutdown().await;
    });

    // Capture stderr into a bounded ring buffer (last 4 KiB).
    let stderr_capture = Arc::new(tokio::sync::Mutex::new(String::new()));
    if let Some(mut err) = child.stderr.take() {
        let cap = stderr_capture.clone();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 4096];
            loop {
                use tokio::io::AsyncReadExt;
                match err.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let mut g = cap.lock().await;
                        g.push_str(&String::from_utf8_lossy(&buf[..n]));
                        if g.len() > 4096 {
                            let len = g.len();
                            *g = g[(len - 4096)..].to_string();
                        }
                    }
                }
            }
        });
    }

    let stdout = child.stdout.take().ok_or_else(|| Error::Provider {
        status: None,
        message: "failed to acquire claude stdout".into(),
        body: None,
        request_id: None,
    })?;

    let byte_stream = tokio_util::io::ReaderStream::new(stdout);
    let event_stream = parse_ndjson(byte_stream);

    let guard = CallGuard { child: Some(child), mcp: Some(mcp), _scratch: scratch };

    Ok((event_stream, guard))
}
```

- [ ] **Step 2: Add `nix` and `tokio-util` to dependencies**

In `crates/anyllm-claude-code/Cargo.toml` `[dependencies]`:

```toml
tokio-util = { version = "0.7", features = ["io"] }
```

And conditionally:

```toml
[target.'cfg(unix)'.dependencies]
nix = { version = "0.29", features = ["signal"] }
```

- [ ] **Step 3: Add `mod chat;` to `lib.rs`**

```rust
mod chat;
```

- [ ] **Step 4: Verify it compiles (no integration test yet — that comes with the mock harness)**

```bash
cargo build -p anyllm-claude-code
```

Expected: Compiles with no warnings. The orchestration is exercised end-to-end in Task 16 once the mock harness exists.

- [ ] **Step 5: Commit**

```bash
git add crates/anyllm-claude-code Cargo.lock
git commit -m "feat(claude-code): per-call orchestration with cleanup guard"
```

### Task 15: Mock subprocess harness (feature-gated)

**Files:**
- Create: `crates/anyllm-claude-code/src/mock.rs`
- Create: `crates/anyllm-claude-code/tests/fake_claude.rs` — a binary built only in tests, plays back scripted stream-json
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `#[cfg(any(test, feature = "mock"))] mod mock;`)
- Modify: `crates/anyllm-claude-code/Cargo.toml` (add `[[bin]]` for fake_claude with `required-features = ["mock"]`-equivalent gate)

Strategy: instead of mocking at the Rust API boundary, we ship a tiny `fake_claude` binary in this crate's `tests/` dir that reads stream-json input and emits scripted stream-json output based on environment variables. Tests then point `claude_path(...)` at the built binary. This exercises the *real* spawn + drain code paths.

- [ ] **Step 1: Create the fake-claude binary**

Create `crates/anyllm-claude-code/tests/fake_claude.rs`:

```rust
//! Fake `claude` binary for integration tests.
//!
//! Reads stream-json input on stdin, emits a scripted set of stream-json
//! events on stdout based on environment variables:
//!
//!   FAKE_CLAUDE_SCRIPT  Path to a file containing newline-delimited
//!                        stream-json events to emit verbatim. If absent,
//!                        emits a one-line "hi" assistant + success result.
//!   FAKE_CLAUDE_EXIT    Exit code after emitting the script (default 0).
//!   FAKE_CLAUDE_STDERR  String to write to stderr before exit.

use std::io::{Read, Write};

fn main() {
    // Drain stdin so claude looks like it consumed input.
    let mut buf = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut buf);

    let script_path = std::env::var("FAKE_CLAUDE_SCRIPT").ok();
    let mut stdout = std::io::stdout();
    if let Some(path) = script_path {
        let contents = std::fs::read_to_string(&path).expect("read FAKE_CLAUDE_SCRIPT file");
        let _ = stdout.write_all(contents.as_bytes());
        if !contents.ends_with('\n') {
            let _ = stdout.write_all(b"\n");
        }
    } else {
        let lines = [
            r#"{"type":"assistant","message":{"id":"msg_fake","model":"fake-model","content":[{"type":"text","text":"hi"}],"stop_reason":null}}"#,
            r#"{"type":"result","subtype":"success","session_id":"sess_fake","is_error":false,"usage":{"input_tokens":1,"output_tokens":1}}"#,
        ];
        for line in lines { let _ = writeln!(stdout, "{line}"); }
    }
    let _ = stdout.flush();

    if let Ok(s) = std::env::var("FAKE_CLAUDE_STDERR") {
        let _ = writeln!(std::io::stderr(), "{s}");
    }
    let code: i32 = std::env::var("FAKE_CLAUDE_EXIT").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    std::process::exit(code);
}
```

- [ ] **Step 2: Wire it as a test binary in Cargo.toml**

In `crates/anyllm-claude-code/Cargo.toml`, append:

```toml
[[bin]]
name = "fake_claude"
path = "tests/fake_claude.rs"
test = false
bench = false
```

- [ ] **Step 3: Create a thin helper for tests**

Create `crates/anyllm-claude-code/src/mock.rs`:

```rust
//! Helpers for tests and examples that drive the provider against a
//! fake claude binary instead of the real CLI.

use std::path::PathBuf;

/// Path to the `fake_claude` binary built by this crate's tests.
///
/// Looks under `target/<profile>/fake_claude(.exe)` relative to the
/// workspace root. Asserts the binary exists; if it does not, run
/// `cargo build -p anyllm-claude-code --bin fake_claude` first.
pub fn fake_claude_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .ancestors()
        .nth(2) // crates/anyllm-claude-code -> crates -> workspace
        .expect("workspace root above crate dir")
        .to_path_buf();
    let target = workspace_root.join("target");
    for profile in ["debug", "release"] {
        let bin = if cfg!(windows) { "fake_claude.exe" } else { "fake_claude" };
        let candidate = target.join(profile).join(bin);
        if candidate.is_file() { return candidate; }
    }
    panic!("fake_claude binary not built; run `cargo build --bin fake_claude` first");
}
```

- [ ] **Step 4: Add module to `lib.rs`**

```rust
#[cfg(any(test, feature = "mock"))]
pub mod mock;
```

- [ ] **Step 5: Verify the binary builds**

```bash
cargo build -p anyllm-claude-code --bin fake_claude
ls target/debug/fake_claude
```

Expected: binary exists.

- [ ] **Step 6: Commit**

```bash
git add crates/anyllm-claude-code
git commit -m "feat(claude-code): fake_claude test binary + mock helper"
```

---

## Phase 8 — Provider, builder, capabilities

### Task 16: Provider, ProviderBuilder, ChatProvider impl

**Files:**
- Modify: `crates/anyllm-claude-code/src/lib.rs` — add the public surface
- Modify: `crates/anyllm-claude-code/src/chat.rs` — add `chat()` and `chat_stream()` flows wrapping `execute_chat`

- [ ] **Step 1: Write failing integration test against `fake_claude`**

Create `crates/anyllm-claude-code/tests/integration.rs`:

```rust
use std::path::PathBuf;
use std::time::Duration;

use anyllm::prelude::*;
use anyllm_claude_code::{Provider, ToolDispatcher};

fn fake_claude_path() -> PathBuf {
    anyllm_claude_code::mock::fake_claude_path()
}

#[tokio::test]
async fn chat_returns_assembled_response_from_fake_claude() {
    let provider = Provider::builder()
        .oauth_token("dummy-token")
        .claude_path(fake_claude_path())
        .request_timeout(Duration::from_secs(10))
        .build()
        .unwrap();

    let response = provider
        .chat(&ChatRequest::new("fake-model").user("hello"))
        .await
        .unwrap();

    assert_eq!(response.text(), Some("hi".into()));
    assert_eq!(response.id.as_deref(), Some("msg_fake"));
    assert_eq!(response.usage.unwrap().output_tokens, Some(1));
}

#[tokio::test]
async fn chat_stream_yields_events_from_fake_claude() {
    let provider = Provider::builder()
        .oauth_token("dummy-token")
        .claude_path(fake_claude_path())
        .build()
        .unwrap();

    let mut stream = provider
        .chat_stream(&ChatRequest::new("fake-model").user("hello"))
        .await
        .unwrap();

    let mut saw_response_start = false;
    let mut saw_response_stop = false;
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::ResponseStart { .. } => saw_response_start = true,
            StreamEvent::TextDelta { text: delta, .. } => text.push_str(&delta),
            StreamEvent::ResponseStop => saw_response_stop = true,
            _ => {}
        }
    }
    assert!(saw_response_start);
    assert!(saw_response_stop);
    assert_eq!(text, "hi");
}

#[tokio::test]
async fn rejects_unsupported_response_format() {
    let provider = Provider::builder()
        .oauth_token("dummy-token")
        .claude_path(fake_claude_path())
        .build()
        .unwrap();

    let req = ChatRequest::new("fake-model")
        .user("give me JSON")
        .response_format(ResponseFormat::JsonObject);

    let err = provider.chat(&req).await.unwrap_err();
    assert!(matches!(err, anyllm::Error::Unsupported(_)));
}
```

- [ ] **Step 2: Implement Provider + ProviderBuilder in `lib.rs`**

Replace the contents of `crates/anyllm-claude-code/src/lib.rs` with:

```rust
#![warn(missing_docs)]
//! Claude Code CLI provider for `anyllm`.
//!
//! Wraps the `claude` CLI as a regular [`anyllm::ChatProvider`], allowing
//! callers to drive their Claude Code subscription through the portable
//! `anyllm` interface.
//!
//! ```no_run
//! # async fn example() -> anyllm::Result<()> {
//! use anyllm::prelude::*;
//! use anyllm_claude_code::Provider;
//!
//! let provider = Provider::from_env()?;
//! let response = provider
//!     .chat(&ChatRequest::new("claude-sonnet-4-6").user("Say hello."))
//!     .await?;
//! println!("{}", response.text_or_empty());
//! # Ok(()) }
//! ```
//!
//! See the [design spec](https://github.com/sagikazarmark/anyllm/blob/main/docs/superpowers/specs/2026-05-03-claude-code-provider-design.md)
//! for the architecture and capability matrix.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyllm::{
    CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatRequest,
    ChatResponse, ChatStream, Error, ProviderIdentity, Result, StreamEvent,
};
use futures_core::Stream;

mod chat;
pub mod dispatcher;
mod error;
mod mcp;
mod render;
mod runtime;
pub mod sandbox;
mod streaming;
mod wire;

#[cfg(any(test, feature = "mock"))]
pub mod mock;

pub use dispatcher::ToolDispatcher;
pub use sandbox::{NoSandbox, Sandbox, SandboxPaths, SpawnSpec};

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Claude Code CLI provider implementing [`anyllm::ChatProvider`].
///
/// Clone is cheap: internals are wrapped in `Arc`.
#[derive(Clone)]
pub struct Provider {
    inner: Arc<Inner>,
}

struct Inner {
    config: chat::ProviderConfig,
    dispatcher: ToolDispatcher,
    capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
}

impl Provider {
    /// Create with just an OAuth token; resolves `claude` from PATH.
    pub fn new(oauth_token: impl Into<String>) -> Result<Self> {
        Self::builder().oauth_token(oauth_token).build()
    }

    /// Create from environment: requires `CLAUDE_CODE_OAUTH_TOKEN`.
    pub fn from_env() -> Result<Self> {
        let token = std::env::var("CLAUDE_CODE_OAUTH_TOKEN").map_err(|_| {
            Error::Auth("CLAUDE_CODE_OAUTH_TOKEN not set".into())
        })?;
        if token.trim().is_empty() {
            return Err(Error::Auth("CLAUDE_CODE_OAUTH_TOKEN not set".into()));
        }
        Self::new(token)
    }

    /// Builder for full configuration.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder::default()
    }

    /// Install a [`ToolDispatcher`] for this Provider.
    ///
    /// Tools defined in `ChatRequest::tools` will be invoked through the
    /// MCP server using these handlers. A tool with no registered handler
    /// returns an error to Claude.
    #[must_use]
    pub fn with_tools(self, dispatcher: ToolDispatcher) -> Self {
        Self {
            inner: Arc::new(Inner {
                config: self.inner.config.clone(),
                dispatcher,
                capability_resolver: self.inner.capability_resolver.clone(),
            }),
        }
    }

    /// Install a capability resolver consulted before the provider's
    /// built-in capability matrix.
    #[must_use]
    pub fn with_chat_capabilities(self, resolver: impl ChatCapabilityResolver) -> Self {
        Self {
            inner: Arc::new(Inner {
                config: self.inner.config.clone(),
                dispatcher: self.inner.dispatcher.clone(),
                capability_resolver: Some(Arc::new(resolver)),
            }),
        }
    }
}

/// Builder for [`Provider`].
#[derive(Default)]
pub struct ProviderBuilder {
    oauth_token: Option<String>,
    claude_path: Option<PathBuf>,
    sandbox: Option<Arc<dyn Sandbox>>,
    request_timeout: Option<Duration>,
    stream_idle_timeout: Option<Duration>,
    dispatcher: Option<ToolDispatcher>,
}

impl ProviderBuilder {
    #[must_use]
    pub fn oauth_token(mut self, token: impl Into<String>) -> Self {
        self.oauth_token = Some(token.into()); self
    }
    #[must_use]
    pub fn claude_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.claude_path = Some(path.into()); self
    }
    #[must_use]
    pub fn sandbox(mut self, sandbox: impl Sandbox + 'static) -> Self {
        self.sandbox = Some(Arc::new(sandbox)); self
    }
    #[must_use]
    pub fn request_timeout(mut self, d: Duration) -> Self {
        self.request_timeout = Some(d); self
    }
    #[must_use]
    pub fn stream_idle_timeout(mut self, d: Duration) -> Self {
        self.stream_idle_timeout = Some(d); self
    }
    #[must_use]
    pub fn tools(mut self, dispatcher: ToolDispatcher) -> Self {
        self.dispatcher = Some(dispatcher); self
    }

    /// Finalize the builder.
    pub fn build(self) -> Result<Provider> {
        let oauth_token = match self.oauth_token {
            Some(s) if !s.trim().is_empty() => s,
            _ => return Err(Error::InvalidRequest("oauth_token is required".into())),
        };
        let claude_path = match self.claude_path {
            Some(p) => p,
            None => resolve_claude_path()?,
        };
        let sandbox: Arc<dyn Sandbox> = self.sandbox.unwrap_or_else(|| Arc::new(NoSandbox));

        Ok(Provider {
            inner: Arc::new(Inner {
                config: chat::ProviderConfig {
                    claude_path,
                    oauth_token,
                    sandbox,
                    request_timeout: self.request_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT),
                    stream_idle_timeout: self.stream_idle_timeout.unwrap_or(DEFAULT_STREAM_IDLE_TIMEOUT),
                },
                dispatcher: self.dispatcher.unwrap_or_default(),
                capability_resolver: None,
            }),
        })
    }
}

fn resolve_claude_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("CLAUDE_CODE_BIN") {
        if !p.trim().is_empty() {
            return Ok(PathBuf::from(p));
        }
    }
    which("claude").ok_or_else(|| {
        Error::InvalidRequest(
            "claude binary not found on PATH; set CLAUDE_CODE_BIN or builder.claude_path(...)".into(),
        )
    })
}

fn which(program: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(program);
        if candidate.is_file() { return Some(candidate); }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{program}.exe"));
            if exe.is_file() { return Some(exe); }
        }
    }
    None
}

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &'static str { "claude-code" }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        chat::run_chat(&self.inner.config, request, &self.inner.dispatcher).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        chat::run_chat_stream(&self.inner.config, request, &self.inner.dispatcher).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        if let Some(r) = &self.inner.capability_resolver {
            if let Some(answer) = r.chat_capability(model, capability) {
                return answer;
            }
        }
        builtin_capability(capability)
    }
}

fn builtin_capability(capability: ChatCapability) -> CapabilitySupport {
    use ChatCapability::*;
    match capability {
        ToolCalls
        | ParallelToolCalls
        | Streaming
        | NativeStreaming
        | ImageInput
        | ReasoningOutput
        | ReasoningReplay
        | ReasoningConfig => CapabilitySupport::Supported,
        StructuredOutput
        | ImageDetail
        | ImageOutput
        | ImageReplay => CapabilitySupport::Unsupported,
        _ => CapabilitySupport::Unknown,
    }
}
```

- [ ] **Step 3: Implement `run_chat` and `run_chat_stream` in `chat.rs`**

Append to `crates/anyllm-claude-code/src/chat.rs`:

```rust
// ---------- Public-facing entry points used by Provider ----------

/// Non-streaming `chat()` flow: spawn → drain → collect into `ChatResponse`.
pub(crate) async fn run_chat(
    config: &ProviderConfig,
    request: &ChatRequest,
    dispatcher: &ToolDispatcher,
) -> Result<ChatResponse> {
    let total = config.request_timeout;
    let idle = config.stream_idle_timeout;

    let chat_future = async {
        let (events, _guard) = execute_chat(config, request, dispatcher).await?;
        // Inject idle-timeout per event boundary.
        let events = idle_timeout_stream(events, idle);
        collect_into_response(events).await
    };

    timeout(total, chat_future)
        .await
        .map_err(|_| Error::Timeout(format!("chat exceeded request_timeout {total:?}")))?
}

/// Streaming `chat_stream()` flow: spawn → return mapped event stream.
pub(crate) async fn run_chat_stream(
    config: &ProviderConfig,
    request: &ChatRequest,
    dispatcher: &ToolDispatcher,
) -> Result<ChatStream> {
    let idle = config.stream_idle_timeout;
    let (events, guard) = execute_chat(config, request, dispatcher).await?;
    let events = idle_timeout_stream(events, idle);

    // Hold the guard alive for the lifetime of the stream by attaching it
    // via `chain` of an empty stream — but we actually want to drop it at
    // end. Easiest: move the guard into the stream's state via `unfold`.
    let mut mapper = StreamEventMapper::new();
    let stream = futures_util::stream::unfold(
        (Box::pin(events), Vec::<StreamEvent>::new(), Some(guard), mapper),
        |(mut events, mut buffered, guard, mut mapper)| async move {
            loop {
                if let Some(evt) = buffered.pop() {
                    return Some((Ok(evt), (events, buffered, guard, mapper)));
                }
                match events.as_mut().next().await {
                    Some(Ok(out_evt)) => {
                        let mapped = mapper.map(out_evt);
                        // Push in reverse so `pop()` returns in order.
                        buffered = mapped.into_iter().rev().collect();
                    }
                    Some(Err(e)) => return Some((Err(e), (events, buffered, guard, mapper))),
                    None => {
                        // Drop guard explicitly here.
                        drop(guard);
                        return None;
                    }
                }
            }
        },
    );

    Ok(Box::pin(stream))
}

/// Wrap a stream of events so that going `idle` between items yields a
/// [`Error::Timeout`] item.
fn idle_timeout_stream<S>(stream: S, idle: Duration) -> impl Stream<Item = Result<OutputEvent>>
where
    S: Stream<Item = Result<OutputEvent>> + Send + 'static,
{
    use futures_util::stream;
    stream::unfold(Box::pin(stream), move |mut s| async move {
        match tokio::time::timeout(idle, s.as_mut().next()).await {
            Ok(Some(item)) => Some((item, s)),
            Ok(None) => None,
            Err(_) => Some((
                Err(Error::Timeout(format!(
                    "no stream-json event within stream_idle_timeout {idle:?}"
                ))),
                s,
            )),
        }
    })
}
```

- [ ] **Step 4: Build the fake_claude bin first, then run the integration test**

```bash
cargo build -p anyllm-claude-code --bin fake_claude
cargo test -p anyllm-claude-code --test integration
```

Expected: All three integration tests pass.

- [ ] **Step 5: Run the full test suite**

```bash
cargo test -p anyllm-claude-code
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/anyllm-claude-code Cargo.lock
git commit -m "feat(claude-code): Provider, ProviderBuilder, ChatProvider impl"
```

### Task 17: Provider-specific request options

**Files:**
- Create: `crates/anyllm-claude-code/src/options.rs`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `mod options;` + re-export)

- [ ] **Step 1: Define typed extras**

Create `crates/anyllm-claude-code/src/options.rs`:

```rust
//! Provider-specific request options for the Claude Code provider.
//!
//! These extend [`anyllm::ChatRequest`] via [`ChatRequest::with_option`].
//! Anything that does not map cleanly onto the portable surface lives
//! here, typed.

use serde::{Deserialize, Serialize};

/// Options that flow through to the spawned `claude` invocation but do
/// not have a portable analogue in [`anyllm::ChatRequest`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatRequestOptions {
    /// Override Claude's per-call max-turns budget. Maps to `--max-turns`.
    /// Useful when callers want to cap tool-loop length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,

    /// Pass extra `--allowed-tools` entries (e.g., a specific built-in
    /// tool the caller wants to re-enable). Names are passed verbatim.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_tools: Vec<String>,
}
```

- [ ] **Step 2: Re-export and read in render**

In `lib.rs`:

```rust
mod options;
pub use options::ChatRequestOptions;
```

In `render.rs` `build_argv`, after the `--disallowed-tools` pair, read the option and append `--max-turns` / `--allowed-tools` if present:

```rust
if let Some(opts) = req.option::<ChatRequestOptions>() {
    if let Some(t) = opts.max_turns {
        argv.push("--max-turns".into());
        argv.push(t.to_string().into());
    }
    if !opts.allowed_tools.is_empty() {
        argv.push("--allowed-tools".into());
        argv.push(opts.allowed_tools.join(",").into());
    }
}
```

(Add `use crate::ChatRequestOptions;` to `render.rs`.)

- [ ] **Step 3: Add a test for the options pass-through**

Append to `crates/anyllm-claude-code/src/render.rs` `argv_tests` mod:

```rust
#[test]
fn argv_includes_max_turns_when_option_set() {
    let req = ChatRequest::new("claude-sonnet-4-6")
        .user("hi")
        .with_option(crate::ChatRequestOptions { max_turns: Some(3), allowed_tools: vec![] });
    let argv = build_argv(&req, &ep());
    let pos = argv.iter().position(|s| s == "--max-turns").unwrap();
    assert_eq!(argv[pos + 1].to_string_lossy(), "3");
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p anyllm-claude-code
```

Expected: All tests still pass plus the new one.

- [ ] **Step 5: Commit**

```bash
git add crates/anyllm-claude-code/src
git commit -m "feat(claude-code): typed ChatRequestOptions for provider extras"
```

### Task 18: Tool round-trip integration test (real subprocess + MCP server)

**Files:**
- Modify: `crates/anyllm-claude-code/tests/integration.rs`
- Create: `crates/anyllm-claude-code/tests/scripts/tool_call.jsonl`

A scripted run that has the fake claude emit a tool_use event, expects the test harness to invoke the registered handler via MCP — but since fake_claude doesn't actually call MCP back, we instead test the path where Claude's *previous turn* contains a tool_call and the next turn is the assistant interpreting it. This still exercises the renderer end-to-end. To exercise the live MCP path, we add a separate test that spawns the MCP server directly and POSTs a `tools/call` request.

- [ ] **Step 1: Test that the MCP server is reachable end-to-end during a chat call**

Append to `crates/anyllm-claude-code/tests/integration.rs`:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[tokio::test]
async fn dispatcher_handler_runs_when_invoked_through_mcp() {
    // We script fake_claude to do nothing interesting, but inside the
    // chat call we manually POST to the per-call MCP server to verify
    // the handler runs. To get the MCP url, we instead exercise the
    // McpServer directly with the same dispatcher to prove wiring.
    use anyllm_claude_code::mcp;
    use anyllm_claude_code::dispatcher::ToolDispatcher;

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();
    let dispatcher = ToolDispatcher::new().register("ping", move |_| {
        let c = counter_clone.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            Ok::<_, anyllm::Error>("pong".to_string())
        }
    });

    let tools = vec![Tool::new(
        "ping",
        serde_json::json!({"type":"object","properties":{}}),
    )];
    let server =
        mcp::McpServer::start(tools, dispatcher.handlers.clone(), "tok".into())
            .await
            .unwrap();
    let url = format!("http://{}/mcp", server.addr);

    let resp: serde_json::Value = reqwest::Client::new()
        .post(&url)
        .header("Authorization", "Bearer tok")
        .json(&serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "ping", "arguments": {}}
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp["result"]["content"][0]["text"], "pong");
    assert_eq!(counter.load(Ordering::SeqCst), 1);
    server.stop().await;
}
```

(For this to compile we need to make the `mcp` module pub-crate-visible to `tests/`. Add `pub` to `mod mcp;` in lib.rs only if necessary; alternatively add an internal API on `Provider` to expose the server for testing. Simpler: re-export under `pub mod __test_only` gated on `#[cfg(any(test, feature = "mock"))]`. Add to lib.rs:

```rust
#[cfg(any(test, feature = "mock"))]
#[doc(hidden)]
pub mod __testing {
    pub use crate::mcp;
}
```

Update the test to use `anyllm_claude_code::__testing::mcp;`.)

- [ ] **Step 2: Run tests**

```bash
cargo test -p anyllm-claude-code --test integration
```

Expected: New test plus the earlier integration tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/anyllm-claude-code
git commit -m "test(claude-code): integration test for MCP tool dispatch"
```

---

## Phase 9 — Conformance & polish

### Task 19: Wire up `anyllm-conformance` fixtures

**Files:**
- Create: `crates/anyllm-claude-code/src/conformance_tests.rs`
- Create: `crates/anyllm-claude-code/fixtures/request.json`
- Create: `crates/anyllm-claude-code/fixtures/response_raw.jsonl`
- Create: `crates/anyllm-claude-code/fixtures/response_expected.json`
- Create: `crates/anyllm-claude-code/fixtures/stream.jsonl`
- Create: `crates/anyllm-claude-code/fixtures/stream_events.json`
- Create: `crates/anyllm-claude-code/fixtures/stream_response_expected.json`
- Modify: `crates/anyllm-claude-code/src/lib.rs` (add `#[cfg(test)] mod conformance_tests;`)

Pattern follows `crates/anyllm-anthropic/src/conformance_tests.rs`. Tests cover:
- Render `ChatRequest` → stream-json input lines, compare to `request.json`.
- Parse a recorded stream-json output (`response_raw.jsonl`) → `ChatResponse`, compare to `response_expected.json`.
- Parse the same lines as a live stream → `Vec<StreamEvent>`, compare to `stream_events.json`; collected → `stream_response_expected.json`.

- [ ] **Step 1: Build the request-render + response-parse fixtures**

Create `crates/anyllm-claude-code/fixtures/request.json` (the rendered input lines as a JSON array, one element per stream-json input event):

```json
[
  {
    "type": "user",
    "message": {
      "role": "user",
      "content": [{ "type": "text", "text": "Find the answer" }]
    }
  }
]
```

Create `crates/anyllm-claude-code/fixtures/response_raw.jsonl`:

```
{"type":"assistant","message":{"id":"msg_01","model":"claude-sonnet-4-6","content":[{"type":"text","text":"4"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":1}}}
{"type":"result","subtype":"success","session_id":"sess_01","is_error":false,"duration_ms":420,"num_turns":1,"usage":{"input_tokens":12,"output_tokens":1},"result":"4"}
```

Create `crates/anyllm-claude-code/fixtures/response_expected.json`:

```json
{
  "content": [{ "type": "text", "text": "4" }],
  "finish_reason": "Stop",
  "usage": { "input_tokens": 12, "output_tokens": 1 },
  "model": "claude-sonnet-4-6",
  "id": "msg_01",
  "metadata": {}
}
```

(Adjust the exact serialized shape to match what `anyllm_conformance::assert_response_fixture_eq` expects — read `crates/anyllm-conformance/src/lib.rs` if any field differs.)

- [ ] **Step 2: Stream fixtures**

Create `crates/anyllm-claude-code/fixtures/stream.jsonl` — same content as `response_raw.jsonl`. The conformance helper expects whatever framing `streaming_from_lines` accepts; we'll add a small helper.

- [ ] **Step 3: Conformance test module**

Create `crates/anyllm-claude-code/src/conformance_tests.rs`:

```rust
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyllm::{ChatRequest};
    use anyllm_conformance::{
        FixtureDir, assert_json_fixture_eq, assert_response_fixture_eq,
        assert_stream_fixture_eq, load_text_fixture,
    };

    fn fixtures() -> FixtureDir {
        FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
    }

    #[test]
    fn request_fixture_matches() {
        let fixtures = fixtures();
        let request = ChatRequest::new("claude-sonnet-4-6").user("Find the answer");
        let actual = crate::render::render_messages(&request).unwrap();
        assert_json_fixture_eq(&actual, &fixtures, "request.json");
    }

    #[test]
    fn response_fixture_matches() {
        use crate::wire::OutputEvent;
        let fixtures = fixtures();
        let raw = load_text_fixture(&fixtures, "response_raw.jsonl");
        let events: Vec<OutputEvent> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("parse fixture line"))
            .collect();
        let stream = futures_util::stream::iter(events.into_iter().map(Ok));
        let response = futures_executor::block_on(crate::streaming::collect_into_response(stream)).unwrap();
        assert_response_fixture_eq(&response, &fixtures, "response_expected.json");
    }

    #[tokio::test]
    async fn stream_fixture_matches() {
        let fixtures = fixtures();
        let stream = stream_from_fixture(&fixtures, "stream.jsonl");
        assert_stream_fixture_eq(
            stream,
            &fixtures,
            "stream_events.json",
            "stream_response_expected.json",
        )
        .await;
    }

    fn stream_from_fixture(fixtures: &FixtureDir, name: &str) -> anyllm::ChatStream {
        let bytes = load_text_fixture(fixtures, name).into_bytes();
        let byte_stream = futures_util::stream::iter([Ok::<_, std::io::Error>(bytes::Bytes::from(bytes))]);
        let events = crate::streaming::parse_ndjson(byte_stream);
        let mut mapper = crate::streaming::StreamEventMapper::new();
        let mapped = futures_util::StreamExt::flat_map(events, move |evt| {
            let items: Vec<anyllm::Result<anyllm::StreamEvent>> = match evt {
                Ok(e) => mapper.map(e).into_iter().map(Ok).collect(),
                Err(e) => vec![Err(e)],
            };
            futures_util::stream::iter(items)
        });
        Box::pin(mapped)
    }
}
```

- [ ] **Step 4: Add `futures-executor` to dev-dependencies**

```toml
futures-executor = { workspace = true }
```

- [ ] **Step 5: Add `mod conformance_tests;` to `lib.rs` (test-only)**

```rust
#[cfg(test)]
mod conformance_tests;
```

- [ ] **Step 6: Run**

```bash
cargo test -p anyllm-claude-code conformance_tests
```

Expected: All three conformance tests pass. If `assert_response_fixture_eq` complains about the `metadata` shape, peek at the helper and adjust the fixture file.

- [ ] **Step 7: Commit**

```bash
git add crates/anyllm-claude-code
git commit -m "test(claude-code): conformance fixtures for request/response/stream"
```

### Task 20: Optional integration test gated on real `claude`

**Files:**
- Create: `crates/anyllm-claude-code/tests/live.rs`

- [ ] **Step 1: Write the live test (ignored by default)**

Create `crates/anyllm-claude-code/tests/live.rs`:

```rust
//! Integration tests against a real `claude` binary.
//!
//! Requires:
//!   - `claude` on PATH (or `CLAUDE_CODE_BIN` set)
//!   - `CLAUDE_CODE_OAUTH_TOKEN` set
//!
//! Run with: `cargo test -p anyllm-claude-code --test live -- --ignored`

use anyllm::prelude::*;
use anyllm_claude_code::Provider;

#[tokio::test]
#[ignore = "requires real Claude Code subscription credentials"]
async fn says_hi_for_real() {
    let provider = Provider::from_env().expect("CLAUDE_CODE_OAUTH_TOKEN must be set");
    let response = provider
        .chat(&ChatRequest::new("claude-haiku-4-5").user("Say only the word `pong` and nothing else."))
        .await
        .unwrap();
    let text = response.text_or_empty().to_lowercase();
    assert!(text.contains("pong"), "expected response to contain 'pong', got: {text}");
}

#[tokio::test]
#[ignore = "requires real Claude Code subscription credentials"]
async fn streams_for_real() {
    let provider = Provider::from_env().expect("CLAUDE_CODE_OAUTH_TOKEN must be set");
    let mut stream = provider
        .chat_stream(&ChatRequest::new("claude-haiku-4-5").user("Say only the word `pong` and nothing else."))
        .await
        .unwrap();

    let mut text = String::new();
    while let Some(evt) = stream.next().await {
        if let StreamEvent::TextDelta { text: delta, .. } = evt.unwrap() {
            text.push_str(&delta);
        }
    }
    assert!(text.to_lowercase().contains("pong"));
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo test -p anyllm-claude-code --test live -- --list
```

Expected: Both tests listed (and ignored).

- [ ] **Step 3: Commit**

```bash
git add crates/anyllm-claude-code/tests/live.rs
git commit -m "test(claude-code): live integration tests behind --ignored"
```

---

## Phase 10 — Examples and documentation

### Task 21: Examples in `crates/anyllm-examples`

**Files:**
- Create: `crates/anyllm-examples/examples/claude_code_chat.rs`
- Create: `crates/anyllm-examples/examples/claude_code_stream.rs`
- Create: `crates/anyllm-examples/examples/claude_code_tools.rs`
- Modify: `crates/anyllm-examples/Cargo.toml` (add `anyllm-claude-code` dep + `[[example]]` entries)

- [ ] **Step 1: Add the dep entry**

In `crates/anyllm-examples/Cargo.toml`, add to `[dependencies]`:

```toml
anyllm-claude-code = { workspace = true }
```

And append:

```toml
[[example]]
name = "claude_code_chat"
path = "examples/claude_code_chat.rs"

[[example]]
name = "claude_code_stream"
path = "examples/claude_code_stream.rs"

[[example]]
name = "claude_code_tools"
path = "examples/claude_code_tools.rs"
```

- [ ] **Step 2: `claude_code_chat.rs`**

```rust
//! One-shot chat against the Claude Code CLI.
//!
//! Run with: `CLAUDE_CODE_OAUTH_TOKEN=... cargo run -p anyllm-examples --example claude_code_chat`

use anyllm::prelude::*;
use anyllm_claude_code::Provider;

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = Provider::from_env()?;
    let response = provider
        .chat(&ChatRequest::new("claude-haiku-4-5").user("Say hello in three words."))
        .await?;
    println!("{}", response.text_or_empty());
    Ok(())
}
```

- [ ] **Step 3: `claude_code_stream.rs`**

```rust
//! Streaming chat against the Claude Code CLI.

use anyllm::prelude::*;
use anyllm_claude_code::Provider;

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = Provider::from_env()?;
    let mut stream = provider
        .chat_stream(&ChatRequest::new("claude-haiku-4-5").user("Count from one to five."))
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { text, .. } => print!("{text}"),
            StreamEvent::ResponseMetadata { finish_reason, .. } => {
                println!();
                println!("done: {:?}", finish_reason);
            }
            _ => {}
        }
    }
    Ok(())
}
```

- [ ] **Step 4: `claude_code_tools.rs`**

```rust
//! Chat with a custom tool exposed via the in-process MCP server.

use anyllm::prelude::*;
use anyllm_claude_code::{Provider, ToolDispatcher};
use serde_json::json;

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let dispatcher = ToolDispatcher::new().register("get_time", |_args| async move {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| anyllm::Error::Provider {
                status: None, message: e.to_string(), body: None, request_id: None,
            })?
            .as_secs();
        Ok(format!("Unix time: {now}"))
    });

    let provider = Provider::from_env()?.with_tools(dispatcher);

    let response = provider
        .chat(
            &ChatRequest::new("claude-haiku-4-5")
                .user("Use the get_time tool and tell me what Unix time it is.")
                .tools(vec![Tool::new(
                    "get_time",
                    json!({"type": "object", "properties": {}}),
                )
                .description("Returns the current Unix timestamp.")]),
        )
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

- [ ] **Step 5: Verify all examples compile**

```bash
cargo check -p anyllm-examples --examples
```

Expected: All compile. Do not run them — they require a real Claude Code subscription.

- [ ] **Step 6: Commit**

```bash
git add crates/anyllm-examples
git commit -m "docs(examples): claude-code chat, stream, and tools examples"
```

### Task 22: Crate README and final polish

**Files:**
- Create: `crates/anyllm-claude-code/README.md`
- Modify: `crates/anyllm-claude-code/src/lib.rs` — final doc-comment pass
- Modify: `crates/anyllm-claude-code/Cargo.toml` — verify `[package.metadata.docs.rs]`

- [ ] **Step 1: Crate README**

Create `crates/anyllm-claude-code/README.md`:

```markdown
# anyllm-claude-code

[![crates.io](https://img.shields.io/crates/v/anyllm-claude-code?style=flat-square)](https://crates.io/crates/anyllm-claude-code)
[![docs.rs](https://img.shields.io/docsrs/anyllm-claude-code?style=flat-square)](https://docs.rs/anyllm-claude-code)

Drive your **Claude Code subscription** through the `anyllm` portable
interface by wrapping the `claude` CLI as a regular `ChatProvider`.

## What this is

Per `chat()` call this crate spawns a one-shot `claude -p
--input-format stream-json --output-format stream-json` subprocess,
pipes the rendered conversation in on stdin, drains stream-json events
from stdout, and exposes the result as an `anyllm` `ChatResponse` /
`ChatStream`. Tools defined on the request are bridged via an in-process
HTTP MCP server bound on a random localhost port and authenticated by a
one-shot bearer token.

## What this is not

- **Production-grade.** Subscription auth wraps a CLI built for
  interactive use. We use it as a portable provider; expect rough edges.
- **An agent framework.** This is `anyllm`: chat in, response out.
- **A sandbox.** v1 only does soft scoping (per-call `$HOME`, scoped
  tmp/plugin/debug dirs, locked-down env vars). True process isolation
  is a future `Sandbox` impl (see the design spec).

## Quick start

```bash
claude setup-token   # one-time, generates an OAuth token
export CLAUDE_CODE_OAUTH_TOKEN="..."
```

```rust
use anyllm::prelude::*;
use anyllm_claude_code::Provider;

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = Provider::from_env()?;
    let response = provider
        .chat(&ChatRequest::new("claude-haiku-4-5").user("Say hi."))
        .await?;
    println!("{}", response.text_or_empty());
    Ok(())
}
```

See `crates/anyllm-examples/examples/claude_code_*.rs` for streaming and
tool-use examples.

## Capability summary

| Feature                          | Status           |
| -------------------------------- | ---------------- |
| Chat (one-shot + streaming)      | ✅                |
| Multi-turn with tool re-entry    | ✅                |
| Tools (via MCP)                  | ✅                |
| Parallel tool calls              | ✅                |
| `ToolChoice::Required`/`Specific`| Approximated     |
| Reasoning config + output        | ✅                |
| Image input                      | ✅                |
| Image output                     | ❌                |
| `ResponseFormat::Json*`          | ❌ native; ✅ via `ExtractingProvider` |
| Embeddings                       | Not implemented  |

See [`docs/superpowers/specs/2026-05-03-claude-code-provider-design.md`](../../docs/superpowers/specs/2026-05-03-claude-code-provider-design.md)
for the full matrix and architecture.

## Note on the Claude Code subscription

The Claude Code subscription is licensed for interactive CLI use. Wrapping
it as a headless provider works technically but bends that contract.
Use this crate for personal experimentation and testing, not production
service deployments. If you need supported headless API access, use
[`anyllm-anthropic`](../anyllm-anthropic) with an Anthropic API key.

## License

MIT
```

- [ ] **Step 2: Polish lib.rs doc comment**

Verify the lib.rs top-of-file doc string includes a working example, links to the spec, and notes the ToS caveat. The Provider/ProviderBuilder/ChatProvider impl methods all have rustdoc — sweep for missing.

- [ ] **Step 3: Run the full test suite + clippy**

```bash
cargo test -p anyllm-claude-code
cargo clippy -p anyllm-claude-code --all-targets -- -D warnings
cargo doc -p anyllm-claude-code --no-deps
```

Expected: All pass; doc builds without warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/anyllm-claude-code/README.md crates/anyllm-claude-code/src/lib.rs
git commit -m "docs(claude-code): crate README and rustdoc polish"
```

---

## Self-review checklist

After implementation, verify against the spec:

1. **Spec coverage:** Walk each numbered section of `2026-05-03-claude-code-provider-design.md`:
   - §3 Public API surface → Tasks 5, 12, 16, 17
   - §4 Architecture → Tasks 11, 14, 16
   - §5 Per-call flow → Task 14 (orchestration), Task 16 (entry points)
   - §6 Subprocess invocation → Task 10 (argv), Task 14 (spawn)
   - §7 Capability matrix → Task 16 (`builtin_capability`), live tests in Task 20 confirm at runtime
   - §8 Env vars → Task 10 (`build_env`)
   - §9 Sandbox abstraction → Task 5
   - §10 Error mapping → Task 4 + scattered `.map_err(...)` in Tasks 11, 14
   - §11 Lifecycle / cancellation → Task 14 (`CallGuard::Drop`), Task 16 (`run_chat` timeout, `run_chat_stream` guard ownership)
   - §12 Crate layout → Tasks 2, 22
   - §13 Validation spike → Task 1
   - §14 Future work → README (Task 22) and design doc only
2. **Sandbox future-proofing:** §9 says `BwrapSandbox` etc. must drop in without touching the Provider. Confirm `Sandbox` is the only point of contact (`Provider` calls `config.sandbox.build_command(spec)` and nothing else). Task 14 step 4 verifies.
3. **Tool-name prefix consistency:** Renderer adds `mcp__anyllm__` (Task 9), parser strips it (Task 8). Names round-trip in the multi-turn integration test (Task 18).
4. **Cleanup on cancellation:** `CallGuard::Drop` runs even when the future is cancelled. Task 14 step 1 explains; consider adding a regression test that cancels mid-stream to confirm. (If it's important, add as a separate task at the end.)
5. **No backward-compat shims:** All env vars, CLI flags, and tool lists are documented as "matches CLI version X" — leave a comment in `render::DISALLOWED_BUILTIN_TOOLS` noting the verified version.

---

## Done

When all tasks above are checked off, the crate ships with:

- A working `ChatProvider` for the Claude Code subscription
- Full mock-driven test coverage in CI
- Real-world integration tests gated behind `--ignored`
- Conformance fixtures aligned with the existing provider pattern
- Three runnable examples
- A README that's honest about scope and limitations
