# anyllm-examples

Runnable end-to-end examples for the env-driven `anyllm` providers.

## Quick start

Pick one HTTP provider, export its credential, and run an example:

```bash
export PROVIDER=openai
export OPENAI_API_KEY=...
# optional: export OPENAI_MODEL=gpt-4.1-mini
cargo run -p anyllm-examples --example provider_chat -- "Explain Rust ownership"
```

Swap `PROVIDER` to `anthropic` or `gemini` and set the matching `*_API_KEY` to
run the same examples against a different provider.

## Providers

- `PROVIDER=openai` uses `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`
- `PROVIDER=anthropic` uses `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_BASE_URL`
- `PROVIDER=gemini` uses `GEMINI_API_KEY` and optionally `GEMINI_BASE_URL`

Cloudflare Workers AI is not part of this loader because it is exercised
through an HTTP smoke endpoint rather than a Rust provider built from local
env vars alone. See [Live tests](#live-tests) below for how to point the live
suite at a running smoke endpoint.

Each provider also supports a model override env var used by these examples:

- OpenAI: `OPENAI_MODEL` (chat), `OPENAI_EMBEDDING_MODEL` (embedding)
- Anthropic: `ANTHROPIC_MODEL` (chat only — no embedding API)
- Gemini: `GEMINI_MODEL` (chat), `GEMINI_EMBEDDING_MODEL` (embedding)

If `PROVIDER` is unset, the loader auto-selects only when exactly one provider
credential env var is configured. Otherwise it fails fast and tells you to set
`PROVIDER` explicitly.

## Examples

Chat examples (all three providers):

- `provider_chat`: one-shot request/response
- `provider_stream`: normalized streaming text and metadata events
- `provider_tools`: tool-call -> tool-result -> follow-up answer loop
- `provider_extract`: env-driven structured extraction across providers
- `record_replay`: portable request/response recording and replay

Embedding examples (OpenAI and Gemini only — Anthropic has no embedding API):

- `provider_embedding`: batch text embedding with vector preview

Example commands:

```bash
PROVIDER=openai cargo run -p anyllm-examples --example provider_chat -- "Explain Rust ownership"
PROVIDER=anthropic cargo run -p anyllm-examples --example provider_chat -- "Explain Rust ownership"
PROVIDER=gemini cargo run -p anyllm-examples --example provider_chat -- "Explain Rust ownership"
PROVIDER=openai cargo run -p anyllm-examples --example provider_stream -- "Explain Rust ownership"
PROVIDER=openai cargo run -p anyllm-examples --example provider_tools -- "San Francisco"
PROVIDER=openai cargo run -p anyllm-examples --example provider_extract -- "Summarize why Rust is good for agents"
PROVIDER=openai cargo run -p anyllm-examples --example record_replay -- "Summarize this failure"
PROVIDER=openai cargo run -p anyllm-examples --example provider_embedding -- "Hello embeddings" "Second input"
PROVIDER=gemini cargo run -p anyllm-examples --example provider_embedding -- "Hello embeddings" "Second input"
```

## Live tests

The crate also includes an opt-in live integration suite that calls the real
provider APIs.

Run one provider:

```bash
ANYLLM_LIVE_PROVIDER=openai cargo test -p anyllm-examples --test live_providers -- --nocapture --test-threads=1
```

Run all configured providers:

```bash
ANYLLM_LIVE_PROVIDER=all cargo test -p anyllm-examples --test live_providers -- --nocapture --test-threads=1
```

`ANYLLM_LIVE_PROVIDER=configured` is accepted as an alias for `all`.

When using `all` or `configured`, the HTTP suite skips providers whose primary
credential env var is unset, but the overall live gate now fails if the
selection resolves to no runnable targets at all.

When selecting a provider explicitly, missing required env vars are treated as a
hard failure rather than a skip.

The HTTP live test runs basic chat, streaming, a validated tool round-trip, and
structured extraction sequentially for each selected provider.

Cloudflare Workers AI is covered separately through a Worker smoke endpoint
rather than a Rust provider. Point the live suite at any running smoke
deployment that speaks the expected HTTP shape:

```bash
ANYLLM_LIVE_PROVIDER=cloudflare-worker \
ANYLLM_CLOUDFLARE_WORKER_SMOKE_URL=http://127.0.0.1:8787 \
cargo test -p anyllm-examples --test live_cloudflare_worker -- --nocapture --test-threads=1
```

Optionally set `CLOUDFLARE_WORKER_MODEL` to override the default smoke app
model per request. `CLOUDFLARE_WORKER_TOOL_MODEL` and
`CLOUDFLARE_WORKER_JSON_MODEL` are also supported for route-specific
overrides.
