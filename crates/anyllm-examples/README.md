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

Each provider also supports a model override env var used by these examples:

- OpenAI: `OPENAI_MODEL` (chat), `OPENAI_EMBEDDING_MODEL` (embedding)
- Anthropic: `ANTHROPIC_MODEL` (chat only; no embedding API)
- Gemini: `GEMINI_MODEL` (chat), `GEMINI_EMBEDDING_MODEL` (embedding)

If `PROVIDER` is unset, the loader auto-selects only when exactly one provider
credential env var is configured. Otherwise it fails fast and tells you to set
`PROVIDER` explicitly.

## Examples

Chat examples (all three providers):

- `provider_chat`: one-shot request/response
- `provider_stream`: normalized streaming text and metadata events
- `provider_tools`: tool-call -> tool-result -> follow-up answer loop
- `provider_parallel_tools`: two tools exercised in one response with `parallel_tool_calls(true)`
- `provider_streaming_tools`: raw `StreamEvent` iteration showing tool-call blocks mid-stream
- `provider_vision`: `Message::user_multimodal` with an image URL
- `provider_reasoning`: `ReasoningConfig` + `reasoning_text()` inspection (see the example for per-provider model override notes)
- `provider_extract`: env-driven structured extraction across providers
- `record_replay`: portable request/response recording and replay

Embedding examples (OpenAI and Gemini only; Anthropic has no embedding API):

- `provider_embedding`: batch text embedding with vector preview

Multi-provider and provider-specific:

- `cross_provider_fallback`: `FallbackChatProvider<OpenAI, Anthropic>` with the default auth-filter policy
- `anthropic_prompt_caching`: Anthropic-specific `CacheControl` usage

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
ANYLLM_LIVE_PROVIDER=openai cargo test -p anyllm-examples --test providers -- --nocapture --test-threads=1
```

Run all configured providers:

```bash
ANYLLM_LIVE_PROVIDER=all cargo test -p anyllm-examples --test providers -- --nocapture --test-threads=1
```

`ANYLLM_LIVE_PROVIDER=configured` is accepted as an alias for `all`.

When using `all` or `configured`, the HTTP suite skips providers whose primary
credential env var is unset, but the overall live gate now fails if the
selection resolves to no runnable targets at all.

When selecting a provider explicitly, missing required env vars are treated as a
hard failure rather than a skip.

The HTTP live test runs basic chat, streaming, a validated tool round-trip, and
structured extraction sequentially for each selected provider.

