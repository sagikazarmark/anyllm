# anyllm-conformance

Fixture-based conformance harness for `anyllm` provider adapters.

This crate exists for provider authors. It gives new adapters a shared bar for
request shaping, response normalization, streaming event ordering, and error
mapping so the provider ecosystem can grow without each crate inventing its own
test contract.

From the workspace root:

```bash
cargo test -p anyllm-conformance
```

## What it checks

The harness supports four main fixture categories:

- request fixtures: provider wire request output
- response fixtures: normalized `ChatResponse` output
- stream fixtures: ordered `StreamEvent` output plus collected final response
- error fixtures: normalized `anyllm::Error` mapping

## What a provider crate should prove

At minimum, a provider should have conformance tests covering:

1. request conversion from `ChatRequest` into the provider wire shape
2. non-streaming response conversion into `ChatResponse`
3. streaming conversion into ordered `StreamEvent`s and a strict final response
4. representative error mapping for auth, rate limit, context length,
   model-not-found when mapped distinctly, and timeout

Provider crates such as OpenAI, Anthropic, and Gemini should follow a
recognizable conformance shape.

## Complementary transport tests

Conformance fixtures prove request shaping, normalization, and streaming logic in
isolation. For first-release provider crates, add a second layer of transport
integration tests that exercise the real provider implementation against a local
mock HTTP server.

These tests should verify at least:

1. URL and endpoint selection
2. required headers and auth wiring
3. one-shot response decoding through `provider.chat(...)`
4. streaming response decoding through `provider.chat_stream(...)`
5. non-2xx HTTP error mapping with real response headers

`anyllm-conformance` exposes a small `TestHttpServer` helper for this purpose so
provider crates can keep the tests focused on provider behavior rather than on
server scaffolding.

Typical transport suite commands:

- `cargo test -p anyllm-openai --test http_integration`
- `cargo test -p anyllm-anthropic --test http_integration`
- `cargo test -p anyllm-gemini --test http_integration`

## Typical test pattern

Provider crates usually add a `conformance_tests.rs` module that:

```rust
use std::path::PathBuf;

use anyllm_conformance::{
    assert_error_fixture_eq,
    assert_json_fixture_eq,
    assert_response_fixture_eq,
    assert_stream_fixture_eq,
    FixtureDir,
    load_json_fixture,
    load_text_fixture,
};

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

#[test]
fn request_fixture_matches() {
    let fixtures = fixtures();
    let actual = provider_specific_request_value();
    assert_json_fixture_eq(&actual, &fixtures, "request.json");
}

#[test]
fn response_fixture_matches() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "response_raw.json");
    let response = provider_specific_response_conversion(raw).unwrap();
    assert_response_fixture_eq(&response, &fixtures, "response_expected.json");
}

#[tokio::test]
async fn stream_fixture_matches() {
    let fixtures = fixtures();
    let sse = load_text_fixture(&fixtures, "stream.sse");
    let stream = provider_specific_stream_conversion(&sse);
    assert_stream_fixture_eq(
        stream,
        &fixtures,
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;
}

#[test]
fn error_fixtures_match() {
    let fixtures = fixtures();
    let error = provider_specific_error_mapping();
    assert_error_fixture_eq(&error, &fixtures, "error_auth.json");
}
```

## Fixture layout

Each provider crate owns its own fixtures under `<provider-crate>/fixtures/`.

Examples:

- `crates/anyllm-openai/fixtures/`
- `crates/anyllm-anthropic/fixtures/`
- `crates/anyllm-gemini/fixtures/`

Common files:

- `request.json`
- `response_raw.json`
- `response_expected.json`
- `stream.sse`
- `stream_events.json`
- `stream_response_expected.json`
- `error_auth.json`
- `error_rate_limited.json`
- `error_context_length.json`
- `error_model_not_found.json` when the provider maps it distinctly
- `error_timeout.json`

You can add more provider-specific fixtures if a provider has meaningful edge
cases, but the common set should stay recognizable across crates.

When a provider has important interrupted-stream behavior to prove, add the
matching extra fixtures explicitly instead of overloading the happy-path files:

- `stream_truncated.sse`
- `stream_truncated_events.json`
- `stream_truncated_response_expected.json`
- `stream_truncated_completeness.json`
- `stream_truncated_error.json`

That lets a provider prove both sides of the contract:

- strict collection via `StreamCollector::finish()` rejects incomplete streams
- recovery via `StreamCollector::finish_partial()` preserves salvageable
  non-tool content, drops incomplete tool calls, and reports
  `StreamCompleteness`

## How to add a new provider

1. Build the provider crate around `anyllm::ChatProvider`.
2. Add request/response/stream/error conversion entry points that are easy to
   call from tests.
3. Create a `fixtures/` directory in the provider crate.
4. Add a `conformance_tests.rs` module in the provider crate.
5. Assert request, response, stream, and error fixtures with this crate.
6. Treat fixture updates as contract changes, not just snapshot churn.

## Important contract details

- Stream conformance now assumes strict `StreamCollector::finish()` semantics.
  A successful stream fixture must include a complete event sequence ending in a
  normal `ResponseStop`.
- Truncated stream fixtures should normally assert both the strict failure and
  the recovery path. In practice that means checking the same event sequence
  with `assert_stream_finish_error_fixture_eq(...)` and
  `assert_partial_stream_fixture_eq(...)`.
- `StreamCompleteness` is part of the public recovery contract. Providers should
  prove whether a truncated stream missed `ResponseStop`, left blocks open, or
  dropped incomplete tool calls.
- Response fixture comparisons use `ChatResponse::to_log_value()`, so typed
  `ResponseMetadata` is compared through its JSON export shape.
- Error fixture comparisons serialize `Error::as_log()`, which keeps fixture formats
  stable even though `anyllm::Error` itself is not directly serializable.

## Why this matters

This crate is one of the main reasons the current provider ecosystem is more
than a loose abstraction demo. It gives adapter authors a concrete way to prove
that a new provider behaves like the existing ones at the `anyllm` boundary.

Used together, fixture conformance tests, mock transport integration tests, and
opt-in live provider tests provide a much better release bar than any one of
those layers on its own.
