# anyllm-openai-compat

Reusable OpenAI-compatible adapter toolkit for `anyllm` provider authors.

This crate is not primarily for end-user application code. It exists to make it
cheap to build a new `anyllm` provider crate when the upstream API looks close
to OpenAI chat completions but still has provider-specific headers, paths,
metadata, or request fields.

## Use it when

- you are building a new provider crate, not just consuming one
- the upstream API is OpenAI-like enough that request/response/stream reuse is worthwhile
- you want to keep provider-specific behavior in thin adapter code instead of forking a full provider implementation

## What it provides

- request translation helpers
- response conversion helpers
- SSE stream event normalization
- transport configuration for non-default header names and paths
- shared error mapping utilities

## Typical shape

An adapter crate usually does three things:

1. define provider-specific request options and metadata types,
2. configure `TransportConfig` for the provider's headers and endpoint path,
3. call the shared conversion and streaming helpers.

`crates/anyllm-openai-compat/tests/reuse_end_to_end.rs` is the in-tree proof
that the shared request/response/stream helpers can power a thin custom adapter.

## Observability

`ProviderBuilder::build()` leaves `provider_name` set to `"unknown"` when the
caller does not specify one. That value surfaces as
`gen_ai.provider.name="unknown"` on tracing spans, which is a deliberate
"not configured" signal rather than a generic label that misrepresents the
upstream.

Pick one:

- use a preset from [`providers`](src/providers.rs) (e.g. `Cloudflare`),
  which injects the correct identity;
- call `.provider_name("your-provider")` on the builder before `.build()`
  when wiring a custom upstream.

## Why this exists

Without this crate, every OpenAI-style provider adapter would need to duplicate:

- wire request shaping
- SSE parsing and ordered stream event reconstruction
- response conversion
- transport quirks and header extraction

`anyllm-openai-compat` keeps those concerns shared while still allowing a thin
provider-specific layer to own branding, extra wire fields, and metadata hooks.
