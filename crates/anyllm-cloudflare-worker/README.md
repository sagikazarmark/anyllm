# anyllm-cloudflare-worker

Cloudflare Workers AI provider adapter for `anyllm`.

## Use it when

- you are already running inside a Cloudflare Worker
- you want to call Workers AI through the shared `anyllm::ChatProvider` trait
- you want native binding-based execution instead of issuing outbound HTTP calls

## Identity

`Provider::provider_name()` returns `"cloudflare"` so OTEL `gen_ai.provider.name`
matches the value emitted by `anyllm-openai-compat`'s Cloudflare preset,
regardless of which transport you choose.

## Capabilities

`anyllm-cloudflare-worker` currently reports:

- tools
- streaming
- structured output

It does not currently support multimodal user input or assistant image replay on
the native Workers AI chat path used by this crate.

The native chat path also rejects unsupported request controls such as
`stop`, `tool_choice`, provider-agnostic reasoning config, and
`parallel_tool_calls` rather than silently ignoring them.

`Structured output` here means request-level `ChatRequest.response_format`
mapping on the non-streaming native chat path. This crate does not currently
expose `anyllm`'s `ExtractExt` convenience API.

For `ResponseFormat::JsonSchema`, this adapter currently supports the schema
payload itself but not the portable `name` / `strict` controls, which are
rejected explicitly instead of being silently dropped.

`chat_stream(...)` is intentionally narrower than `chat(...)`: it rejects tools
and `response_format` requests because Workers AI JSON Mode is non-streaming and
the native stream exposed through `worker::Ai::run_bytes()` does not surface a
stable streamed tool-call event shape.

## Quick start

```toml
[dependencies]
anyllm = "0.1.0"
anyllm-cloudflare-worker = "0.1.0"
futures-util = "0.3"
worker = "0.7"
```

```rust,ignore
use anyllm::prelude::*;
use anyllm_cloudflare_worker::Provider;

async fn handler(env: &worker::Env) -> Result<String> {
    let ai = env.ai("AI").map_err(|e| anyllm::Error::Provider {
        status: None,
        message: e.to_string(),
        body: None,
        request_id: None,
    })?;

    let provider = Provider::new(ai);

    let response = provider
        .chat(&ChatRequest::new("@cf/meta/llama-3.1-8b-instruct").user("What is Rust?"))
        .await?;

    Ok(response.text_or_empty())
}
```

## Notes

- This crate does not use `from_env()` because construction happens from a
  Cloudflare `worker::Ai` binding.
- The binding name must match your `wrangler.toml` configuration:

```toml
[ai]
binding = "AI"
```
