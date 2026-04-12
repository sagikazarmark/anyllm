# anyllm-openai

OpenAI provider adapter for `anyllm`.

## Use it when

- you want to call OpenAI models through the shared `anyllm::ChatProvider` API
- you want tools, streaming, and model-dependent OpenAI features like vision or native structured outputs
- you may later swap providers without rewriting application logic

## Capabilities

`anyllm-openai` currently reports:

- tools
- streaming
- model-dependent vision support as `Unknown`
- model-dependent structured output support as `Unknown`
- model-dependent parallel tool call support as `Unknown`

It does not currently expose reasoning/thinking content as first-class
`ContentBlock::Reasoning` output.

## Quick start

```toml
[dependencies]
anyllm = "0.1.0"
anyllm-openai = "0.1.0"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use anyllm::prelude::*;
use anyllm_openai::Provider;

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Provider::from_env()?;

    let response = provider
        .chat(&ChatRequest::new("gpt-4o").user("Say hello in one sentence."))
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

Environment variables:

- required: `OPENAI_API_KEY`
- optional: `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`

## Notes

- Use `Provider::new_with_base_url(...)` or `Provider::builder()` when you need
  a custom base URL or HTTP client.
- The default transport applies a 120 second timeout to non-streaming requests
  only. Streaming requests intentionally avoid a whole-request wall clock
  timeout so long-lived SSE responses are not cut off mid-stream.
- Inject your own `reqwest::Client` with `Provider::builder().client(...)` when
  you need a different transport policy.
- Portable `ChatRequest.seed(...)` and `ChatRequest.parallel_tool_calls(...)`
  are the supported ways to set those controls. `ChatRequestOptions` is only for
  OpenAI-specific request fields.
- Response metadata is exposed as typed `ChatResponseMetadata` in
  `ChatResponse.metadata`.
- `OpenAIReasoningEffort` is re-exported from this crate so callers do not need
  a direct dependency on `anyllm-openai-compat` to use OpenAI-specific
  reasoning-effort controls.
- If you need to build an adapter for an OpenAI-style non-OpenAI provider, look
  at `anyllm-openai-compat` instead of copying this crate.
