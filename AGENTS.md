# AGENTS.md

## Scope

- Build `anyllm` as a low-level, universal client for LLM applications.
- Keep it a building block. Do not add agent loops or turn the library into an agent framework.
- Scope may grow to adjacent model operations such as embeddings, transcription, and speech, but keep the same low-level, provider-agnostic design.
- Do not force adjacent capabilities into chat-shaped APIs just to preserve a single uniform surface.

## Abstraction

- Preserve a portable core for concepts that genuinely map across providers.
- Never fake portability. If a capability does not align cleanly across providers, keep it provider-specific.
- Prefer explicit escape hatches over leaky abstractions. Provider-specific behavior belongs in typed options, metadata, extra fields, or provider crates, not in the shared API.
- Do not add provider-branded concepts, request fields, or response types to the common surface unless the semantics are truly portable.
- When promoting a provider feature into the core API, design the neutral domain model first. Do not wrap one provider and rename it.
- Provide shared APIs for cross-cutting capabilities such as structured output extraction and tracing when their semantics remain provider-agnostic end to end.
- If a provider exposes a proprietary version first, design the neutral domain model before exposing it in the common API.

## API Design

- Optimize first for user experience in Rust application code: easy to discover, easy to read, and idiomatic.
- Keep common request and response flows direct and inspectable. Avoid forcing users through opaque builder-only or provider-shaped APIs.
- Preserve extensibility for library users. Do not introduce patterns that make custom providers, wrappers, or extensions harder to build.
- Use precise, correct domain language. Prefer stable LLM concepts over vendor marketing terms or transport details.
- Design for the breadth of LLM application use cases. Do not narrow the abstraction around one provider or a single happy path.

## Acceptance

- Passing the existing Rust quality gates is the minimum bar, not the finish line.
- A change is not done until it also preserves or improves domain language, API clarity, abstraction quality, and extensibility.
- Reject changes that add capability by coupling the application-facing API to one provider's quirks.
- Reject changes that make the API harder to understand even if they are technically more complete.
- Provider work needs rigorous unit and integration coverage.
- When provider behavior is non-trivial, prefer runnable examples over doc-only examples.

## Conventions

- Structure Rust source files with the primary public trait or struct first, then impls, then supporting types, then internal helpers, then tests.
- Prefer `foo.rs` + `foo/` over `foo/mod.rs`.

## Example

```rust
#[derive(Clone)]
struct ProviderHint {
    route: &'static str,
}

let request = ChatRequest::new("demo-model")
    .system("You are concise.")
    .user("Say hello")
    .with_option(ProviderHint { route: "fast-path" });
```

- Keep the common path this direct. Add provider-specific power through typed options or extensions, not by making the portable API provider-shaped.
