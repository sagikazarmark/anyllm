//! Chat completion primitives.
//!
//! This module declares the chat capability submodules and re-exports their
//! public types. See [`provider`] for the core trait machinery.

mod content;
mod fallback;
mod message;
mod provider;
mod request;
mod response;
mod retry;
mod stream;
mod system;
mod tool;

#[cfg(any(test, feature = "mock"))]
mod mock;

#[cfg(feature = "extract")]
mod extract;

#[cfg(feature = "tracing")]
mod tracing;

pub use content::{ContentBlock, ImageBlockRef, OwnedToolCall, ToolCallRef};
pub use fallback::FallbackChatProvider;
pub use message::{
    AssistantMessageRef, ContentPart, ImagePartRef, ImageSource, Message, ToolMessageRef,
    ToolResultContent, UserContent, UserMessageRef,
};
pub use provider::{
    ChatCapability, ChatCapabilityResolver, ChatProvider, ChatProviderExt, DynChatProvider,
};
pub use request::{
    ChatRequest, ChatRequestRecord, ReasoningConfig, ReasoningEffort, ResponseFormat,
};
pub use response::{ChatResponse, ChatResponseRecord, FinishReason};
pub use retry::{RetryPolicy, RetryingChatProvider};
pub use stream::{
    ChatStream, ChatStreamExt, CollectedResponse, SingleResponseStream, StreamBlockType,
    StreamCollector, StreamCompleteness, StreamEvent, UsageMetadataMode,
};
pub use system::{SystemOptions, SystemPrompt};
pub use tool::{Tool, ToolChoice};

#[cfg(any(test, feature = "mock"))]
pub use mock::{
    ChatResponseBuilder, MockProvider, MockProviderBuilder, MockResponse, MockStreamEvent,
    MockStreamingProvider, MockStreamingProviderBuilder, MockToolRoundTrip,
};

#[cfg(feature = "extract")]
pub use extract::{
    ExtractError, ExtractExt, Extracted, ExtractingProvider, ExtractionMetadata, ExtractionMode,
    ExtractionRequest, Extractor,
};

#[cfg(feature = "tracing")]
pub use tracing::{TracingChatProvider, TracingContentConfig, otel_genai_provider_name};
