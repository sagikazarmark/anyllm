#![warn(missing_docs)]
//! Provider-agnostic chat completion primitives and adapters.
//!
//! The crate root exposes the common day-to-day API surface directly:
//! [`ChatProvider`], [`ChatRequest`], [`ChatResponse`], [`Tool`], and related
//! streaming and wrapper types.
//!
//! Use [`prelude`] when you want a compact import for typical application code.
//! Reach for explicit root imports when writing libraries or examples that need
//! to make advanced types like [`ChatRequestRecord`], [`ChatResponseRecord`],
//! [`CollectedResponse`], or [`StreamCompleteness`] obvious at the callsite.
//!
//! ## Portability Model
//!
//! `anyllm` aims to keep the common request/response path provider-agnostic,
//! while still leaving explicit escape hatches for provider-specific features.
//!
//! The portable core is centered on [`ChatRequest`], [`ChatResponse`],
//! [`Message`], [`ContentBlock`], [`Tool`], and the streaming event model.
//! These types intentionally expose their portable fields directly and pair
//! them with fluent constructors/helpers so application and test code can build
//! and adjust requests and responses without going through opaque builders.
//!
//! Provider-specific data lives in a few deliberate places:
//! [`RequestOptions`] and [`ResponseMetadata`] for typed extensions,
//! [`ExtraMap`] and `extensions` fields for portable JSON-shaped extras, and
//! `Other` enum variants for provider payloads that do not fit the normalized
//! model yet.
//!
//! This means portability is best-effort rather than absolute. Converting to
//! [`ChatRequestRecord`] or [`ChatResponseRecord`] preserves the portable core,
//! but typed provider-specific data may be dropped or flattened to JSON.

mod chat;
mod content;
mod error;
#[cfg(feature = "extract")]
mod extract;
mod fallback;
mod message;
/// Mock providers for testing. Enable the `mock` feature to use these types
/// as a library consumer. These types are not covered by the semver stability
/// guarantee.
#[cfg(any(test, feature = "mock"))]
mod mock;
mod options;
mod request;
mod response;
mod retry;
mod stream;
mod tool;
mod usage;
mod utils;

#[cfg(feature = "tracing")]
mod tracing;
#[cfg(feature = "tracing")]
pub use tracing::{TracingChatProvider, TracingContentConfig, otel_genai_provider_name};

pub use chat::{
    CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatProviderExt,
    DynChatProvider,
};
pub use content::{ContentBlock, ImageBlockRef, OwnedToolCall, ToolCallRef};
pub use error::{Error, ErrorLog, Result, SerializationError};
#[cfg(feature = "extract")]
pub use extract::{
    ExtractError, ExtractExt, Extracted, ExtractingProvider, ExtractionMetadata, ExtractionMode,
    ExtractionRequest, Extractor,
};
pub use fallback::FallbackChatProvider;
pub use message::{
    AssistantMessageRef, ContentPart, ImagePartRef, ImageSource, Message, ToolMessageRef,
    ToolResultContent, UserContent, UserMessageRef,
};
#[cfg(any(test, feature = "mock"))]
pub use mock::{
    ChatResponseBuilder, MockProvider, MockProviderBuilder, MockResponse, MockStreamEvent,
    MockStreamingProvider, MockStreamingProviderBuilder, MockToolRoundTrip,
};
pub use options::{RequestOptions, ResponseMetadata, ResponseMetadataType};
pub use request::{
    ChatRequest, ChatRequestRecord, ReasoningConfig, ReasoningEffort, ResponseFormat,
};
pub use response::{ChatResponse, ChatResponseRecord, FinishReason};
pub use retry::{RetryPolicy, RetryingChatProvider};
pub use stream::{
    ChatStream, ChatStreamExt, CollectedResponse, SingleResponseStream, StreamBlockType,
    StreamCollector, StreamCompleteness, StreamEvent, UsageMetadataMode,
};
pub use tool::{Tool, ToolChoice};
pub use usage::Usage;

/// Portable JSON object used for provider-specific escape hatches.
/// Uses `Map<String, Value>` instead of `Value` to enforce object semantics at compile time.
pub type ExtraMap = serde_json::Map<String, serde_json::Value>;

/// Prelude module — import `use anyllm::prelude::*` for common application-facing types.
///
/// The prelude intentionally omits some more specialized record and stream
/// diagnostics types so those remain explicit when used.
pub mod prelude {
    pub use futures_util::StreamExt;

    pub use crate::{
        CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatProviderExt,
        ChatRequest, ChatResponse, ChatStream, ChatStreamExt, ContentBlock, ContentPart,
        DynChatProvider, Error, ErrorLog, ExtraMap, FallbackChatProvider, FinishReason,
        ImageSource, Message, OwnedToolCall, ReasoningConfig, ReasoningEffort, ResponseFormat,
        Result, RetryPolicy, RetryingChatProvider, SingleResponseStream, StreamBlockType,
        StreamCollector, StreamEvent, Tool, ToolCallRef, ToolChoice, ToolResultContent, Usage,
        UserContent,
    };

    #[cfg(any(test, feature = "mock"))]
    pub use crate::{
        ChatResponseBuilder, MockProvider, MockProviderBuilder, MockResponse, MockStreamEvent,
        MockStreamingProvider, MockStreamingProviderBuilder, MockToolRoundTrip,
    };

    #[cfg(feature = "extract")]
    pub use crate::{
        ExtractError, ExtractExt, Extracted, ExtractingProvider, ExtractionMetadata,
        ExtractionMode, ExtractionRequest, Extractor,
    };

    #[cfg(feature = "tracing")]
    pub use crate::{TracingChatProvider, TracingContentConfig};
}
