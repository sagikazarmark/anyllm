use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::{CapabilitySupport, ProviderIdentity, Result};

mod content;
#[cfg(feature = "extract")]
mod extract;
mod fallback;
mod message;
#[cfg(any(test, feature = "mock"))]
mod mock;
mod request;
mod response;
mod retry;
mod stream;
mod tool;
#[cfg(feature = "tracing")]
mod tracing;

pub use content::{ContentBlock, ImageBlockRef, OwnedToolCall, ToolCallRef};
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
#[cfg(feature = "tracing")]
pub use tracing::{TracingChatProvider, TracingContentConfig, otel_genai_provider_name};

/// Core trait for LLM chat completion providers.
///
/// Implementors must provide [`chat`](ChatProvider::chat) and a concrete
/// [`Stream`](ChatProvider::Stream) type for [`chat_stream`](ChatProvider::chat_stream).
/// Providers that do not support true streaming can return
/// [`crate::SingleResponseStream`] to normalize a full [`ChatResponse`](crate::ChatResponse)
/// into ordered [`StreamEvent`]s without boxing.
///
/// Methods return `impl Future<…> + Send` so that the returned futures are
/// always `Send`, which wrapper types (retry, fallback, dyn dispatch) rely on.
/// Implementors should use `async fn`: the compiler desugars it to the same
/// RPITIT form and automatically satisfies the `Send` bound when all captured
/// state is `Send`.
///
/// For dynamic dispatch, use [`DynChatProvider`] which wraps any
/// `ChatProvider` behind a manually-erased vtable, boxing futures only at the
/// dyn boundary.
pub trait ChatProvider: ProviderIdentity {
    /// Concrete stream type returned by [`chat_stream`](Self::chat_stream)
    type Stream: Stream<Item = Result<StreamEvent>> + Send;

    /// Send a chat completion request and return the full response.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](crate::Error) on provider communication failures.
    fn chat(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send;

    /// Send a chat completion request and return a stream of chunks.
    ///
    /// Implementors choose their concrete stream type via [`Stream`](ChatProvider::Stream).
    /// Providers without native streaming can return [`SingleResponseStream`](crate::SingleResponseStream)
    /// to normalize a full response into ordered [`StreamEvent`]s.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](crate::Error) if the stream cannot be established.
    /// Individual stream items may also carry errors.
    fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = Result<Self::Stream>> + Send;

    /// Returns support information for a provider/model capability query.
    fn chat_capability(&self, _model: &str, _capability: ChatCapability) -> CapabilitySupport {
        CapabilitySupport::Unknown
    }
}

/// Portable chat features that a provider/model may support.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatCapability {
    /// Supports tool definitions and model-emitted tool call blocks.
    ToolCalls,
    /// Supports requesting multiple tool calls in a single assistant turn.
    ParallelToolCalls,
    /// Supports incremental response delivery through [`ChatProvider::chat_stream`].
    Streaming,
    /// Supports image parts in user messages.
    ImageInput,
    /// Supports per-image detail hints on user image parts.
    ImageDetail,
    /// May emit image blocks in assistant responses.
    ImageOutput,
    /// Supports replaying assistant image blocks back to the model in later turns.
    ImageReplay,
    /// Supports non-text response formats such as JSON or JSON Schema output.
    StructuredOutput,
    /// May emit reasoning blocks in assistant responses.
    ReasoningOutput,
    /// Supports replaying assistant reasoning blocks back to the model in later turns.
    ReasoningReplay,
    /// Supports request-side reasoning configuration via [`crate::ReasoningConfig`].
    ReasoningConfig,
}

/// Optional resolver used to customize a provider's chat capability answers.
///
/// Return `None` to defer to the provider's built-in answer. Return
/// `Some(...)` to override it, including `Some(CapabilitySupport::Unknown)`.
pub trait ChatCapabilityResolver: Send + Sync + 'static {
    /// Return an override for the queried capability, or `None` to defer to
    /// the provider's built-in capability logic.
    fn chat_capability(&self, model: &str, capability: ChatCapability)
    -> Option<CapabilitySupport>;
}

impl<F> ChatCapabilityResolver for F
where
    F: for<'a> Fn(&'a str, ChatCapability) -> Option<CapabilitySupport> + Send + Sync + 'static,
{
    fn chat_capability(
        &self,
        model: &str,
        capability: ChatCapability,
    ) -> Option<CapabilitySupport> {
        self(model, capability)
    }
}

impl<T> ChatProvider for &T
where
    T: ChatProvider + ?Sized,
{
    type Stream = T::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        T::chat(*self, request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        T::chat_stream(*self, request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        T::chat_capability(*self, model, capability)
    }
}

impl<T> ChatProvider for Box<T>
where
    T: ChatProvider + ?Sized,
{
    type Stream = T::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        T::chat(self.as_ref(), request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        T::chat_stream(self.as_ref(), request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        T::chat_capability(self.as_ref(), model, capability)
    }
}

impl<T> ChatProvider for Arc<T>
where
    T: ChatProvider + ?Sized,
{
    type Stream = T::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        T::chat(self.as_ref(), request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        T::chat_stream(self.as_ref(), request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        T::chat_capability(self.as_ref(), model, capability)
    }
}

/// A type-erased chat provider for dynamic dispatch.
///
/// Wraps any `T: ChatProvider + 'static` behind a vtable, boxing the async method futures.
/// Use this wherever you previously used `Box<dyn ChatProvider>` or `&dyn ChatProvider`.
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use anyllm::{ChatProvider, ChatRequest, DynChatProvider};
/// # fn example(owned_provider: impl ChatProvider + 'static, shared_provider: Arc<impl ChatProvider + 'static>, request: ChatRequest) {
/// let provider: DynChatProvider = DynChatProvider::new(owned_provider);
/// let shared: DynChatProvider = shared_provider.into();
/// // let response = provider.chat(&request).await?;
/// # }
/// ```
#[derive(Clone)]
pub struct DynChatProvider(Arc<dyn ChatProviderErased>);

impl DynChatProvider {
    /// Erase a concrete provider into a `DynChatProvider`.
    #[must_use]
    pub fn new<T>(provider: T) -> Self
    where
        T: ChatProvider + 'static,
        T::Stream: 'static,
    {
        Self(Arc::new(provider))
    }
}

impl<T> From<Arc<T>> for DynChatProvider
where
    T: ChatProvider + 'static,
    T::Stream: 'static,
{
    fn from(provider: Arc<T>) -> Self {
        Self(provider)
    }
}

impl std::fmt::Debug for DynChatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynChatProvider")
            .field("provider", &self.0.provider_name())
            .finish()
    }
}

impl ProviderIdentity for DynChatProvider {
    fn provider_name(&self) -> &'static str {
        self.0.provider_name()
    }
}

impl ChatProvider for DynChatProvider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.0.chat_erased(request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        self.0.chat_stream_erased(request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.0.chat_capability_erased(model, capability)
    }
}

/// Object-safe internal trait that manually boxes the async method futures.
///
/// This is sealed: only the blanket impl for `T: ChatProvider` exists.
/// Consumers interact with [`DynChatProvider`] instead.
trait ChatProviderErased: ProviderIdentity {
    fn chat_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>;

    fn chat_stream_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatStream>> + Send + 'a>>;

    fn chat_capability_erased(&self, model: &str, capability: ChatCapability) -> CapabilitySupport;
}

impl<T> ChatProviderErased for T
where
    T: ChatProvider,
    T::Stream: 'static,
{
    fn chat_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(ChatProvider::chat(self, request))
    }

    fn chat_stream_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatStream>> + Send + 'a>> {
        Box::pin(async move {
            let stream = ChatProvider::chat_stream(self, request).await?;
            Ok(Box::pin(stream) as ChatStream)
        })
    }

    fn chat_capability_erased(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        ChatProvider::chat_capability(self, model, capability)
    }
}

/// Convenience extension methods for [`ChatProvider`] implementors.
pub trait ChatProviderExt: ChatProvider {
    /// Quick one-shot text question.
    ///
    /// # Errors
    ///
    /// Propagates any [`Error`](crate::Error) from the underlying
    /// [`chat`](ChatProvider::chat) call, and returns
    /// [`Error::UnexpectedResponse`](crate::Error::UnexpectedResponse) if the
    /// provider response contains no text content.
    fn ask(
        &self,
        model: &str,
        message: impl Into<String>,
    ) -> impl Future<Output = Result<String>> + Send {
        let message = message.into();

        async move {
            let response = self.chat(&ChatRequest::new(model).user(message)).await?;

            response.text().ok_or_else(|| {
                crate::Error::UnexpectedResponse(format!(
                    "provider '{}' returned no text content for ask()",
                    self.provider_name()
                ))
            })
        }
    }

    /// Quick one-shot text question with a system message.
    ///
    /// # Errors
    ///
    /// Propagates any [`Error`](crate::Error) from the underlying
    /// [`chat`](ChatProvider::chat) call, and returns
    /// [`Error::UnexpectedResponse`](crate::Error::UnexpectedResponse) if the
    /// provider response contains no text content.
    fn ask_with_system(
        &self,
        model: &str,
        system: impl Into<String>,
        message: impl Into<String>,
    ) -> impl Future<Output = Result<String>> + Send {
        let system = system.into();
        let message = message.into();

        async move {
            let response = self
                .chat(&ChatRequest::new(model).system(system).user(message))
                .await?;

            response.text().ok_or_else(|| {
                crate::Error::UnexpectedResponse(format!(
                    "provider '{}' returned no text content for ask_with_system()",
                    self.provider_name()
                ))
            })
        }
    }
}

impl<T: ChatProvider> ChatProviderExt for T {}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::{
        ChatResponseBuilder, ChatStreamExt, ContentBlock, Error, FinishReason, Message,
        MockProvider, ResponseMetadata, SingleResponseStream,
    };

    #[derive(Debug)]
    struct DefaultOnlyProvider {
        response: Mutex<ChatResponse>,
    }

    impl DefaultOnlyProvider {
        fn new(response: ChatResponse) -> Self {
            Self {
                response: Mutex::new(response),
            }
        }
    }

    impl ProviderIdentity for DefaultOnlyProvider {}

    impl ChatProvider for DefaultOnlyProvider {
        type Stream = SingleResponseStream;

        async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(self.response.lock().unwrap().clone())
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            Ok(self.chat(request).await?.into())
        }
    }

    #[test]
    fn trait_default_methods_return_defaults() {
        let model = DefaultOnlyProvider::new(ChatResponseBuilder::new().text("test").build());

        assert_eq!(
            model.chat_capability("demo", ChatCapability::ToolCalls),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            model.chat_capability("demo", ChatCapability::Streaming),
            CapabilitySupport::Unknown
        );
        assert_eq!(model.provider_name(), "unknown");
    }

    #[tokio::test]
    async fn dyn_provider_from_concrete() {
        let model = DynChatProvider::new(
            MockProvider::with_response(ChatResponseBuilder::new().text("from dyn").build())
                .with_chat_capability(
                    ChatCapability::ReasoningOutput,
                    CapabilitySupport::Supported,
                )
                .with_provider_name("dyn-mock"),
        );
        let request = ChatRequest::new("mock-model").message(Message::user("hi"));
        let response = model.chat(&request).await.unwrap();
        assert_eq!(response.text(), Some("from dyn".to_string()));
        assert_eq!(
            model.chat_capability("mock-model", ChatCapability::ReasoningOutput),
            CapabilitySupport::Supported
        );
        assert_eq!(model.provider_name(), "dyn-mock");
    }

    #[tokio::test]
    async fn dyn_provider_from_arc() {
        let inner = Arc::new(
            MockProvider::with_response(ChatResponseBuilder::new().text("from arc").build())
                .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
                .with_provider_name("arc-mock"),
        );
        let model: DynChatProvider = inner.into();
        let request = ChatRequest::new("mock-model").message(Message::user("hi"));
        let response = model.chat(&request).await.unwrap();
        assert_eq!(response.text(), Some("from arc".to_string()));
        assert_eq!(
            model.chat_capability("mock-model", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(model.provider_name(), "arc-mock");
    }

    #[tokio::test]
    async fn dyn_provider_is_cloneable() {
        let model = DynChatProvider::new(
            MockProvider::with_response(ChatResponseBuilder::new().text("from clone").build())
                .with_provider_name("clone-mock"),
        );
        let cloned = model.clone();
        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        let response = cloned.chat(&request).await.unwrap();
        assert_eq!(response.text(), Some("from clone".to_string()));
        assert_eq!(cloned.provider_name(), "clone-mock");
    }

    #[tokio::test]
    async fn forwarding_impls_delegate_to_inner_provider() {
        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        let borrowed_inner =
            MockProvider::with_response(ChatResponseBuilder::new().text("from ref").build())
                .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
                .with_provider_name("ref-mock");
        let borrowed = &borrowed_inner;
        let borrowed_response = borrowed.chat(&request).await.unwrap();
        assert_eq!(borrowed_response.text(), Some("from ref".to_string()));
        assert_eq!(
            borrowed.chat_capability("mock-model", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(borrowed.provider_name(), "ref-mock");

        let boxed = Box::new(
            MockProvider::with_response(ChatResponseBuilder::new().text("from box").build())
                .with_chat_capability(
                    ChatCapability::ReasoningOutput,
                    CapabilitySupport::Supported,
                )
                .with_provider_name("box-mock"),
        );
        let boxed_response = boxed.chat(&request).await.unwrap();
        assert_eq!(boxed_response.text(), Some("from box".to_string()));
        assert_eq!(
            boxed.chat_capability("mock-model", ChatCapability::ReasoningOutput),
            CapabilitySupport::Supported
        );
        assert_eq!(boxed.provider_name(), "box-mock");

        let shared = Arc::new(
            MockProvider::with_response(ChatResponseBuilder::new().text("from arc").build())
                .with_chat_capability(ChatCapability::Streaming, CapabilitySupport::Supported)
                .with_provider_name("arc-forward-mock"),
        );
        let shared_response = shared.chat(&request).await.unwrap();
        assert_eq!(shared_response.text(), Some("from arc".to_string()));
        assert_eq!(
            shared.chat_capability("mock-model", ChatCapability::Streaming),
            CapabilitySupport::Supported
        );
        assert_eq!(shared.provider_name(), "arc-forward-mock");
    }

    #[tokio::test]
    async fn ask_returns_text() {
        let model = MockProvider::with_response(ChatResponseBuilder::new().text("world").build());
        let result = model.ask("mock-model", "hello").await.unwrap();

        assert_eq!(result, "world");

        let requests = model.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "mock-model");
        assert_eq!(requests[0].messages, vec![Message::user("hello")]);
    }

    #[tokio::test]
    async fn ask_errors_for_no_text() {
        let response = ChatResponse {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".to_string(),
                name: "search".to_string(),
                arguments: "{}".to_string(),
            }],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
            model: None,
            id: None,
            metadata: ResponseMetadata::new(),
        };
        let model = MockProvider::with_response(response);
        let err = model
            .ask("mock-model", "search something")
            .await
            .unwrap_err();
        match err {
            Error::UnexpectedResponse(message) => {
                assert_eq!(
                    message,
                    "provider 'mock' returned no text content for ask()"
                )
            }
            other => panic!("expected unexpected response error, got {other:?}"),
        }

        let requests = model.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].messages,
            vec![Message::user("search something")]
        );
    }

    #[tokio::test]
    async fn ask_propagates_chat_errors() {
        let model = MockProvider::with_error(Error::Timeout("slow".to_string()));

        match model.ask("mock-model", "hello").await {
            Err(Error::Timeout(message)) => assert_eq!(message, "slow"),
            Err(other) => panic!("expected timeout error, got {other:?}"),
            Ok(_) => panic!("expected ask to return an error"),
        }
    }

    #[tokio::test]
    async fn ask_with_system_errors_for_no_text() {
        let response = ChatResponse {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".to_string(),
                name: "search".to_string(),
                arguments: "{}".to_string(),
            }],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
            model: None,
            id: None,
            metadata: ResponseMetadata::new(),
        };
        let model = MockProvider::with_response(response);
        let err = model
            .ask_with_system("mock-model", "You are helpful", "search something")
            .await
            .unwrap_err();

        match err {
            Error::UnexpectedResponse(message) => {
                assert_eq!(
                    message,
                    "provider 'mock' returned no text content for ask_with_system()"
                )
            }
            other => panic!("expected unexpected response error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn ask_with_system_propagates_chat_errors() {
        let model = MockProvider::with_error(Error::Timeout("slow".to_string()));

        match model
            .ask_with_system("mock-model", "You are helpful", "hello")
            .await
        {
            Err(Error::Timeout(message)) => assert_eq!(message, "slow"),
            Err(other) => panic!("expected timeout error, got {other:?}"),
            Ok(_) => panic!("expected ask_with_system to return an error"),
        }
    }

    #[tokio::test]
    async fn ask_with_system_returns_text() {
        let model = MockProvider::with_response(
            ChatResponseBuilder::new().text("helpful response").build(),
        );
        let result = model
            .ask_with_system("mock-model", "You are helpful", "help me")
            .await
            .unwrap();
        assert_eq!(result, "helpful response");

        let requests = model.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "mock-model");
        assert_eq!(
            requests[0].messages,
            vec![Message::system("You are helpful"), Message::user("help me")]
        );
    }

    #[tokio::test]
    async fn dyn_provider_chat_stream_erases_stream_type() {
        let model = DynChatProvider::new(
            MockProvider::with_response(
                ChatResponseBuilder::new()
                    .reasoning("thinking")
                    .text("from dyn stream")
                    .usage(10, 5)
                    .model("mock")
                    .id("resp_dyn_stream")
                    .build(),
            )
            .with_provider_name("dyn-stream-mock"),
        );
        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        let response = model
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.reasoning_text(), Some("thinking".to_string()));
        assert_eq!(response.text(), Some("from dyn stream".to_string()));
        assert_eq!(response.model.as_deref(), Some("mock"));
        assert_eq!(response.id.as_deref(), Some("resp_dyn_stream"));
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(5));
    }

    #[tokio::test]
    async fn dyn_provider_chat_stream_propagates_errors() {
        let model = DynChatProvider::new(MockProvider::with_error(Error::Timeout("slow".into())));
        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        match model.chat_stream(&request).await {
            Err(Error::Timeout(message)) => assert_eq!(message, "slow"),
            Err(other) => panic!("expected timeout error, got {other:?}"),
            Ok(_) => panic!("expected chat_stream to return an error"),
        }
    }

    #[test]
    fn dyn_provider_debug_includes_provider_name() {
        let model = DynChatProvider::new(
            MockProvider::with_response(ChatResponseBuilder::new().text("debug").build())
                .with_provider_name("debug-mock"),
        );

        let debug = format!("{model:?}");
        assert!(debug.contains("DynChatProvider"));
        assert!(debug.contains("debug-mock"));
    }

    #[test]
    fn closure_chat_capability_resolver_returns_custom_answer() {
        let resolver = |model: &str, capability| {
            if model == "demo" && capability == ChatCapability::StructuredOutput {
                Some(CapabilitySupport::Unknown)
            } else {
                None
            }
        };

        assert_eq!(
            resolver.chat_capability("demo", ChatCapability::StructuredOutput),
            Some(CapabilitySupport::Unknown)
        );
        assert_eq!(
            resolver.chat_capability("demo", ChatCapability::ToolCalls),
            None
        );
    }
}
