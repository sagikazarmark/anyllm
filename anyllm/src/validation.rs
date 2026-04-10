use std::fmt;

use crate::{
    ChatCapability, ChatProvider, ChatRequest, ChatResponse, ContentBlock, ContentPart, Error,
    Message, ResponseFormat, Result, ToolChoice, UserContent,
};

/// Controls how [`ValidatingChatProvider`] handles request/provider mismatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Reject unsupported or inconsistent requests before dispatch.
    #[default]
    Strict,
    /// Emit warnings and continue dispatch.
    Permissive,
}

/// A [`ChatProvider`] wrapper that validates requests against provider capabilities.
///
/// This wrapper is intentionally opt-in so callers can choose between fail-fast
/// validation and direct provider dispatch. If you also use
/// [`TracingChatProvider`](crate::TracingChatProvider), wrap the validating
/// provider inside tracing so permissive warnings land on the request span.
///
/// ```rust,no_run
/// use anyllm::prelude::*;
///
/// struct StaticProvider;
///
/// impl ChatProvider for StaticProvider {
///     type Stream = SingleResponseStream;
///
///     async fn chat(&self, request: &ChatRequest) -> anyllm::Result<ChatResponse> {
///         Ok(ChatResponse::new(vec![ContentBlock::Text {
///             text: format!("validated response for {}", request.model),
///         }])
///         .finish_reason(FinishReason::Stop)
///         .model(request.model.clone()))
///     }
///
///     async fn chat_stream(&self, request: &ChatRequest) -> anyllm::Result<Self::Stream> {
///         Ok(self.chat(request).await?.into_stream())
///     }
///
///     fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
///         match capability {
///             ChatCapability::Streaming => CapabilitySupport::Supported,
///             _ => CapabilitySupport::Unknown,
///         }
///     }
/// }
///
/// # async fn example() -> anyllm::Result<()> {
/// let provider = ValidatingChatProvider::new(StaticProvider)
///     .with_mode(ValidationMode::Strict);
/// let request = ChatRequest::new("demo-model").user("Say hello");
/// let response = provider.chat(&request).await?;
/// assert_eq!(response.text().as_deref(), Some("validated response for demo-model"));
/// # Ok(())
/// # }
/// ```
pub struct ValidatingChatProvider<T> {
    inner: T,
    mode: ValidationMode,
}

impl<T> ValidatingChatProvider<T> {
    /// Wrap a provider with strict request validation enabled.
    #[must_use]
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            mode: ValidationMode::Strict,
        }
    }

    /// Set how validation issues should be handled.
    #[must_use]
    pub fn with_mode(mut self, mode: ValidationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Return the configured validation mode.
    #[must_use]
    pub fn mode(&self) -> ValidationMode {
        self.mode
    }

    /// Consume the wrapper and return the wrapped provider.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Consume the wrapper and return the provider plus validation mode.
    #[must_use]
    pub fn into_parts(self) -> (T, ValidationMode) {
        (self.inner, self.mode)
    }
}

impl<T> fmt::Debug for ValidatingChatProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidatingChatProvider")
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl<T> ChatProvider for ValidatingChatProvider<T>
where
    T: ChatProvider,
    T::Stream: 'static,
{
    type Stream = T::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.validate(request, ValidationTarget::Chat)?;
        self.inner.chat(request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        self.validate(request, ValidationTarget::ChatStream)?;
        self.inner.chat_stream(request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> crate::CapabilitySupport {
        self.inner.chat_capability(model, capability)
    }

    fn provider_name(&self) -> &'static str {
        self.inner.provider_name()
    }
}

#[cfg(feature = "extract")]
impl<T> crate::ExtractExt for ValidatingChatProvider<T>
where
    T: ChatProvider + Sync,
    T::Stream: 'static,
{
}

impl<T> ValidatingChatProvider<T>
where
    T: ChatProvider,
{
    fn validate(&self, request: &ChatRequest, target: ValidationTarget) -> Result<()> {
        let issues = validate_request(&self.inner, request, target);
        if issues.is_empty() {
            return Ok(());
        }

        match self.mode {
            ValidationMode::Strict => Err(issues[0].clone().into_error()),
            ValidationMode::Permissive => {
                for issue in issues {
                    log_validation_warning(self.inner.provider_name(), &issue);
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationTarget {
    Chat,
    ChatStream,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationIssueKind {
    Unsupported,
    InvalidRequest,
}

impl ValidationIssueKind {
    #[cfg(feature = "tracing")]
    fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::InvalidRequest => "invalid_request",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValidationIssue {
    kind: ValidationIssueKind,
    message: String,
}

impl ValidationIssue {
    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: ValidationIssueKind::Unsupported,
            message: message.into(),
        }
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            kind: ValidationIssueKind::InvalidRequest,
            message: message.into(),
        }
    }

    fn into_error(self) -> Error {
        match self.kind {
            ValidationIssueKind::Unsupported => Error::Unsupported(self.message),
            ValidationIssueKind::InvalidRequest => Error::InvalidRequest(self.message),
        }
    }
}

fn validate_request(
    provider: &(impl ChatProvider + ?Sized),
    request: &ChatRequest,
    target: ValidationTarget,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let provider_name = provider.provider_name();
    let model = request.model.as_str();

    if matches!(target, ValidationTarget::ChatStream)
        && is_unsupported(provider, model, ChatCapability::Streaming)
    {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support streaming chat completions"
        )));
    }

    if request
        .response_format
        .as_ref()
        .is_some_and(requires_structured_output)
        && is_unsupported(provider, model, ChatCapability::StructuredOutput)
    {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support structured response formats"
        )));
    }

    if request.parallel_tool_calls == Some(true)
        && is_unsupported(provider, model, ChatCapability::ParallelToolCalls)
    {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support parallel tool calls"
        )));
    }

    if let Some(ToolChoice::Specific { name }) = &request.tool_choice {
        let has_named_tool = request
            .tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|tool| tool.name == *name));
        if !has_named_tool {
            issues.push(ValidationIssue::invalid_request(format!(
                "tool_choice requested tool '{name}', but that tool is not present in request.tools"
            )));
        }
    }

    if request_uses_tool_calls(request)
        && is_unsupported(provider, model, ChatCapability::ToolCalls)
    {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support tool calls"
        )));
    }

    if request.reasoning.is_some()
        && is_unsupported(provider, model, ChatCapability::ReasoningConfig)
    {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support reasoning configuration"
        )));
    }

    for message in &request.messages {
        match message {
            Message::User { content, .. } => {
                validate_user_content(provider_name, provider, model, content, &mut issues)
            }
            Message::Assistant { content, .. } => {
                validate_assistant_content(provider_name, provider, model, content, &mut issues)
            }
            Message::System { .. } | Message::Tool { .. } => {}
        }
    }

    issues
}

fn is_unsupported(
    provider: &(impl ChatProvider + ?Sized),
    model: &str,
    capability: ChatCapability,
) -> bool {
    provider.chat_capability(model, capability) == crate::CapabilitySupport::Unsupported
}

fn requires_structured_output(format: &ResponseFormat) -> bool {
    !matches!(format, ResponseFormat::Text)
}

fn request_uses_tool_calls(request: &ChatRequest) -> bool {
    request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        || request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| !matches!(choice, ToolChoice::Disabled))
        || request
            .messages
            .iter()
            .any(|message| matches!(message, Message::Tool { .. }))
}

fn validate_user_content(
    provider_name: &str,
    provider: &(impl ChatProvider + ?Sized),
    model: &str,
    content: &UserContent,
    issues: &mut Vec<ValidationIssue>,
) {
    let UserContent::Parts(parts) = content else {
        return;
    };

    let has_image = parts
        .iter()
        .any(|part| matches!(part, ContentPart::Image { .. }));
    if has_image && is_unsupported(provider, model, ChatCapability::ImageInput) {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support image input"
        )));
    }

    let has_image_detail = parts.iter().any(|part| {
        matches!(
            part,
            ContentPart::Image {
                detail: Some(_),
                ..
            }
        )
    });
    if has_image_detail && is_unsupported(provider, model, ChatCapability::ImageDetail) {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support image detail hints"
        )));
    }
}

fn validate_assistant_content(
    provider_name: &str,
    provider: &(impl ChatProvider + ?Sized),
    model: &str,
    content: &[ContentBlock],
    issues: &mut Vec<ValidationIssue>,
) {
    let has_image = content
        .iter()
        .any(|block| matches!(block, ContentBlock::Image { .. }));
    if has_image && is_unsupported(provider, model, ChatCapability::ImageReplay) {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support replaying assistant image blocks"
        )));
    }

    let has_reasoning = content
        .iter()
        .any(|block| matches!(block, ContentBlock::Reasoning { .. }));
    if has_reasoning && is_unsupported(provider, model, ChatCapability::ReasoningReplay) {
        issues.push(ValidationIssue::unsupported(format!(
            "provider '{provider_name}' does not support replaying assistant reasoning blocks"
        )));
    }
}

#[cfg(feature = "tracing")]
fn log_validation_warning(provider_name: &str, issue: &ValidationIssue) {
    tracing::warn!(
        provider = provider_name,
        validation_kind = issue.kind.as_str(),
        validation_message = %issue.message,
        "request validation warning"
    );
}

#[cfg(not(feature = "tracing"))]
fn log_validation_warning(_provider_name: &str, _issue: &ValidationIssue) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapabilitySupport, ChatCapability, ChatResponseBuilder, ChatStreamExt, ImageSource,
        MockProvider, MockStreamingProvider, ResponseFormat, Tool,
    };
    use serde_json::json;

    #[test]
    fn validating_provider_defaults_to_strict_mode() {
        let provider = ValidatingChatProvider::new(MockProvider::with_text("ok"));

        assert_eq!(provider.mode(), ValidationMode::Strict);
        assert_eq!(provider.provider_name(), "mock");
    }

    #[test]
    fn validating_provider_supports_ownership_recovery() {
        let provider = ValidatingChatProvider::new(MockProvider::with_text("ok"))
            .with_mode(ValidationMode::Permissive);

        let (inner, mode) = provider.into_parts();

        assert_eq!(mode, ValidationMode::Permissive);
        assert_eq!(inner.provider_name(), "mock");
    }

    #[tokio::test]
    async fn strict_mode_blocks_unsupported_structured_output_before_dispatch() {
        let inner = MockProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            );
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .response_format(ResponseFormat::Json),
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, Error::Unsupported(message) if message.contains("structured response formats"))
        );
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_allows_supported_structured_output() {
        let inner = MockProvider::with_response(ChatResponseBuilder::new().text("ok").build())
            .with_provider_name("fixture")
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            );
        let provider = ValidatingChatProvider::new(inner.clone());

        let response = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .response_format(ResponseFormat::Json),
            )
            .await
            .unwrap();

        assert_eq!(response.text().as_deref(), Some("ok"));
        assert_eq!(inner.call_count(), 1);
    }

    #[tokio::test]
    async fn strict_mode_blocks_invalid_specific_tool_choice_before_dispatch() {
        let inner = MockProvider::with_text("ok").with_provider_name("fixture");
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .tool_choice(ToolChoice::Specific {
                        name: "missing_tool".into(),
                    }),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidRequest(message) if message.contains("missing_tool")));
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_blocks_unsupported_tool_definitions_before_dispatch() {
        let inner = MockProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .tools([Tool::new("search", json!({"type": "object"}))]),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Unsupported(message) if message.contains("tool calls")));
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_blocks_unsupported_tool_choice_before_dispatch() {
        let inner = MockProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .tools([Tool::new("search", json!({"type": "object"}))])
                    .tool_choice(ToolChoice::Required),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Unsupported(message) if message.contains("tool calls")));
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_blocks_tool_result_replay_when_tool_calls_unsupported() {
        let inner = MockProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .message(Message::tool_result("call_1", "search", "done")),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Unsupported(message) if message.contains("tool calls")));
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_blocks_unsupported_user_image_input_before_dispatch() {
        let inner = MockProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(ChatCapability::ImageInput, CapabilitySupport::Unsupported);
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = provider
            .chat(
                &ChatRequest::new("test").message(Message::user_multimodal(vec![
                    ContentPart::image_url("https://example.com/cat.png"),
                ])),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Unsupported(message) if message.contains("image input")));
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_blocks_assistant_replay_for_unsupported_blocks() {
        let inner = MockProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(
                ChatCapability::ReasoningReplay,
                CapabilitySupport::Unsupported,
            )
            .with_chat_capability(ChatCapability::ImageReplay, CapabilitySupport::Unsupported);
        let provider = ValidatingChatProvider::new(inner.clone());

        let request = ChatRequest::new("test")
            .message(Message::Assistant {
                content: vec![ContentBlock::Reasoning {
                    text: "thinking".into(),
                    signature: None,
                }],
                name: None,
                extensions: None,
            })
            .message(Message::Assistant {
                content: vec![ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/output.png".into(),
                    },
                }],
                name: None,
                extensions: None,
            });

        let err = provider.chat(&request).await.unwrap_err();

        assert!(
            matches!(err, Error::Unsupported(message) if message.contains("reasoning blocks") || message.contains("image blocks"))
        );
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_allows_supported_assistant_replay_blocks() {
        let inner = MockProvider::with_response(ChatResponseBuilder::new().text("ok").build())
            .with_provider_name("fixture")
            .with_supported_chat_capabilities([
                ChatCapability::ReasoningReplay,
                ChatCapability::ImageReplay,
            ]);
        let provider = ValidatingChatProvider::new(inner.clone());

        let request = ChatRequest::new("test")
            .message(Message::Assistant {
                content: vec![ContentBlock::Reasoning {
                    text: "thinking".into(),
                    signature: None,
                }],
                name: None,
                extensions: None,
            })
            .message(Message::Assistant {
                content: vec![ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/output.png".into(),
                    },
                }],
                name: None,
                extensions: None,
            });

        let response = provider.chat(&request).await.unwrap();

        assert_eq!(response.text().as_deref(), Some("ok"));
        assert_eq!(inner.call_count(), 1);
    }

    #[tokio::test]
    async fn strict_mode_validates_stream_requests_before_dispatch() {
        let inner = MockStreamingProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(ChatCapability::ImageInput, CapabilitySupport::Unsupported)
            .with_chat_capability(ChatCapability::ImageDetail, CapabilitySupport::Unsupported);
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = match provider
            .chat_stream(
                &ChatRequest::new("test").message(Message::user_multimodal(vec![
                    ContentPart::Image {
                        source: ImageSource::Url {
                            url: "https://example.com/cat.png".into(),
                        },
                        detail: Some("high".into()),
                    },
                ])),
            )
            .await
        {
            Ok(_) => panic!("expected validation failure"),
            Err(err) => err,
        };

        assert!(matches!(err, Error::Unsupported(_)));
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn strict_mode_blocks_unsupported_streaming_before_dispatch() {
        let inner = MockStreamingProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(ChatCapability::Streaming, CapabilitySupport::Unsupported);
        let provider = ValidatingChatProvider::new(inner.clone());

        let err = match provider
            .chat_stream(&ChatRequest::new("test").message(Message::user("hi")))
            .await
        {
            Ok(_) => panic!("expected validation failure"),
            Err(err) => err,
        };

        assert!(matches!(err, Error::Unsupported(message) if message.contains("streaming")));
        assert_eq!(inner.call_count(), 0);
    }

    #[tokio::test]
    async fn permissive_mode_allows_dispatch() {
        let inner = MockProvider::with_response(ChatResponseBuilder::new().text("ok").build())
            .with_provider_name("fixture")
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            );
        let provider =
            ValidatingChatProvider::new(inner.clone()).with_mode(ValidationMode::Permissive);

        let response = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .response_format(ResponseFormat::Json),
            )
            .await
            .unwrap();

        assert_eq!(response.text().as_deref(), Some("ok"));
        assert_eq!(inner.call_count(), 1);
    }

    #[tokio::test]
    async fn permissive_mode_allows_stream_dispatch_when_streaming_is_unsupported() {
        let inner = MockStreamingProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(ChatCapability::Streaming, CapabilitySupport::Unsupported);
        let provider =
            ValidatingChatProvider::new(inner.clone()).with_mode(ValidationMode::Permissive);

        let response = provider
            .chat_stream(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();

        assert_eq!(response.text().as_deref(), Some("ok"));
        assert_eq!(inner.call_count(), 1);
    }

    #[cfg(feature = "tracing")]
    #[tokio::test]
    async fn permissive_mode_emits_tracing_warning() {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        use tracing::field::{Field, Visit};
        use tracing_subscriber::Registry;
        use tracing_subscriber::layer::SubscriberExt;

        #[derive(Default)]
        struct EventRecorder {
            fields: HashMap<String, String>,
        }

        impl Visit for EventRecorder {
            fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
                self.fields
                    .insert(field.name().to_string(), format!("{value:?}"));
            }

            fn record_str(&mut self, field: &Field, value: &str) {
                self.fields
                    .insert(field.name().to_string(), value.to_string());
            }
        }

        struct WarningRecorder(Arc<Mutex<Vec<HashMap<String, String>>>>);

        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for WarningRecorder {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                let mut visitor = EventRecorder::default();
                event.record(&mut visitor);
                self.0.lock().unwrap().push(visitor.fields);
            }
        }

        let recorded = Arc::new(Mutex::new(Vec::<HashMap<String, String>>::new()));
        let subscriber = Registry::default().with(WarningRecorder(recorded.clone()));
        let guard = tracing::subscriber::set_default(subscriber);

        let inner = MockProvider::with_text("ok")
            .with_provider_name("fixture")
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            );
        let provider = ValidatingChatProvider::new(inner).with_mode(ValidationMode::Permissive);

        let _ = provider
            .chat(
                &ChatRequest::new("test")
                    .message(Message::user("hi"))
                    .response_format(ResponseFormat::Json),
            )
            .await
            .unwrap();

        drop(guard);

        let events = recorded.lock().unwrap();
        assert!(events.iter().any(|fields| {
            fields.get("message") == Some(&"request validation warning".to_string())
                && fields.get("provider") == Some(&"fixture".to_string())
                && fields.get("validation_kind") == Some(&"unsupported".to_string())
        }));
    }
}
