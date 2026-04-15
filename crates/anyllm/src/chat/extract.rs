use std::fmt;
use std::future::Future;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, DynChatProvider,
    Message, ProviderIdentity, ResponseFormat, Tool, ToolChoice, UserContent,
};

/// Extension trait for providers that support structured data extraction.
///
/// Provides [`extract`](ExtractExt::extract), which returns the parsed value,
/// extraction-pass [`ChatResponse`], and extraction metadata as [`Extracted<T>`].
pub trait ExtractExt: ChatProvider {
    /// Extract structured data from a chat request.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Extract`](crate::Error::Extract) on extraction failures
    /// (unsupported mode, parse error, tool conflict) or propagates provider errors.
    fn extract<'a, T>(
        &'a self,
        request: &'a ChatRequest,
    ) -> impl Future<Output = crate::Result<Extracted<T>>> + Send + 'a
    where
        T: JsonSchema + DeserializeOwned + Send + 'a,
    {
        default_extract_response(self, request)
    }
}

/// Controls how structured data is extracted from a provider response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExtractionMode {
    /// Automatically select the best strategy based on provider capabilities.
    /// Prefers `Native` when structured output is explicitly supported,
    /// otherwise falls back to `ForcedTool` when tool calls are explicitly
    /// supported.
    #[default]
    Auto,
    /// Use the provider's built-in structured output / JSON mode.
    Native,
    /// Inject a synthetic tool and force the provider to call it,
    /// extracting the structured data from the tool call arguments.
    ForcedTool,
}

/// The result of a structured extraction: a typed value, the underlying
/// provider response, and metadata about how the extraction was performed.
#[derive(Debug, Clone)]
pub struct Extracted<T> {
    /// Parsed structured value
    pub value: T,
    /// Raw provider response used for extraction
    pub response: ChatResponse,
    /// Extraction execution metadata
    pub metadata: ExtractionMetadata,
}

/// Metadata about how a structured extraction was performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionMetadata {
    /// Number of LLM calls made during extraction.
    pub passes: u8,
    /// Whether the raw output required repair (e.g., JSON fixup) before parsing.
    pub repaired: bool,
}

/// Input for a dedicated structured extraction pass.
///
/// This intentionally keeps only the completed conversation state needed for a
/// dedicated extraction call, rather than the full original [`ChatRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionRequest {
    /// Provider-specific model identifier
    pub model: String,
    /// Conversation messages used for the extraction pass
    pub messages: Vec<Message>,
}

impl ExtractionRequest {
    /// Create a new extraction request for the given model.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
        }
    }

    /// Replace all messages in the extraction request.
    #[must_use]
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Append a single message to the extraction request.
    #[must_use]
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Shorthand for `.message(Message::system(content))`.
    #[must_use]
    pub fn system(self, content: impl Into<String>) -> Self {
        self.message(Message::system(content))
    }

    /// Shorthand for `.message(Message::user(content))`.
    #[must_use]
    pub fn user(self, content: impl Into<UserContent>) -> Self {
        self.message(Message::user(content))
    }

    /// Shorthand for `.message(Message::assistant(content))`.
    #[must_use]
    pub fn assistant(self, content: impl Into<String>) -> Self {
        self.message(Message::assistant(content))
    }
}

impl From<&ChatRequest> for ExtractionRequest {
    fn from(request: &ChatRequest) -> Self {
        Self {
            model: request.model.clone(),
            messages: request.messages.clone(),
        }
    }
}

impl From<ExtractionRequest> for ChatRequest {
    fn from(request: ExtractionRequest) -> Self {
        ChatRequest::new(request.model).messages(request.messages)
    }
}

/// Orchestrates structured extraction from LLM providers.
///
/// Use `Extractor` when you want a dedicated extraction call over an existing
/// conversation transcript instead of extracting directly from the original
/// [`ChatRequest`]. This is useful when the source request still carries caller
/// tools, response-format settings, or other concerns that should not be sent
/// on the extraction call itself.
///
/// `Extractor` does not execute tools or orchestrate an earlier generation
/// phase. It assumes the messages you pass in already contain the completed
/// context to extract from. For simple one-shot extraction, use
/// [`ExtractExt::extract`] directly on the provider.
///
/// `Extractor` builds a clean extraction request from an
/// [`ExtractionRequest`] and delegates to the provider's extraction
/// capabilities (native structured output when available, forced-tool
/// fallback otherwise).
#[derive(Debug, Clone, Copy, Default)]
pub struct Extractor;

impl Extractor {
    /// Creates a new extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract a typed value using an extraction-specific request.
    ///
    /// Accepts either an explicit [`ExtractionRequest`] or a borrowed
    /// [`ChatRequest`], which is projected to the extraction-relevant model and
    /// messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Extract`](crate::Error::Extract) if the provider does
    /// not support the required extraction mode, if tool conflicts prevent
    /// extraction, or if the response cannot be parsed into `T`. Also propagates
    /// any provider communication errors from the underlying chat call.
    pub async fn extract<P, R, T>(&self, provider: &P, request: R) -> crate::Result<Extracted<T>>
    where
        P: ChatProvider + ?Sized,
        R: Into<ExtractionRequest>,
        T: JsonSchema + DeserializeOwned + Send,
    {
        default_extract_response(provider, &(request.into().into())).await
    }
}

/// A [`ChatProvider`] wrapper that adds orchestrated structured extraction
/// via [`ExtractExt`] using a bundled [`Extractor`].
///
/// Wrapping a provider with `ExtractingProvider` changes the behavior of
/// [`ExtractExt::extract`]: instead of extracting directly from the request,
/// it builds a clean extraction request (stripping tools, response format,
/// etc.) and performs extraction in a dedicated pass over the request's
/// existing messages.
///
/// This wrapper does not execute tools or run a separate pre-extraction
/// conversation step. Use it when the original request still carries
/// non-extraction settings that should be ignored for the extraction call.
///
/// For simple one-shot extraction without this dedicated pass, use
/// [`ExtractExt::extract`] directly on the unwrapped provider.
pub struct ExtractingProvider<P> {
    inner: P,
    extractor: Extractor,
}

impl<P> ExtractingProvider<P> {
    /// Wraps a provider with orchestrated extraction.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            extractor: Extractor::new(),
        }
    }

    /// Consumes the wrapper, returning the wrapped provider.
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P> ProviderIdentity for ExtractingProvider<P>
where
    P: ProviderIdentity,
{
    fn provider_name(&self) -> &'static str {
        self.inner.provider_name()
    }
}

impl<P> ChatProvider for ExtractingProvider<P>
where
    P: ChatProvider,
{
    type Stream = P::Stream;

    async fn chat(&self, request: &ChatRequest) -> crate::Result<ChatResponse> {
        self.inner.chat(request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> crate::Result<Self::Stream> {
        self.inner.chat_stream(request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner.chat_capability(model, capability)
    }
}

impl<P> ExtractExt for ExtractingProvider<P>
where
    P: ChatProvider + Sync,
{
    fn extract<'a, T>(
        &'a self,
        request: &'a ChatRequest,
    ) -> impl Future<Output = crate::Result<Extracted<T>>> + Send + 'a
    where
        T: JsonSchema + DeserializeOwned + Send + 'a,
    {
        self.extractor.extract(&self.inner, request)
    }
}

impl ExtractExt for DynChatProvider {}

impl<P> fmt::Debug for ExtractingProvider<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtractingProvider").finish_non_exhaustive()
    }
}

/// Error produced while extracting structured output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExtractError {
    /// Structured text output was missing
    MissingStructuredText {
        /// Extraction mode that was attempted
        mode: ExtractionMode,
        /// Provider name that produced the response
        provider: String,
    },
    /// Expected extraction tool call was missing
    MissingToolCall {
        /// Expected extraction tool name
        tool_name: String,
        /// Provider name that produced the response
        provider: String,
    },
    /// Multiple extraction tool calls were returned when exactly one was expected
    UnexpectedToolCallCount {
        /// Expected extraction tool name
        tool_name: String,
        /// Provider name that produced the response
        provider: String,
        /// Number of matching tool calls that were returned
        count: usize,
    },
    /// Requested extraction mode is unsupported
    Unsupported {
        /// Extraction mode that was requested
        mode: ExtractionMode,
        /// Provider name that rejected the request
        provider: String,
    },
    /// Extraction mode conflicted with the request shape
    ToolConflict {
        /// Extraction mode that was requested
        mode: ExtractionMode,
        /// Provider name associated with the request
        provider: String,
    },
    /// Structured output could not be parsed
    Parse {
        /// Extraction mode that was attempted
        mode: ExtractionMode,
        /// Provider name that produced the response
        provider: String,
        /// Raw output that failed to parse
        raw: String,
        /// Parse failure description
        message: String,
    },
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::MissingStructuredText { mode, provider } => write!(
                f,
                "provider '{provider}' did not return structured text for extraction mode {mode:?}"
            ),
            ExtractError::MissingToolCall {
                tool_name,
                provider,
            } => write!(
                f,
                "provider '{provider}' did not return the extraction tool call '{tool_name}'"
            ),
            ExtractError::UnexpectedToolCallCount {
                tool_name,
                provider,
                count,
            } => write!(
                f,
                "provider '{provider}' returned {count} extraction tool calls named '{tool_name}', expected exactly one"
            ),
            ExtractError::Unsupported { mode, provider } => write!(
                f,
                "provider '{provider}' does not support built-in extraction mode {mode:?}"
            ),
            ExtractError::ToolConflict { mode, provider } => write!(
                f,
                "provider '{provider}' cannot use built-in extraction mode {mode:?} with existing request tools because this mode is exclusive"
            ),
            ExtractError::Parse {
                mode,
                provider,
                message,
                ..
            } => write!(
                f,
                "failed to parse extracted output from provider '{provider}' using mode {mode:?}: {message}"
            ),
        }
    }
}

impl std::error::Error for ExtractError {}

impl ExtractError {
    /// Returns the raw unparsed output when available
    #[must_use]
    pub fn raw_output(&self) -> Option<&str> {
        match self {
            ExtractError::Parse { raw, .. } => Some(raw),
            _ => None,
        }
    }
}

async fn default_extract_response<P, T>(
    provider: &P,
    request: &ChatRequest,
) -> crate::Result<Extracted<T>>
where
    P: ChatProvider + ?Sized,
    T: JsonSchema + DeserializeOwned + Send,
{
    let mode = effective_mode(
        provider,
        request,
        request.option::<ExtractionMode>().copied(),
    );
    validate_mode_request(request, mode, provider.provider_name())
        .map_err(|err| crate::Error::Extract(Box::new(err)))?;
    let derived_request = build_mode_request::<T>(request, mode, provider.provider_name())
        .map_err(|err| crate::Error::Extract(Box::new(err)))?;
    let response = provider.chat(&derived_request).await?;
    let value =
        parse_extracted_response::<T>(&response, mode, &derived_request, provider.provider_name())
            .map_err(|err| crate::Error::Extract(Box::new(err)))?;

    Ok(Extracted {
        value,
        response,
        metadata: ExtractionMetadata {
            passes: 1,
            repaired: false,
        },
    })
}

fn effective_mode(
    provider: &(impl ChatProvider + ?Sized),
    request: &ChatRequest,
    requested: Option<ExtractionMode>,
) -> ExtractionMode {
    match requested.unwrap_or_default() {
        ExtractionMode::Auto => {
            let structured_output =
                provider.chat_capability(&request.model, ChatCapability::StructuredOutput);
            let tool_calls = provider.chat_capability(&request.model, ChatCapability::ToolCalls);

            if structured_output == CapabilitySupport::Supported {
                ExtractionMode::Native
            } else if tool_calls == CapabilitySupport::Supported {
                ExtractionMode::ForcedTool
            } else {
                ExtractionMode::Auto
            }
        }
        other => other,
    }
}

fn build_mode_request<T>(
    request: &ChatRequest,
    mode: ExtractionMode,
    provider_name: &'static str,
) -> Result<ChatRequest, ExtractError>
where
    T: JsonSchema,
{
    match mode {
        ExtractionMode::Auto => Err(ExtractError::Unsupported {
            mode,
            provider: provider_name.to_owned(),
        }),
        ExtractionMode::Native => build_json_schema_request::<T>(request),
        ExtractionMode::ForcedTool => build_forced_tool_request::<T>(request),
    }
}

fn request_has_tools(request: &ChatRequest) -> bool {
    request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
}

fn validate_mode_request(
    request: &ChatRequest,
    mode: ExtractionMode,
    provider_name: &'static str,
) -> Result<(), ExtractError> {
    if mode == ExtractionMode::ForcedTool && request_has_tools(request) {
        return Err(ExtractError::ToolConflict {
            mode,
            provider: provider_name.to_owned(),
        });
    }

    Ok(())
}

fn build_json_schema_request<T>(request: &ChatRequest) -> Result<ChatRequest, ExtractError>
where
    T: JsonSchema,
{
    let schema =
        serde_json::to_value(schemars::schema_for!(T)).map_err(|e| ExtractError::Parse {
            mode: ExtractionMode::Native,
            provider: "schema".to_owned(),
            raw: String::new(),
            message: format!("failed to serialize extraction schema: {e}"),
        })?;

    Ok(ChatRequest {
        model: request.model.clone(),
        system: request.system.clone(),
        messages: request.messages.clone(),
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        stop: request.stop.clone(),
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        tools: request.tools.clone(),
        tool_choice: request.tool_choice.clone(),
        response_format: Some(ResponseFormat::JsonSchema {
            name: Some(extraction_schema_name::<T>(&schema)),
            schema,
            strict: Some(true),
        }),
        seed: request.seed,
        reasoning: request.reasoning.clone(),
        parallel_tool_calls: request.parallel_tool_calls,
        options: request.options.clone(),
    })
}

fn extraction_schema_name<T>(schema: &serde_json::Value) -> String {
    if let Some(title) = schema.get("title").and_then(serde_json::Value::as_str) {
        let sanitized = sanitize_schema_name(title);
        if sanitized != "structured_output" {
            return sanitized;
        }
    }

    let raw = std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap_or("structured_output");
    sanitize_schema_name(raw)
}

fn sanitize_schema_name(raw: &str) -> String {
    let mut name = String::with_capacity(raw.len());
    let mut last_was_underscore = false;

    for ch in raw.chars() {
        let normalized = if ch.is_ascii_alphanumeric() { ch } else { '_' };
        if normalized == '_' {
            if !last_was_underscore && !name.is_empty() {
                name.push('_');
            }
            last_was_underscore = true;
        } else {
            name.push(normalized.to_ascii_lowercase());
            last_was_underscore = false;
        }
    }

    let trimmed = name.trim_matches('_');
    if trimmed.is_empty() {
        "structured_output".into()
    } else {
        trimmed.into()
    }
}

fn build_forced_tool_request<T>(request: &ChatRequest) -> Result<ChatRequest, ExtractError>
where
    T: JsonSchema,
{
    let schema =
        serde_json::to_value(schemars::schema_for!(T)).map_err(|e| ExtractError::Parse {
            mode: ExtractionMode::ForcedTool,
            provider: "schema".to_owned(),
            raw: String::new(),
            message: format!("failed to serialize extraction schema: {e}"),
        })?;

    let tool_name = extraction_tool_name();
    Ok(ChatRequest {
        model: request.model.clone(),
        system: request.system.clone(),
        messages: request.messages.clone(),
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        stop: request.stop.clone(),
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        tools: Some(vec![
            Tool::new(tool_name, schema).description("Submit extracted structured data"),
        ]),
        tool_choice: Some(ToolChoice::Specific {
            name: tool_name.to_owned(),
        }),
        response_format: None,
        seed: request.seed,
        reasoning: request.reasoning.clone(),
        parallel_tool_calls: None,
        options: request.options.clone(),
    })
}

fn parse_extracted_response<T>(
    response: &ChatResponse,
    mode: ExtractionMode,
    request: &ChatRequest,
    provider_name: &'static str,
) -> Result<T, ExtractError>
where
    T: DeserializeOwned,
{
    match mode {
        ExtractionMode::Native => {
            let text = response.text().ok_or(ExtractError::MissingStructuredText {
                mode,
                provider: provider_name.to_owned(),
            })?;

            serde_json::from_str(&text).map_err(|e| ExtractError::Parse {
                mode,
                provider: provider_name.to_owned(),
                raw: text,
                message: e.to_string(),
            })
        }
        ExtractionMode::ForcedTool => {
            let tool_name = extraction_tool_name_from_request(request);
            let matching_tool_calls: Vec<_> = response
                .tool_calls()
                .filter(|tool_call| tool_call.name == tool_name)
                .collect();

            let tool_call = match matching_tool_calls.as_slice() {
                [] => {
                    return Err(ExtractError::MissingToolCall {
                        tool_name: tool_name.to_owned(),
                        provider: provider_name.to_owned(),
                    });
                }
                [tool_call] => tool_call,
                tool_calls => {
                    return Err(ExtractError::UnexpectedToolCallCount {
                        tool_name: tool_name.to_owned(),
                        provider: provider_name.to_owned(),
                        count: tool_calls.len(),
                    });
                }
            };

            let raw = tool_call.arguments.to_string();
            serde_json::from_str(tool_call.arguments).map_err(|e| ExtractError::Parse {
                mode,
                provider: provider_name.to_owned(),
                raw,
                message: e.to_string(),
            })
        }
        // INVARIANT: `build_mode_request` rejects `Auto` with `ExtractError::Unsupported`
        // before this function is called. Defensive error return instead of panic.
        ExtractionMode::Auto => {
            debug_assert!(false, "mode should resolve before parsing");
            Err(ExtractError::Unsupported {
                mode: ExtractionMode::Auto,
                provider: provider_name.to_owned(),
            })
        }
    }
}

fn extraction_tool_name() -> &'static str {
    "submit_structured_output"
}

fn extraction_tool_name_from_request(request: &ChatRequest) -> &str {
    match &request.tool_choice {
        Some(ToolChoice::Specific { name }) => name,
        _ => extraction_tool_name(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapabilitySupport, ChatCapability, ChatResponseBuilder, ContentBlock, FinishReason,
        Message, MockProvider, MockResponse, ResponseMetadata,
    };
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
    struct Review {
        title: String,
        rating: u8,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
    #[serde(tag = "kind")]
    enum ClassifiedResult {
        Answer { summary: String },
        Escalation { reason: String },
        Refusal { message: String },
    }

    #[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
    #[serde(tag = "kind", content = "data")]
    enum AdjacentResult {
        Answer { summary: String },
        Escalation { reason: String },
    }

    #[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
    #[serde(untagged)]
    enum UntaggedResult {
        Answer { summary: String },
        Escalation { reason: String },
    }

    fn json_text_response(text: serde_json::Value) -> ChatResponse {
        ChatResponseBuilder::new()
            .text(text.to_string())
            .finish_reason(FinishReason::Stop)
            .usage(10, 5)
            .model("mock-model")
            .id("resp_1")
            .build()
    }

    fn tool_call_response(arguments: serde_json::Value) -> ChatResponse {
        ChatResponse {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "submit_structured_output".into(),
                arguments: arguments.to_string(),
            }],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
            model: Some("mock-model".into()),
            id: Some("resp_2".into()),
            metadata: ResponseMetadata::new(),
        }
    }

    #[tokio::test]
    async fn extract_uses_native_strategy_when_supported() {
        let provider = MockProvider::new([json_text_response(json!({
            "title": "Dune",
            "rating": 9
        }))])
        .with_supported_chat_capabilities([
            ChatCapability::StructuredOutput,
            ChatCapability::ToolCalls,
        ]);

        let request = ChatRequest::new("gpt-4o").message(Message::user("Review Dune"));
        let review = provider.extract::<Review>(&request).await.unwrap().value;

        assert_eq!(review.title, "Dune");
        let recorded = provider.requests();
        assert!(matches!(
            recorded[0].response_format,
            Some(ResponseFormat::JsonSchema {
                name: Some(ref name),
                strict: Some(true),
                ..
            }) if name == "review"
        ));
    }

    #[tokio::test]
    async fn extract_auto_falls_back_to_forced_tool_strategy() {
        let provider = MockProvider::new([tool_call_response(json!({
            "title": "Heat",
            "rating": 10
        }))])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let request = ChatRequest::new("claude").message(Message::user("Review Heat"));
        let review = provider.extract::<Review>(&request).await.unwrap().value;
        assert_eq!(review.title, "Heat");

        let recorded = provider.requests();
        assert!(matches!(
            recorded[0].tool_choice,
            Some(ToolChoice::Specific { ref name }) if name == "submit_structured_output"
        ));
    }

    #[tokio::test]
    async fn extraction_mode_request_option_overrides_auto() {
        let provider = MockProvider::new([tool_call_response(json!({
            "title": "Primer",
            "rating": 8
        }))])
        .with_supported_chat_capabilities([
            ChatCapability::StructuredOutput,
            ChatCapability::ToolCalls,
        ]);

        let request = ChatRequest::new("hybrid")
            .message(Message::user("Review Primer"))
            .with_option(ExtractionMode::ForcedTool);

        let review = provider.extract::<Review>(&request).await.unwrap().value;
        assert_eq!(review.title, "Primer");

        let recorded = provider.requests();
        assert!(matches!(
            recorded[0].tool_choice,
            Some(ToolChoice::Specific { ref name }) if name == "submit_structured_output"
        ));
        assert_eq!(recorded[0].parallel_tool_calls, None);
    }

    #[tokio::test]
    async fn forced_tool_extraction_ignores_non_extraction_tool_calls() {
        let provider = MockProvider::new([ChatResponse {
            content: vec![
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    arguments: json!({"q": "rust"}).to_string(),
                },
                ContentBlock::ToolCall {
                    id: "call_2".into(),
                    name: "submit_structured_output".into(),
                    arguments: json!({
                        "title": "Primer",
                        "rating": 8
                    })
                    .to_string(),
                },
            ],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
            model: Some("mock-model".into()),
            id: Some("resp_2".into()),
            metadata: ResponseMetadata::new(),
        }])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let request = ChatRequest::new("claude")
            .message(Message::user("Review Primer"))
            .with_option(ExtractionMode::ForcedTool)
            .parallel_tool_calls(true);

        let review = provider.extract::<Review>(&request).await.unwrap().value;
        assert_eq!(review.title, "Primer");
    }

    #[tokio::test]
    async fn forced_tool_extraction_errors_when_expected_tool_call_missing() {
        let provider = MockProvider::new([ChatResponse {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: json!({"q": "rust"}).to_string(),
            }],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
            model: Some("mock-model".into()),
            id: Some("resp_missing".into()),
            metadata: ResponseMetadata::new(),
        }])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let request = ChatRequest::new("claude")
            .message(Message::user("Review Primer"))
            .with_option(ExtractionMode::ForcedTool);

        let err = provider.extract::<Review>(&request).await.unwrap_err();
        match err {
            crate::Error::Extract(inner) => match *inner {
                ExtractError::MissingToolCall { ref tool_name, .. } => {
                    assert_eq!(tool_name, "submit_structured_output");
                }
                other => panic!("expected missing tool call error, got {other:?}"),
            },
            other => panic!("expected extract error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn forced_tool_extraction_errors_when_multiple_expected_tool_calls_exist() {
        let provider = MockProvider::new([ChatResponse {
            content: vec![
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "submit_structured_output".into(),
                    arguments: json!({
                        "title": "Primer",
                        "rating": 8
                    })
                    .to_string(),
                },
                ContentBlock::ToolCall {
                    id: "call_2".into(),
                    name: "submit_structured_output".into(),
                    arguments: json!({
                        "title": "Primer",
                        "rating": 9
                    })
                    .to_string(),
                },
            ],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
            model: Some("mock-model".into()),
            id: Some("resp_multi".into()),
            metadata: ResponseMetadata::new(),
        }])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let request = ChatRequest::new("claude")
            .message(Message::user("Review Primer"))
            .with_option(ExtractionMode::ForcedTool);

        let err = provider.extract::<Review>(&request).await.unwrap_err();
        match err {
            crate::Error::Extract(inner) => match *inner {
                ExtractError::UnexpectedToolCallCount { count, .. } => {
                    assert_eq!(count, 2);
                }
                other => panic!("expected unexpected tool count error, got {other:?}"),
            },
            other => panic!("expected extract error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_preserves_metadata_and_usage() {
        let provider = MockProvider::new([json_text_response(json!({
            "title": "Alien",
            "rating": 8
        }))])
        .with_supported_chat_capabilities([ChatCapability::StructuredOutput]);

        let request = ChatRequest::new("gpt-4o").message(Message::user("Review Alien"));
        let extracted: Extracted<Review> = provider.extract(&request).await.unwrap();
        assert_eq!(extracted.value.title, "Alien");
        assert_eq!(extracted.response.id.as_deref(), Some("resp_1"));
    }

    #[tokio::test]
    async fn multishape_extract_supports_internally_tagged_enum_in_native_mode() {
        let provider = MockProvider::new([json_text_response(json!({
            "kind": "Answer",
            "summary": "The user is satisfied."
        }))])
        .with_supported_chat_capabilities([
            ChatCapability::StructuredOutput,
            ChatCapability::ToolCalls,
        ]);

        let request = ChatRequest::new("gpt-4o").message(Message::user(
            "Classify the outcome and respond as structured data",
        ));
        let result = provider
            .extract::<ClassifiedResult>(&request)
            .await
            .unwrap()
            .value;

        assert_eq!(
            result,
            ClassifiedResult::Answer {
                summary: "The user is satisfied.".into(),
            }
        );

        let recorded = provider.requests();
        assert!(matches!(
            recorded[0].response_format,
            Some(ResponseFormat::JsonSchema {
                name: Some(ref name),
                strict: Some(true),
                ..
            }) if name == "classifiedresult"
        ));
    }

    #[test]
    fn extraction_schema_name_sanitizes_type_names() {
        assert_eq!(sanitize_schema_name("Review"), "review");
        assert_eq!(sanitize_schema_name("ClassifiedResult"), "classifiedresult");
        assert_eq!(sanitize_schema_name("Vec<Review>"), "vec_review");
    }

    #[test]
    fn extraction_schema_name_prefers_schema_title_then_falls_back() {
        assert_eq!(
            extraction_schema_name::<Review>(&json!({"title": "Review"})),
            "review"
        );
        assert_eq!(
            extraction_schema_name::<Vec<Review>>(&json!({"title": "Array_of_Review"})),
            "array_of_review"
        );
        assert_eq!(extraction_schema_name::<Review>(&json!({})), "review");
    }

    #[tokio::test]
    async fn multishape_extract_supports_internally_tagged_enum_in_forced_tool_mode() {
        let provider = MockProvider::new([tool_call_response(json!({
            "kind": "Escalation",
            "reason": "Account access requires manual review."
        }))])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let request = ChatRequest::new("claude").message(Message::user(
            "Classify the outcome and respond as structured data",
        ));
        let result = provider
            .extract::<ClassifiedResult>(&request)
            .await
            .unwrap()
            .value;

        assert_eq!(
            result,
            ClassifiedResult::Escalation {
                reason: "Account access requires manual review.".into(),
            }
        );

        let recorded = provider.requests();
        assert!(matches!(
            recorded[0].tool_choice,
            Some(ToolChoice::Specific { ref name }) if name == "submit_structured_output"
        ));
    }

    #[tokio::test]
    async fn multishape_extract_supports_adjacently_tagged_enum() {
        let provider = MockProvider::new([json_text_response(json!({
            "kind": "Answer",
            "data": { "summary": "Everything looks good." }
        }))])
        .with_supported_chat_capabilities([ChatCapability::StructuredOutput]);

        let request = ChatRequest::new("gpt-4o").message(Message::user(
            "Classify the outcome and respond as structured data",
        ));
        let result = provider
            .extract::<AdjacentResult>(&request)
            .await
            .unwrap()
            .value;

        assert_eq!(
            result,
            AdjacentResult::Answer {
                summary: "Everything looks good.".into(),
            }
        );
    }

    #[tokio::test]
    async fn multishape_extract_can_parse_untagged_enum_but_we_do_not_prefer_it() {
        let provider = MockProvider::new([json_text_response(json!({
            "reason": "Manual review required."
        }))])
        .with_supported_chat_capabilities([ChatCapability::StructuredOutput]);

        let request = ChatRequest::new("gpt-4o").message(Message::user(
            "Classify the outcome and respond as structured data",
        ));
        let result = provider
            .extract::<UntaggedResult>(&request)
            .await
            .unwrap()
            .value;

        assert_eq!(
            result,
            UntaggedResult::Escalation {
                reason: "Manual review required.".into(),
            }
        );
    }

    #[tokio::test]
    async fn extract_returns_error_extract_for_parse_failure() {
        let provider = MockProvider::new([json_text_response(json!("not-json"))])
            .with_supported_chat_capabilities([ChatCapability::StructuredOutput]);

        let request = ChatRequest::new("gpt-4o").message(Message::user("Review"));
        let err = provider.extract::<Review>(&request).await.unwrap_err();

        match err {
            crate::Error::Extract(inner) => match *inner {
                ExtractError::Parse { mode, .. } => {
                    assert_eq!(mode, ExtractionMode::Native)
                }
                other => panic!("expected parse error, got {other:?}"),
            },
            other => panic!("expected extract error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_auto_returns_unsupported_when_provider_has_no_extraction_capability() {
        let provider = MockProvider::new(Vec::<crate::Result<ChatResponse>>::new())
            .with_provider_name("plain_mock");

        let request = ChatRequest::new("basic-model").message(Message::user("Review"));
        let err = provider.extract::<Review>(&request).await.unwrap_err();

        match err {
            crate::Error::Extract(inner) => match *inner {
                ExtractError::Unsupported { mode, provider } => {
                    assert_eq!(mode, ExtractionMode::Auto);
                    assert_eq!(provider, "plain_mock");
                }
                other => panic!("expected unsupported error, got {other:?}"),
            },
            other => panic!("expected extract error, got {other:?}"),
        }

        assert_eq!(provider.call_count(), 0);
    }

    #[tokio::test]
    async fn extract_auto_uses_forced_tool_when_structured_output_is_unknown() {
        let provider = MockProvider::new([MockResponse::tool_call(
            "call_extract_1",
            "submit_structured_output",
            json!({ "title": "Heat", "rating": 10 }),
        )])
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
        .with_provider_name("tool_only_mock");

        let request = ChatRequest::new("basic-model").message(Message::user("Review"));
        let extracted = provider.extract::<Review>(&request).await.unwrap();

        assert_eq!(extracted.value.title, "Heat");
        assert_eq!(extracted.value.rating, 10);
        assert_eq!(provider.call_count(), 1);

        let recorded = provider.last_request().unwrap();
        assert_eq!(recorded.response_format, None);
        assert!(
            matches!(recorded.tool_choice, Some(ToolChoice::Specific { ref name }) if name == "submit_structured_output")
        );
        assert_eq!(recorded.tools.as_ref().map(Vec::len), Some(1));
    }

    #[tokio::test]
    async fn parse_errors_expose_raw_output_and_display_message() {
        let provider = MockProvider::new([json_text_response(json!("not-json"))])
            .with_supported_chat_capabilities([ChatCapability::StructuredOutput])
            .with_provider_name("native_mock");

        let request = ChatRequest::new("gpt-4o").message(Message::user("Review"));
        let err = provider.extract::<Review>(&request).await.unwrap_err();

        match err {
            crate::Error::Extract(inner) => {
                let message = inner.to_string();
                assert!(message.starts_with(
                    "failed to parse extracted output from provider 'native_mock' using mode Native: invalid type: string \"not-json\", expected struct Review"
                ));
                assert_eq!(inner.raw_output(), Some("\"not-json\""));
            }
            other => panic!("expected extract error, got {other:?}"),
        }
    }

    #[test]
    fn non_parse_errors_do_not_expose_raw_output() {
        let unsupported = ExtractError::Unsupported {
            mode: ExtractionMode::Auto,
            provider: "mock".into(),
        };
        let missing_tool = ExtractError::MissingToolCall {
            tool_name: "submit_structured_output".into(),
            provider: "mock".into(),
        };

        assert_eq!(unsupported.raw_output(), None);
        assert_eq!(missing_tool.raw_output(), None);
        assert_eq!(
            unsupported.to_string(),
            "provider 'mock' does not support built-in extraction mode Auto"
        );
        assert_eq!(
            missing_tool.to_string(),
            "provider 'mock' did not return the extraction tool call 'submit_structured_output'"
        );
    }

    #[tokio::test]
    async fn extract_auto_needs_orchestration_when_request_already_has_tools() {
        let provider = MockProvider::new(Vec::<crate::Result<ChatResponse>>::new())
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            )
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let request = ChatRequest::new("claude")
            .message(Message::user("Review with supporting tool usage"))
            .tools(vec![
                Tool::new(
                    "lookup_customer",
                    json!({
                        "type": "object",
                        "properties": { "id": { "type": "string" } },
                        "required": ["id"],
                        "additionalProperties": false,
                    }),
                )
                .description("Look up customer details"),
            ]);

        let err = provider.extract::<Review>(&request).await.unwrap_err();

        match err {
            crate::Error::Extract(inner) => match *inner {
                ExtractError::ToolConflict { mode, provider } => {
                    assert_eq!(mode, ExtractionMode::ForcedTool);
                    assert_eq!(provider, "mock");
                }
                other => panic!("expected tool-conflict error, got {other:?}"),
            },
            other => panic!("expected extract error, got {other:?}"),
        }
    }

    #[test]
    fn extraction_request_from_chat_request_preserves_model_and_messages() {
        let request = ChatRequest::new("claude-sonnet")
            .system("You are a reviewer")
            .message(Message::user("Review this interaction"));

        let extraction_request: ExtractionRequest = (&request).into();
        assert_eq!(extraction_request.model, "claude-sonnet");
        assert_eq!(extraction_request.messages, request.messages);
    }

    #[test]
    fn extraction_request_into_chat_request_reuses_only_model_and_messages() {
        let original = ChatRequest::new("claude-sonnet")
            .system("You are a reviewer")
            .message(Message::user("Review this interaction"))
            .temperature(0.8)
            .max_tokens(512)
            .tools(vec![
                Tool::new(
                    "lookup_customer",
                    json!({
                        "type": "object",
                        "properties": { "id": { "type": "string" } },
                        "required": ["id"],
                        "additionalProperties": false,
                    }),
                )
                .description("Look up customer details"),
            ])
            .tool_choice(ToolChoice::Required)
            .response_format(ResponseFormat::Json)
            .with_option(ExtractionMode::ForcedTool);

        let extraction_request: ExtractionRequest = (&original).into();
        let extraction_req: ChatRequest = extraction_request.into();

        assert_eq!(extraction_req.model, original.model);
        assert_eq!(extraction_req.messages, original.messages);
        assert_eq!(extraction_req.tools, None);
        assert_eq!(extraction_req.tool_choice, None);
        assert_eq!(extraction_req.response_format, None);
        assert_eq!(extraction_req.temperature, None);
        assert_eq!(extraction_req.max_tokens, None);
        assert!(extraction_req.option::<ExtractionMode>().is_none());
    }

    #[test]
    fn extraction_request_builder_helpers_match_chat_request_message_shape() {
        let request = ExtractionRequest::new("gpt-4o")
            .system("You are a reviewer")
            .user("Review Arrival")
            .assistant("Sure.")
            .message(Message::tool_result("call_1", "lookup_review", "ok"));

        assert_eq!(request.model, "gpt-4o");
        assert_eq!(request.messages.len(), 4);
        assert!(matches!(request.messages[0], Message::System { .. }));
        assert!(matches!(request.messages[1], Message::User { .. }));
        assert!(matches!(request.messages[2], Message::Assistant { .. }));
        assert!(matches!(request.messages[3], Message::Tool { .. }));
    }

    #[tokio::test]
    async fn extracting_provider_strips_tools_and_options_from_request() {
        let provider = MockProvider::new([tool_call_response(json!({
            "title": "Heat",
            "rating": 10
        }))])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let wrapped = ExtractingProvider::new(provider);
        let request = ChatRequest::new("claude")
            .message(Message::user("Review Heat"))
            .tools(vec![
                Tool::new(
                    "lookup_customer",
                    json!({
                        "type": "object",
                        "properties": { "id": { "type": "string" } },
                        "required": ["id"],
                        "additionalProperties": false,
                    }),
                )
                .description("Look up customer details"),
            ])
            .tool_choice(ToolChoice::Required)
            .response_format(ResponseFormat::Json);

        let review = wrapped.extract::<Review>(&request).await.unwrap().value;
        assert_eq!(review.title, "Heat");

        let recorded = wrapped.into_inner().requests();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].messages, request.messages);
        assert!(matches!(
            recorded[0].tools,
            Some(ref tools) if tools.len() == 1 && tools[0].name == "submit_structured_output"
        ));
        assert_eq!(
            recorded[0].tool_choice,
            Some(ToolChoice::Specific {
                name: "submit_structured_output".into(),
            })
        );
        assert_eq!(recorded[0].response_format, None);
    }

    #[tokio::test]
    async fn extractor_extracts_from_extraction_request_using_native_strategy() {
        let provider = MockProvider::new([json_text_response(json!({
            "title": "Dune",
            "rating": 9
        }))])
        .with_supported_chat_capabilities([
            ChatCapability::StructuredOutput,
            ChatCapability::ToolCalls,
        ]);

        let extracted: Extracted<Review> = Extractor::new()
            .extract(
                &provider,
                ExtractionRequest {
                    model: "gpt-4o".into(),
                    messages: vec![Message::user("Review Dune")],
                },
            )
            .await
            .unwrap();

        assert_eq!(extracted.value.title, "Dune");
        assert_eq!(
            extracted.metadata,
            ExtractionMetadata {
                passes: 1,
                repaired: false,
            }
        );

        let recorded = provider.requests();
        assert!(matches!(
            recorded[0].response_format,
            Some(ResponseFormat::JsonSchema {
                strict: Some(true),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn extractor_extracts_from_extraction_request_using_forced_tool() {
        let provider = MockProvider::new([tool_call_response(json!({
            "title": "Heat",
            "rating": 10
        }))])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let extracted: Extracted<Review> = Extractor::new()
            .extract(
                &provider,
                ExtractionRequest {
                    model: "claude".into(),
                    messages: vec![Message::user("Review Heat")],
                },
            )
            .await
            .unwrap();

        assert_eq!(extracted.value.title, "Heat");
        assert_eq!(
            extracted.metadata,
            ExtractionMetadata {
                passes: 1,
                repaired: false,
            }
        );

        let recorded = provider.requests();
        assert!(matches!(
            recorded[0].tool_choice,
            Some(ToolChoice::Specific { ref name }) if name == "submit_structured_output"
        ));
    }

    #[tokio::test]
    async fn extractor_accepts_borrowed_chat_request_directly() {
        let provider = MockProvider::new([json_text_response(json!({
            "title": "Arrival",
            "rating": 8
        }))])
        .with_supported_chat_capabilities([
            ChatCapability::StructuredOutput,
            ChatCapability::ToolCalls,
        ]);

        let request = ChatRequest::new("gpt-4o")
            .message(Message::user("Review Arrival"))
            .tools(vec![
                Tool::new(
                    "lookup_customer",
                    json!({
                        "type": "object",
                        "properties": { "id": { "type": "string" } },
                        "required": ["id"],
                        "additionalProperties": false,
                    }),
                )
                .description("Look up customer details"),
            ])
            .tool_choice(ToolChoice::Required)
            .response_format(ResponseFormat::Json)
            .with_option(ExtractionMode::ForcedTool);

        let extracted: Extracted<Review> =
            Extractor::new().extract(&provider, &request).await.unwrap();

        assert_eq!(extracted.value.title, "Arrival");

        let recorded = provider.requests();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].model, request.model);
        assert_eq!(recorded[0].messages, request.messages);
        assert!(matches!(
            recorded[0].response_format,
            Some(ResponseFormat::JsonSchema {
                strict: Some(true),
                ..
            })
        ));
        assert_eq!(recorded[0].tools, None);
        assert_eq!(recorded[0].tool_choice, None);
        assert!(recorded[0].option::<ExtractionMode>().is_none());
    }
}
