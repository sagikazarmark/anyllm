use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use futures_core::Stream;
use serde::Serialize;
use tracing::{Instrument, Span};

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream, Error,
    FinishReason, ResponseFormat, Result, StreamCollector, StreamEvent, Usage, UsageMetadataMode,
};

/// A wrapper around any [`ChatProvider`] that emits `tracing` spans with
/// [OTel Gen AI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
/// field names for each `chat()` and `chat_stream()` call.
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
///             text: format!("traced response for {}", request.model),
///         }])
///         .finish_reason(FinishReason::Stop)
///         .model(request.model.clone()))
///     }
///
///     async fn chat_stream(&self, request: &ChatRequest) -> anyllm::Result<Self::Stream> {
///         Ok(self.chat(request).await?.into_stream())
///     }
///
///     fn provider_name(&self) -> &'static str {
///         "demo"
///     }
/// }
///
/// # async fn example() -> anyllm::Result<()> {
/// let provider = TracingChatProvider::new(StaticProvider).with_content_capture(
///     TracingContentConfig {
///         capture_input_messages: true,
///         capture_output_messages: true,
///         max_payload_chars: 256,
///     },
/// );
///
/// let request = ChatRequest::new("demo-model").user("Say hello");
/// let response = provider.chat(&request).await?;
/// assert_eq!(response.text().as_deref(), Some("traced response for demo-model"));
/// # Ok(())
/// # }
/// ```
pub struct TracingChatProvider<T> {
    inner: T,
    content_config: TracingContentConfig,
}

impl<T> fmt::Debug for TracingChatProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracingChatProvider")
            .field("content_config", &self.content_config)
            .finish_non_exhaustive()
    }
}

impl<T> TracingChatProvider<T> {
    /// Wrap a provider with tracing instrumentation and default content-capture settings.
    #[must_use]
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            content_config: TracingContentConfig::default(),
        }
    }

    /// Configure opt-in input/output message capture with truncation and redaction.
    #[must_use]
    pub fn with_content_capture(mut self, config: TracingContentConfig) -> Self {
        self.content_config = config;
        self
    }

    /// Borrow the current content-capture configuration.
    #[must_use]
    pub fn content_config(&self) -> &TracingContentConfig {
        &self.content_config
    }

    /// Consume the wrapper and return the wrapped provider.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Consume the wrapper and return the provider plus tracing configuration.
    #[must_use]
    pub fn into_parts(self) -> (T, TracingContentConfig) {
        (self.inner, self.content_config)
    }
}

/// Maps internal provider names onto OTEL GenAI provider semantic-convention
/// values where the spec defines a standard name.
#[must_use]
pub fn otel_genai_provider_name(provider_name: &str) -> &str {
    match provider_name {
        "gemini" => "gcp.gemini",
        "alt_openai_compat" => "openai",
        "cloudflare_worker" => "cloudflare",
        other => other,
    }
}

impl<T> ChatProvider for TracingChatProvider<T>
where
    T: ChatProvider,
    T::Stream: 'static,
{
    type Stream = ChatStream;

    fn chat(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send {
        let span = make_chat_span(self.inner.provider_name(), request, &self.content_config);

        async move {
            let result = self.inner.chat(request).instrument(span.clone()).await;

            match &result {
                Ok(response) => record_response(&span, response, &self.content_config),
                Err(err) => record_error(&span, err),
            }

            result
        }
    }

    fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = Result<Self::Stream>> + Send {
        let span = make_chat_span(self.inner.provider_name(), request, &self.content_config);

        async move {
            let inner_stream = match self
                .inner
                .chat_stream(request)
                .instrument(span.clone())
                .await
            {
                Ok(stream) => stream,
                Err(err) => {
                    record_error(&span, &err);
                    return Err(err);
                }
            };

            let tracing_stream = TracingStream {
                inner: Box::pin(inner_stream),
                span,
                content_config: self.content_config,
                started_at: Instant::now(),
                accumulated_usage: None,
                finish_reason: None,
                response_id: None,
                response_model: None,
                output_collector: self
                    .content_config
                    .capture_output_messages
                    .then(StreamCollector::new),
                ttft_recorded: false,
                finalized: false,
            };

            Ok(Box::pin(tracing_stream) as ChatStream)
        }
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner.chat_capability(model, capability)
    }

    fn provider_name(&self) -> &'static str {
        self.inner.provider_name()
    }
}

#[cfg(feature = "extract")]
impl<T> crate::ExtractExt for TracingChatProvider<T>
where
    T: ChatProvider + Sync,
    T::Stream: 'static,
{
}

/// Opt-in capture settings for tracing GenAI message content.
#[derive(Debug, Clone, Copy)]
pub struct TracingContentConfig {
    /// Whether input messages should be recorded on the span.
    pub capture_input_messages: bool,
    /// Whether output messages should be recorded on the span.
    pub capture_output_messages: bool,
    /// Maximum number of characters captured per payload after truncation.
    pub max_payload_chars: usize,
}

impl Default for TracingContentConfig {
    fn default() -> Self {
        Self {
            capture_input_messages: false,
            capture_output_messages: false,
            max_payload_chars: 4096,
        }
    }
}

/// A stream wrapper that accumulates response metadata and records it on the
/// tracing span when the stream completes or is dropped early.
struct TracingStream<S> {
    inner: Pin<Box<S>>,
    span: Span,
    content_config: TracingContentConfig,
    started_at: Instant,
    accumulated_usage: Option<Usage>,
    finish_reason: Option<FinishReason>,
    response_id: Option<String>,
    response_model: Option<String>,
    output_collector: Option<StreamCollector>,
    ttft_recorded: bool,
    finalized: bool,
}

impl<S> TracingStream<S> {
    fn record_ttft_if_needed(&mut self, event: &StreamEvent) {
        if self.ttft_recorded {
            return;
        }

        if matches!(
            event,
            StreamEvent::TextDelta { .. }
                | StreamEvent::ReasoningDelta { .. }
                | StreamEvent::ToolCallDelta { .. }
        ) {
            self.ttft_recorded = true;
            self.span.record(
                "anyllm.response.ttft_ms",
                self.started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    fn collect_output_event(&mut self, event: &StreamEvent) {
        if let Some(collector) = &mut self.output_collector
            && collector.push_ref(event).is_err()
        {
            self.output_collector = None;
        }
    }

    fn observe_event(&mut self, event: &StreamEvent) {
        self.record_ttft_if_needed(event);
        self.collect_output_event(event);

        match event {
            StreamEvent::ResponseStart { id, model } => {
                if let Some(id) = id {
                    self.response_id = Some(id.clone());
                }
                if let Some(model) = model {
                    self.response_model = Some(model.clone());
                }
            }
            StreamEvent::ResponseMetadata {
                usage,
                usage_mode,
                finish_reason,
                id,
                model,
                ..
            } => {
                if let Some(chunk_usage) = usage {
                    match *usage_mode {
                        UsageMetadataMode::Delta => match &mut self.accumulated_usage {
                            Some(existing) => *existing += chunk_usage,
                            None => self.accumulated_usage = Some(chunk_usage.clone()),
                        },
                        UsageMetadataMode::Snapshot => match &mut self.accumulated_usage {
                            Some(existing) => existing.clone_from(chunk_usage),
                            None => self.accumulated_usage = Some(chunk_usage.clone()),
                        },
                    }
                }

                if let Some(reason) = finish_reason {
                    match &mut self.finish_reason {
                        Some(existing) => existing.clone_from(reason),
                        None => self.finish_reason = Some(reason.clone()),
                    }
                }

                if let Some(id) = id {
                    self.response_id = Some(id.clone());
                }
                if let Some(model) = model {
                    self.response_model = Some(model.clone());
                }
            }
            _ => {}
        }
    }

    fn finalize_once(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;

        if let Some(ref usage) = self.accumulated_usage {
            record_usage(&self.span, usage);
        }

        if let Some(ref reason) = self.finish_reason {
            record_finish_reason(&self.span, reason);
        }

        if let Some(ref id) = self.response_id {
            self.span.record("gen_ai.response.id", id.as_str());
        }

        if let Some(ref model) = self.response_model {
            self.span.record("gen_ai.response.model", model.as_str());
        }

        if self.content_config.capture_output_messages
            && let Some(collector) = self.output_collector.take()
            && let Ok(collected) = collector.finish_partial()
        {
            record_output_messages(&self.span, &collected.response, &self.content_config);
        }
    }
}

impl<S> Drop for TracingStream<S> {
    fn drop(&mut self) {
        self.finalize_once();
    }
}

impl<S> Stream for TracingStream<S>
where
    S: Stream<Item = Result<StreamEvent>> + Send,
{
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY of get_mut(): TracingStream does not require pinning guarantees —
        // its inner field is Pin<Box<S>>, which is Unpin regardless of whether S is.
        let this = self.get_mut();
        let span = this.span.clone();
        let _guard = span.enter();

        let poll = this.inner.as_mut().poll_next(cx);

        match &poll {
            Poll::Ready(Some(Ok(event))) => this.observe_event(event),
            Poll::Ready(Some(Err(err))) => {
                record_error(&this.span, err);
            }
            Poll::Ready(None) => {
                this.finalize_once();
            }
            _ => {}
        }

        poll
    }
}

/// Creates a tracing span with OTel Gen AI semantic convention fields.
fn make_chat_span(provider_name: &str, request: &ChatRequest, cfg: &TracingContentConfig) -> Span {
    let provider_name = otel_genai_provider_name(provider_name);
    let span = tracing::info_span!(
        "chat",
        "gen_ai.operation.name" = "chat",
        "gen_ai.provider.name" = provider_name,
        "gen_ai.request.model" = %request.model,
        "gen_ai.request.max_tokens" = tracing::field::Empty,
        "gen_ai.request.temperature" = tracing::field::Empty,
        "gen_ai.request.top_p" = tracing::field::Empty,
        "gen_ai.request.frequency_penalty" = tracing::field::Empty,
        "gen_ai.request.presence_penalty" = tracing::field::Empty,
        "gen_ai.request.stop_sequences" = tracing::field::Empty,
        "gen_ai.request.seed" = tracing::field::Empty,
        "gen_ai.output.type" = tracing::field::Empty,
        "gen_ai.usage.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.output_tokens" = tracing::field::Empty,
        "gen_ai.usage.cache_read.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.cache_creation.input_tokens" = tracing::field::Empty,
        "gen_ai.response.finish_reasons" = tracing::field::Empty,
        "gen_ai.response.id" = tracing::field::Empty,
        "gen_ai.response.model" = tracing::field::Empty,
        "gen_ai.retry.max_attempts" = tracing::field::Empty,
        "gen_ai.retry.attempts" = tracing::field::Empty,
        "gen_ai.retry.used" = tracing::field::Empty,
        "gen_ai.retry.last_delay_ms" = tracing::field::Empty,
        "gen_ai.retry.last_error_type" = tracing::field::Empty,
        "gen_ai.fallback.used" = tracing::field::Empty,
        "gen_ai.fallback.provider" = tracing::field::Empty,
        "gen_ai.fallback.error_type" = tracing::field::Empty,
        // OTEL GenAI currently defines TTFT as a metric concept, not a span
        // attribute, so keep the additional timing signal in a crate-local
        // namespace instead of minting a fake gen_ai.* semantic-convention key.
        "anyllm.response.ttft_ms" = tracing::field::Empty,
        "gen_ai.input.messages" = tracing::field::Empty,
        "gen_ai.output.messages" = tracing::field::Empty,
        "error.type" = tracing::field::Empty,
    );

    if let Some(max_tokens) = request.max_tokens {
        span.record("gen_ai.request.max_tokens", max_tokens);
    }

    if let Some(temperature) = request.temperature {
        span.record("gen_ai.request.temperature", f64::from(temperature));
    }

    if let Some(top_p) = request.top_p {
        span.record("gen_ai.request.top_p", f64::from(top_p));
    }

    if let Some(freq_penalty) = request.frequency_penalty {
        span.record("gen_ai.request.frequency_penalty", f64::from(freq_penalty));
    }

    if let Some(presence_penalty) = request.presence_penalty {
        span.record(
            "gen_ai.request.presence_penalty",
            f64::from(presence_penalty),
        );
    }

    if let Some(stop_sequences) = &request.stop {
        span.record(
            "gen_ai.request.stop_sequences",
            serde_json::to_string(stop_sequences)
                .unwrap_or_default()
                .as_str(),
        );
    }

    if let Some(seed) = request.seed {
        span.record("gen_ai.request.seed", seed);
    }

    if let Some(format) = &request.response_format {
        let output_type = match format {
            ResponseFormat::Text => Some("text"),
            ResponseFormat::Json | ResponseFormat::JsonSchema { .. } => Some("json"),
        };
        if let Some(output_type) = output_type {
            span.record("gen_ai.output.type", output_type);
        }
    }

    if cfg.capture_input_messages
        && let Some(payload) = serialize_redacted(&request.messages, cfg.max_payload_chars)
    {
        span.record("gen_ai.input.messages", payload.as_str());
    }

    span
}

/// Records response-level attributes on the span.
fn record_response(span: &Span, response: &ChatResponse, cfg: &TracingContentConfig) {
    if let Some(ref usage) = response.usage {
        record_usage(span, usage);
    }

    if let Some(ref reason) = response.finish_reason {
        record_finish_reason(span, reason);
    }

    if let Some(ref id) = response.id {
        span.record("gen_ai.response.id", id.as_str());
    }

    if let Some(ref model) = response.model {
        span.record("gen_ai.response.model", model.as_str());
    }

    record_output_messages(span, response, cfg);
}

fn record_output_messages(span: &Span, response: &ChatResponse, cfg: &TracingContentConfig) {
    if !cfg.capture_output_messages {
        return;
    }

    let output = serde_json::json!([{
        "role": "assistant",
        "parts": &response.content,
        "finish_reason": response.finish_reason.as_ref().map(FinishReason::as_str),
    }]);
    if let Some(payload) = serialize_redacted(&output, cfg.max_payload_chars) {
        span.record("gen_ai.output.messages", payload.as_str());
    }
}

fn serialize_redacted<T: Serialize>(value: &T, max_chars: usize) -> Option<String> {
    let mut json = serde_json::to_value(value).ok()?;
    redact_sensitive_fields(&mut json);
    let mut text = serde_json::to_string(&json).ok()?;
    if text.chars().count() > max_chars {
        let truncated: String = text.chars().take(max_chars).collect();
        text = format!("{truncated}...[TRUNCATED]");
    }
    Some(text)
}

fn redact_sensitive_fields(value: &mut serde_json::Value) {
    const SENSITIVE_KEYS: &[&str] = &[
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
    ];

    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                if SENSITIVE_KEYS
                    .iter()
                    .any(|sensitive| key.eq_ignore_ascii_case(sensitive))
                {
                    *val = serde_json::Value::String("[REDACTED]".to_string());
                } else {
                    redact_sensitive_fields(val);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                redact_sensitive_fields(item);
            }
        }
        _ => {}
    }
}

fn record_error(span: &Span, err: &Error) {
    span.record("error.type", err.telemetry_type());
}

fn record_usage(span: &Span, usage: &Usage) {
    if let Some(input) = usage
        .input_tokens
        .or_else(|| {
            usage
                .cached_input_tokens
                .zip(usage.cache_creation_input_tokens)
                .map(|(a, b)| a.saturating_add(b))
        })
        .or(usage.cached_input_tokens)
        .or(usage.cache_creation_input_tokens)
    {
        span.record("gen_ai.usage.input_tokens", input);
    }
    if let Some(output) = usage.output_tokens {
        span.record("gen_ai.usage.output_tokens", output);
    }
    if let Some(cache_read) = usage.cached_input_tokens {
        span.record("gen_ai.usage.cache_read.input_tokens", cache_read);
    }
    if let Some(cache_write) = usage.cache_creation_input_tokens {
        span.record("gen_ai.usage.cache_creation.input_tokens", cache_write);
    }
}

fn record_finish_reason(span: &Span, reason: &FinishReason) {
    let reasons = serde_json::to_string(&[reason.as_str()]).unwrap_or_default();
    span.record("gen_ai.response.finish_reasons", reasons.as_str());
}

#[cfg(test)]
mod tests {
    use super::otel_genai_provider_name;

    #[test]
    fn otel_provider_name_uses_standard_values_when_known() {
        assert_eq!(otel_genai_provider_name("openai"), "openai");
        assert_eq!(otel_genai_provider_name("anthropic"), "anthropic");
        assert_eq!(otel_genai_provider_name("gemini"), "gcp.gemini");
        assert_eq!(otel_genai_provider_name("alt_openai_compat"), "openai");
        assert_eq!(otel_genai_provider_name("cloudflare_worker"), "cloudflare");
    }

    #[test]
    fn otel_provider_name_preserves_unknown_custom_values() {
        assert_eq!(
            otel_genai_provider_name("custom_provider"),
            "custom_provider"
        );
    }
}
