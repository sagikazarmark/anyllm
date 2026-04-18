#![cfg(all(feature = "tracing", feature = "mock"))]

use anyllm::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse,
    ChatResponseBuilder, ChatStreamExt, ContentBlock, ContentPart, ExtraMap, FallbackChatProvider,
    FinishReason, ImageSource, Message, MockProvider, MockStreamEvent, MockStreamingProvider,
    ProviderIdentity, RetryPolicy, RetryingChatProvider, StreamBlockType, StreamEvent,
    TracingChatProvider, TracingContentConfig, Usage, UsageMetadataMode,
};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::field::{Field, Visit};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

#[derive(Debug, Clone)]
struct RecordedSpan {
    name: String,
    fields: HashMap<String, String>,
}

#[derive(Default)]
struct FieldRecorder {
    fields: HashMap<String, String>,
}

impl Visit for FieldRecorder {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }
}

struct SpanRecorder {
    by_id: Arc<Mutex<HashMap<u64, RecordedSpan>>>,
    spans: Arc<Mutex<Vec<RecordedSpan>>>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for SpanRecorder {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = FieldRecorder::default();
        attrs.record(&mut visitor);
        self.by_id.lock().unwrap().insert(
            id.into_u64(),
            RecordedSpan {
                name: attrs.metadata().name().to_string(),
                fields: visitor.fields,
            },
        );
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if let Some(span) = self.by_id.lock().unwrap().get_mut(&id.into_u64()) {
            let mut visitor = FieldRecorder::default();
            values.record(&mut visitor);
            span.fields.extend(visitor.fields);
        }
    }

    fn on_close(&self, id: tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = self.by_id.lock().unwrap().remove(&id.into_u64()) {
            self.spans.lock().unwrap().push(span);
        }
    }
}

fn standard_provider() -> MockProvider {
    MockProvider::with_response(
        ChatResponseBuilder::new()
            .text("hello")
            .usage(10, 5)
            .model("test-model")
            .id("resp-123")
            .build(),
    )
    .with_supported_chat_capabilities([ChatCapability::Streaming, ChatCapability::ToolCalls])
    .with_provider_name("test_provider")
}

fn streaming_provider() -> MockStreamingProvider {
    MockStreamingProvider::with_stream(vec![
        StreamEvent::ResponseStart {
            id: Some("resp-stream".to_string()),
            model: Some("stream-model".to_string()),
        },
        StreamEvent::BlockStart {
            index: 0,
            block_type: StreamBlockType::Text,
            id: None,
            name: None,
            type_name: None,
            data: None,
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "hel".to_string(),
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "lo".to_string(),
        },
        StreamEvent::BlockStop { index: 0 },
        StreamEvent::ResponseMetadata {
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage::new().input_tokens(8).output_tokens(3)),
            usage_mode: UsageMetadataMode::Snapshot,
            id: None,
            model: None,
            metadata: serde_json::Map::new(),
        },
        StreamEvent::ResponseStop,
    ])
    .with_provider_name("test_streaming")
}

fn install_span_recorder() -> (
    Arc<Mutex<Vec<RecordedSpan>>>,
    tracing::subscriber::DefaultGuard,
) {
    let spans = Arc::new(Mutex::new(Vec::<RecordedSpan>::new()));
    let recorder = SpanRecorder {
        by_id: Arc::new(Mutex::new(HashMap::new())),
        spans: spans.clone(),
    };
    let subscriber = Registry::default().with(recorder);
    let guard = tracing::subscriber::set_default(subscriber);
    (spans, guard)
}

fn find_chat_span(spans: &Arc<Mutex<Vec<RecordedSpan>>>) -> RecordedSpan {
    spans
        .lock()
        .unwrap()
        .iter()
        .find(|s| s.name == "chat")
        .cloned()
        .expect("missing chat span")
}

fn parse_f64_field(span: &RecordedSpan, field: &str) -> f64 {
    span.fields
        .get(field)
        .unwrap_or_else(|| panic!("missing field {field}"))
        .parse::<f64>()
        .unwrap_or_else(|err| panic!("failed to parse {field} as f64: {err}"))
}

#[tokio::test]
async fn tracing_chat_provider_delegates_chat() {
    let provider = standard_provider();
    let model = TracingChatProvider::new(provider.clone());

    let request = ChatRequest::new("test-model").message(Message::user("hi"));
    let response = model.chat(&request).await.unwrap();

    // Verify the inner provider was called.
    assert_eq!(provider.call_count(), 1);

    // Verify response is passed through correctly
    assert_eq!(response.text(), Some("hello".to_string()));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.id, Some("resp-123".to_string()));

    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(5));
}

#[tokio::test]
async fn tracing_chat_provider_delegates_chat_stream() {
    let model = TracingChatProvider::new(streaming_provider());

    let request = ChatRequest::new("stream-model").message(Message::user("hi"));
    let stream = model.chat_stream(&request).await.unwrap();
    let response = stream.collect_response().await.unwrap();

    // Verify the stream produced the correct collected result
    assert_eq!(response.text(), Some("hello".to_string()));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));

    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(8));
    assert_eq!(usage.output_tokens, Some(3));
}

#[tokio::test]
async fn tracing_chat_provider_records_genai_attributes() {
    let (spans, _guard) = install_span_recorder();

    let model = TracingChatProvider::new(standard_provider().with_provider_name("gemini"));

    let request = ChatRequest::new("test-model")
        .message(Message::user("hi"))
        .max_tokens(42)
        .temperature(0.2)
        .top_p(0.9)
        .frequency_penalty(0.1)
        .presence_penalty(0.2)
        .seed(7)
        .stop(["END"]);

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);

    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.operation.name")
            .map(String::as_str),
        Some("chat")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.provider.name")
            .map(String::as_str),
        Some("gcp.gemini")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.request.model")
            .map(String::as_str),
        Some("test-model")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.response.model")
            .map(String::as_str),
        Some("test-model")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.input_tokens")
            .map(String::as_str),
        Some("10")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.output_tokens")
            .map(String::as_str),
        Some("5")
    );
}

#[tokio::test]
async fn tracing_chat_provider_opt_in_captures_truncated_messages() {
    let (spans, _guard) = install_span_recorder();

    let model =
        TracingChatProvider::new(standard_provider()).with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: true,
            max_payload_chars: 24,
        });

    let request = ChatRequest::new("test-model").message(Message::user(
        "this is a very long message that should be truncated",
    ));

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);

    let input = chat_span
        .fields
        .get("gen_ai.input.messages")
        .expect("missing input messages");
    assert!(input.contains("[TRUNCATED]"));

    let output = chat_span
        .fields
        .get("gen_ai.output.messages")
        .expect("missing output messages");
    assert!(output.contains("[TRUNCATED]"));
}

#[tokio::test]
async fn tracing_records_retry_attributes() {
    let (spans, _guard) = install_span_recorder();

    let inner = RetryingChatProvider::new(MockProvider::build(|builder| {
        builder
            .provider_name("retry_provider")
            .error(anyllm::Error::Timeout("slow".into()))
            .response(
                ChatResponseBuilder::new()
                    .text("recovered")
                    .model("retry-model")
                    .id("retry-1")
                    .build(),
            )
    }))
    .with_policy(RetryPolicy {
        max_attempts: 2,
        base_delay: std::time::Duration::ZERO,
        max_delay: std::time::Duration::ZERO,
        ..RetryPolicy::default()
    });
    let model = TracingChatProvider::new(inner);

    let _ = model
        .chat(&ChatRequest::new("retry-model").message(Message::user("hi")))
        .await
        .unwrap();

    let chat_span = find_chat_span(&spans);

    assert_eq!(
        chat_span
            .fields
            .get("anyllm.retry.max_attempts")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        chat_span
            .fields
            .get("anyllm.retry.used")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        chat_span
            .fields
            .get("anyllm.retry.attempts")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        chat_span
            .fields
            .get("anyllm.retry.last_error_type")
            .map(String::as_str),
        Some("timeout")
    );
}

#[tokio::test]
async fn tracing_records_fallback_attributes() {
    let (spans, _guard) = install_span_recorder();

    let primary = MockProvider::with_error(anyllm::Error::Overloaded {
        message: "busy".into(),
        retry_after: None,
        request_id: None,
    })
    .with_provider_name("primary_provider");
    let fallback = MockProvider::with_response(
        ChatResponseBuilder::new()
            .text("fallback ok")
            .model("fallback-model")
            .id("fallback-1")
            .build(),
    )
    .with_provider_name("fallback_provider");
    let model = TracingChatProvider::new(FallbackChatProvider::new(primary, fallback));

    let _ = model
        .chat(&ChatRequest::new("fallback-model").message(Message::user("hi")))
        .await
        .unwrap();

    let chat_span = find_chat_span(&spans);

    assert_eq!(
        chat_span
            .fields
            .get("anyllm.fallback.used")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        chat_span
            .fields
            .get("anyllm.fallback.provider")
            .map(String::as_str),
        Some("fallback_provider")
    );
    assert_eq!(
        chat_span
            .fields
            .get("anyllm.fallback.error_type")
            .map(String::as_str),
        Some("overloaded")
    );
}

#[test]
fn tracing_chat_provider_exposes_content_config_and_parts() {
    let config = TracingContentConfig {
        capture_input_messages: true,
        capture_output_messages: false,
        max_payload_chars: 128,
    };
    let model =
        TracingChatProvider::new(MockProvider::with_text("hello")).with_content_capture(config);

    assert!(model.content_config().capture_input_messages);
    assert_eq!(model.content_config().max_payload_chars, 128);

    let (inner, config) = model.into_parts();
    assert_eq!(inner.provider_name(), "mock");
    assert!(config.capture_input_messages);
    assert_eq!(config.max_payload_chars, 128);
}

#[test]
fn tracing_chat_provider_supports_ownership_recovery() {
    let model = TracingChatProvider::new(
        MockProvider::with_text("updated").with_provider_name("updated_mock"),
    );

    let inner = model.into_inner();
    assert_eq!(inner.provider_name(), "updated_mock");
}

#[test]
fn tracing_chat_provider_reports_inner_identity_and_capabilities() {
    let inner = MockProvider::with_text("hello")
        .with_provider_name("traced_provider")
        .with_supported_chat_capabilities([ChatCapability::ToolCalls, ChatCapability::Streaming]);
    let model = TracingChatProvider::new(inner);

    assert_eq!(model.provider_name(), "traced_provider");
    assert_eq!(
        model.chat_capability("test", ChatCapability::ToolCalls),
        CapabilitySupport::Supported
    );
    assert_eq!(
        model.chat_capability("test", ChatCapability::Streaming),
        CapabilitySupport::Supported
    );
    assert_eq!(
        model.chat_capability("test", ChatCapability::ReasoningOutput),
        CapabilitySupport::Unknown
    );
}

#[test]
fn tracing_chat_provider_debug_is_non_exhaustive() {
    let model = TracingChatProvider::new(MockProvider::with_text("hello"));
    let debug = format!("{model:?}");
    assert!(debug.contains("TracingChatProvider"));
    assert!(debug.contains("content_config"));
    assert!(debug.contains(".."));
}

#[tokio::test]
async fn tracing_chat_provider_records_json_output_type_and_redacts_sensitive_content() {
    let (spans, _guard) = install_span_recorder();

    let model = TracingChatProvider::new(MockProvider::with_response(
        ChatResponseBuilder::new().text("ok").build(),
    ))
    .with_content_capture(TracingContentConfig {
        capture_input_messages: true,
        capture_output_messages: false,
        max_payload_chars: 4096,
    });

    let request = ChatRequest::new("json-model")
        .message(Message::user("hello").with_extension("token", serde_json::json!("top-secret")))
        .response_format(anyllm::ResponseFormat::JsonSchema {
            name: Some("tool_output".into()),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {
                            "password": { "type": "string" }
                        }
                    }
                }
            }),
            strict: Some(true),
        });

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.output.type")
            .map(String::as_str),
        Some("json")
    );

    let input = chat_span
        .fields
        .get("gen_ai.input.messages")
        .expect("missing input messages");
    assert!(input.contains("[REDACTED]"));
    assert!(!input.contains("top-secret"));
}

#[tokio::test]
async fn tracing_chat_provider_records_chat_errors() {
    let (spans, _guard) = install_span_recorder();

    let model = TracingChatProvider::new(MockProvider::with_error(anyllm::Error::Timeout(
        "slow".into(),
    )));

    let err = model
        .chat(&ChatRequest::new("test-model").message(Message::user("hi")))
        .await
        .unwrap_err();
    assert!(matches!(err, anyllm::Error::Timeout(message) if message == "slow"));

    let chat_span = find_chat_span(&spans);
    assert_eq!(
        chat_span.fields.get("error.type").map(String::as_str),
        Some("timeout")
    );
}

#[tokio::test]
async fn tracing_chat_provider_records_stream_usage_finish_reason_and_errors() {
    let (spans, _guard) = install_span_recorder();

    let provider = MockStreamingProvider::with_stream(vec![
        MockStreamEvent::from(StreamEvent::ResponseStart {
            id: Some("resp-stream".to_string()),
            model: Some("stream-model".to_string()),
        }),
        MockStreamEvent::from(anyllm::Error::Timeout("slow stream".into())),
    ]);

    let model = TracingChatProvider::new(provider);
    let request = ChatRequest::new("stream-model").message(Message::user("hi"));
    let stream = model.chat_stream(&request).await.unwrap();

    match stream.collect_response().await {
        Err(anyllm::Error::Timeout(message)) => assert_eq!(message, "slow stream"),
        Err(other) => panic!("expected timeout error, got {other:?}"),
        Ok(_) => panic!("expected collect_response to return an error"),
    }

    let chat_span = find_chat_span(&spans);
    assert_eq!(
        chat_span.fields.get("error.type").map(String::as_str),
        Some("timeout")
    );
}

#[tokio::test]
async fn tracing_chat_provider_records_stream_finish_reason_on_completion() {
    let (spans, _guard) = install_span_recorder();

    let provider = MockStreamingProvider::with_stream(vec![
        StreamEvent::ResponseStart {
            id: Some("resp-stream".to_string()),
            model: Some("stream-model".to_string()),
        },
        StreamEvent::ResponseMetadata {
            finish_reason: Some(FinishReason::Length),
            usage: Some(Usage::new().input_tokens(4).output_tokens(2)),
            usage_mode: UsageMetadataMode::Snapshot,
            id: None,
            model: None,
            metadata: serde_json::Map::new(),
        },
        StreamEvent::ResponseStop,
    ]);

    let model = TracingChatProvider::new(provider);
    let request = ChatRequest::new("stream-model").message(Message::user("hi"));
    let response = model
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Length));

    let chat_span = find_chat_span(&spans);
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.response.finish_reasons")
            .map(String::as_str),
        Some("[\"length\"]")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.input_tokens")
            .map(String::as_str),
        Some("4")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.output_tokens")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.response.id")
            .map(String::as_str),
        Some("resp-stream")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.response.model")
            .map(String::as_str),
        Some("stream-model")
    );
}

#[tokio::test]
async fn tracing_chat_provider_records_stream_cache_usage_on_completion() {
    let (spans, _guard) = install_span_recorder();

    let provider = MockStreamingProvider::with_stream(vec![
        StreamEvent::ResponseStart {
            id: Some("resp-stream".to_string()),
            model: Some("stream-model".to_string()),
        },
        StreamEvent::ResponseMetadata {
            finish_reason: Some(FinishReason::Stop),
            usage: Some(
                Usage::new()
                    .cached_input_tokens(2)
                    .cache_creation_input_tokens(1),
            ),
            usage_mode: UsageMetadataMode::Snapshot,
            id: None,
            model: None,
            metadata: serde_json::Map::new(),
        },
        StreamEvent::ResponseStop,
    ]);

    let model = TracingChatProvider::new(provider);
    let request = ChatRequest::new("stream-model").message(Message::user("hi"));
    let _ = model
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();

    let chat_span = find_chat_span(&spans);
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.input_tokens")
            .map(String::as_str),
        Some("3")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.cache_read.input_tokens")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.cache_creation.input_tokens")
            .map(String::as_str),
        Some("1")
    );
}

#[tokio::test]
async fn tracing_chat_provider_flushes_stream_usage_when_dropped_early() {
    let (spans, _guard) = install_span_recorder();

    let provider = MockStreamingProvider::with_stream(vec![
        MockStreamEvent::from(StreamEvent::ResponseStart {
            id: Some("resp-early".to_string()),
            model: Some("stream-model".to_string()),
        }),
        MockStreamEvent::from(StreamEvent::ResponseMetadata {
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage::new().input_tokens(7).output_tokens(3)),
            usage_mode: UsageMetadataMode::Snapshot,
            id: None,
            model: None,
            metadata: serde_json::Map::new(),
        }),
        MockStreamEvent::delayed(std::time::Duration::from_millis(20)),
        MockStreamEvent::from(StreamEvent::TextDelta {
            index: 0,
            text: "ignored".into(),
        }),
    ]);

    let model = TracingChatProvider::new(provider);
    let request = ChatRequest::new("stream-model").message(Message::user("hi"));
    let mut stream = model.chat_stream(&request).await.unwrap();

    while let Some(item) = stream.next().await {
        let event = item.unwrap();
        if matches!(event, StreamEvent::ResponseMetadata { .. }) {
            break;
        }
    }

    drop(stream);

    let chat_span = find_chat_span(&spans);
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.input_tokens")
            .map(String::as_str),
        Some("7")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.usage.output_tokens")
            .map(String::as_str),
        Some("3")
    );
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.response.id")
            .map(String::as_str),
        Some("resp-early")
    );
}

#[tokio::test]
async fn tracing_chat_provider_records_stream_ttft_and_partial_output_capture() {
    let (spans, _guard) = install_span_recorder();

    let provider = MockStreamingProvider::with_stream(vec![
        MockStreamEvent::from(StreamEvent::ResponseStart {
            id: Some("resp-ttft".to_string()),
            model: Some("stream-model".to_string()),
        }),
        MockStreamEvent::from(StreamEvent::BlockStart {
            index: 0,
            block_type: StreamBlockType::Text,
            id: None,
            name: None,
            type_name: None,
            data: None,
        }),
        MockStreamEvent::delayed(std::time::Duration::from_millis(10)),
        MockStreamEvent::from(StreamEvent::TextDelta {
            index: 0,
            text: "hello".into(),
        }),
        MockStreamEvent::from(StreamEvent::BlockStop { index: 0 }),
        MockStreamEvent::from(StreamEvent::ResponseMetadata {
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage::new().input_tokens(4).output_tokens(1)),
            usage_mode: UsageMetadataMode::Snapshot,
            id: None,
            model: None,
            metadata: serde_json::Map::new(),
        }),
        MockStreamEvent::from(StreamEvent::ResponseStop),
    ]);

    let model = TracingChatProvider::new(provider).with_content_capture(TracingContentConfig {
        capture_input_messages: false,
        capture_output_messages: true,
        max_payload_chars: 4096,
    });

    let request = ChatRequest::new("stream-model").message(Message::user("hi"));
    let _ = model
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();

    let chat_span = find_chat_span(&spans);
    assert!(parse_f64_field(&chat_span, "anyllm.response.ttft_ms") >= 1.0);
    assert_eq!(
        chat_span
            .fields
            .get("gen_ai.response.id")
            .map(String::as_str),
        Some("resp-ttft")
    );

    let output = chat_span
        .fields
        .get("gen_ai.output.messages")
        .expect("missing output messages");
    assert!(output.contains("\"finish_reason\":\"stop\""));
    assert!(output.contains("\"hello\""));
}

#[tokio::test]
async fn tracing_sets_otel_name_override_for_span() {
    let (spans, _guard) = install_span_recorder();

    let model = TracingChatProvider::new(standard_provider());
    let _ = model
        .chat(&ChatRequest::new("gpt-4o-mini").message(Message::user("hi")))
        .await
        .unwrap();

    let chat_span = find_chat_span(&spans);
    assert_eq!(
        chat_span.fields.get("otel.name").map(String::as_str),
        Some("chat gpt-4o-mini"),
        "tracing-opentelemetry bridge needs otel.name to produce the spec-shaped span name"
    );
}

#[tokio::test]
async fn tracing_records_tool_result_message_in_otel_shape() {
    let (spans, _guard) = install_span_recorder();

    let model =
        TracingChatProvider::new(standard_provider()).with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: false,
            max_payload_chars: 4096,
        });

    let request = ChatRequest::new("test-model")
        .message(Message::user("use the tool"))
        .message(Message::tool_result("call_1", "lookup", "rainy, 57°F"));

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);
    let input_raw = chat_span
        .fields
        .get("gen_ai.input.messages")
        .expect("missing input messages");
    let input: serde_json::Value = serde_json::from_str(input_raw).expect("input payload is JSON");
    let messages = input.as_array().expect("input is array");
    assert_eq!(messages.len(), 2);

    let tool_msg = &messages[1];
    assert_eq!(tool_msg["role"], "tool");
    let parts = tool_msg["parts"].as_array().expect("parts is array");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0]["type"], "tool_call_response");
    assert_eq!(parts[0]["id"], "call_1");
    assert_eq!(parts[0]["response"], "rainy, 57°F");
    assert_eq!(
        parts[0]["anyllm.tool_name"], "lookup",
        "tool name must be preserved under the crate-local key"
    );
}

#[tokio::test]
async fn tracing_records_messages_in_otel_semconv_shape() {
    let (spans, _guard) = install_span_recorder();

    let tool_call_response = ChatResponseBuilder::new()
        .tool_call(
            "call_1",
            "lookup",
            serde_json::json!({"city": "Paris", "api_key": "top-secret"}),
        )
        .finish_reason(FinishReason::ToolCalls)
        .build();

    let model = TracingChatProvider::new(MockProvider::with_response(tool_call_response))
        .with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: true,
            max_payload_chars: 4096,
        });

    let request = ChatRequest::new("test-model").message(Message::user("Weather in Paris?"));
    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);

    let input_raw = chat_span
        .fields
        .get("gen_ai.input.messages")
        .expect("missing input messages");
    let input: serde_json::Value = serde_json::from_str(input_raw).expect("input payload is JSON");
    let input_arr = input.as_array().expect("input is array");
    assert_eq!(input_arr.len(), 1);
    let user = &input_arr[0];
    assert_eq!(user["role"], "user");
    let parts = user["parts"].as_array().expect("parts is array");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["content"], "Weather in Paris?");
    assert!(
        parts[0].get("text").is_none(),
        "legacy 'text' field must not be emitted"
    );
    assert!(
        user.get("content").is_none(),
        "legacy message-level 'content' field must not be emitted"
    );

    let output_raw = chat_span
        .fields
        .get("gen_ai.output.messages")
        .expect("missing output messages");
    let output: serde_json::Value =
        serde_json::from_str(output_raw).expect("output payload is JSON");
    let assistant = &output[0];
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["finish_reason"], "tool_calls");
    let parts = assistant["parts"].as_array().expect("parts is array");
    assert_eq!(parts[0]["type"], "tool_call");
    assert_eq!(parts[0]["id"], "call_1");
    assert_eq!(parts[0]["name"], "lookup");

    let args = &parts[0]["arguments"];
    assert!(args.is_object(), "tool_call arguments must be an object");
    assert_eq!(args["city"], "Paris");
    assert_eq!(args["api_key"], "[REDACTED]");
}

#[tokio::test]
async fn tracing_records_system_instructions_in_otel_shape() {
    let (spans, _guard) = install_span_recorder();

    let model =
        TracingChatProvider::new(standard_provider()).with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: false,
            max_payload_chars: 4096,
        });

    let request = ChatRequest::new("test-model")
        .system("You are concise.")
        .system("Answer in English.")
        .message(Message::user("hi"));

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);
    let raw = chat_span
        .fields
        .get("gen_ai.system_instructions")
        .expect("missing system instructions");
    let value: serde_json::Value = serde_json::from_str(raw).expect("system payload is JSON");
    let parts = value.as_array().expect("system instructions is array");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["content"], "You are concise.");
    assert_eq!(parts[1]["content"], "Answer in English.");
}

#[tokio::test]
async fn tracing_omits_system_instructions_when_capture_disabled_or_empty() {
    let (spans, _guard) = install_span_recorder();

    let captured =
        TracingChatProvider::new(standard_provider()).with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: false,
            max_payload_chars: 4096,
        });
    let _ = captured
        .chat(&ChatRequest::new("test-model").message(Message::user("hi")))
        .await
        .unwrap();

    let uncaptured =
        TracingChatProvider::new(standard_provider()).with_content_capture(TracingContentConfig {
            capture_input_messages: false,
            capture_output_messages: false,
            max_payload_chars: 4096,
        });
    let _ = uncaptured
        .chat(
            &ChatRequest::new("test-model")
                .system("Be concise.")
                .message(Message::user("hi")),
        )
        .await
        .unwrap();

    for span in spans.lock().unwrap().iter().filter(|s| s.name == "chat") {
        assert!(
            !span.fields.contains_key("gen_ai.system_instructions"),
            "system instructions must not be recorded when empty or when capture is disabled"
        );
    }
}

#[tokio::test]
async fn tracing_records_image_parts_in_otel_shape() {
    let (spans, _guard) = install_span_recorder();

    let assistant_image = ChatResponse::new(vec![ContentBlock::Image {
        source: ImageSource::Url {
            url: "https://example.com/out.png".into(),
        },
    }])
    .finish_reason(FinishReason::Stop);

    let model = TracingChatProvider::new(MockProvider::with_response(assistant_image))
        .with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: true,
            max_payload_chars: 4096,
        });

    let request = ChatRequest::new("test-model").message(Message::user(vec![
        ContentPart::text("describe these"),
        ContentPart::Image {
            source: ImageSource::Url {
                url: "https://example.com/in.png".into(),
            },
            detail: Some("high".into()),
        },
        ContentPart::image_base64("image/png", "aGVsbG8="),
    ]));

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);

    let input: serde_json::Value = serde_json::from_str(
        chat_span
            .fields
            .get("gen_ai.input.messages")
            .expect("missing input messages"),
    )
    .expect("input payload is JSON");
    let parts = input[0]["parts"].as_array().expect("user parts array");
    assert_eq!(parts.len(), 3);

    assert_eq!(parts[1]["type"], "uri");
    assert_eq!(parts[1]["modality"], "image");
    assert_eq!(parts[1]["uri"], "https://example.com/in.png");
    assert_eq!(parts[1]["detail"], "high");
    assert!(parts[1].get("url").is_none(), "raw 'url' must not leak");

    assert_eq!(parts[2]["type"], "blob");
    assert_eq!(parts[2]["modality"], "image");
    assert_eq!(parts[2]["mime_type"], "image/png");
    assert_eq!(parts[2]["content"], "aGVsbG8=");
    assert!(parts[2].get("data").is_none(), "raw 'data' must not leak");

    let output: serde_json::Value = serde_json::from_str(
        chat_span
            .fields
            .get("gen_ai.output.messages")
            .expect("missing output messages"),
    )
    .expect("output payload is JSON");
    let out_parts = output[0]["parts"]
        .as_array()
        .expect("assistant parts array");
    assert_eq!(out_parts[0]["type"], "uri");
    assert_eq!(out_parts[0]["modality"], "image");
    assert_eq!(out_parts[0]["uri"], "https://example.com/out.png");
}

#[tokio::test]
async fn tracing_records_reasoning_blocks_in_otel_shape() {
    let (spans, _guard) = install_span_recorder();

    let response = ChatResponseBuilder::new()
        .reasoning("unsigned thought")
        .reasoning_with_signature("signed thought", "sig-abc")
        .text("final answer")
        .finish_reason(FinishReason::Stop)
        .build();

    let model = TracingChatProvider::new(MockProvider::with_response(response))
        .with_content_capture(TracingContentConfig {
            capture_input_messages: false,
            capture_output_messages: true,
            max_payload_chars: 4096,
        });

    let _ = model
        .chat(&ChatRequest::new("test-model").message(Message::user("think")))
        .await
        .unwrap();

    let chat_span = find_chat_span(&spans);
    let output: serde_json::Value = serde_json::from_str(
        chat_span
            .fields
            .get("gen_ai.output.messages")
            .expect("missing output messages"),
    )
    .expect("output payload is JSON");
    let parts = output[0]["parts"]
        .as_array()
        .expect("assistant parts array");
    assert_eq!(parts.len(), 3);

    assert_eq!(parts[0]["type"], "reasoning");
    assert_eq!(parts[0]["content"], "unsigned thought");
    assert!(parts[0].get("signature").is_none());
    assert!(
        parts[0].get("text").is_none(),
        "legacy reasoning 'text' key must not be emitted"
    );

    assert_eq!(parts[1]["type"], "reasoning");
    assert_eq!(parts[1]["content"], "signed thought");
    assert_eq!(parts[1]["signature"], "sig-abc");

    assert_eq!(parts[2]["type"], "text");
    assert_eq!(parts[2]["content"], "final answer");
}

#[tokio::test]
async fn tracing_records_streamed_output_in_otel_semconv_shape() {
    let (spans, _guard) = install_span_recorder();

    let provider = MockStreamingProvider::with_stream(vec![
        MockStreamEvent::from(StreamEvent::ResponseStart {
            id: Some("resp-stream".into()),
            model: Some("stream-model".into()),
        }),
        MockStreamEvent::from(StreamEvent::BlockStart {
            index: 0,
            block_type: StreamBlockType::Text,
            id: None,
            name: None,
            type_name: None,
            data: None,
        }),
        MockStreamEvent::from(StreamEvent::TextDelta {
            index: 0,
            text: "streamed answer".into(),
        }),
        MockStreamEvent::from(StreamEvent::BlockStop { index: 0 }),
        MockStreamEvent::from(StreamEvent::ResponseMetadata {
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage::new().input_tokens(2).output_tokens(3)),
            usage_mode: UsageMetadataMode::Snapshot,
            id: None,
            model: None,
            metadata: serde_json::Map::new(),
        }),
        MockStreamEvent::from(StreamEvent::ResponseStop),
    ]);

    let model = TracingChatProvider::new(provider).with_content_capture(TracingContentConfig {
        capture_input_messages: false,
        capture_output_messages: true,
        max_payload_chars: 4096,
    });

    let _ = model
        .chat_stream(&ChatRequest::new("stream-model").message(Message::user("hi")))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();

    let chat_span = find_chat_span(&spans);
    let output: serde_json::Value = serde_json::from_str(
        chat_span
            .fields
            .get("gen_ai.output.messages")
            .expect("missing output messages"),
    )
    .expect("output payload is JSON");

    let assistant = &output[0];
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["finish_reason"], "stop");

    let parts = assistant["parts"].as_array().expect("parts is array");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["content"], "streamed answer");
    assert!(
        parts[0].get("text").is_none(),
        "streamed text part must use OTEL `content`, not legacy `text`"
    );
    assert!(
        assistant.get("content").is_none(),
        "streamed assistant message must not emit a legacy message-level `content` field"
    );
}

#[tokio::test]
async fn tracing_records_other_content_via_generic_part_passthrough() {
    let (spans, _guard) = install_span_recorder();

    let mut assistant_data = ExtraMap::new();
    assistant_data.insert("citation_url".into(), "https://example.com".into());
    assistant_data.insert("confidence".into(), serde_json::json!(0.9));
    let assistant_response = ChatResponse::new(vec![ContentBlock::Other {
        type_name: "citation".into(),
        data: assistant_data,
    }])
    .finish_reason(FinishReason::Stop);

    let model = TracingChatProvider::new(MockProvider::with_response(assistant_response))
        .with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: true,
            max_payload_chars: 4096,
        });

    let mut user_data = ExtraMap::new();
    user_data.insert("format".into(), "wav".into());
    user_data.insert("data".into(), "base64-audio".into());
    let request = ChatRequest::new("test-model").message(Message::user(vec![ContentPart::Other {
        type_name: "input_audio".into(),
        data: user_data,
    }]));

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);

    let input: serde_json::Value = serde_json::from_str(
        chat_span
            .fields
            .get("gen_ai.input.messages")
            .expect("missing input messages"),
    )
    .expect("input payload is JSON");
    let user_part = &input[0]["parts"][0];
    assert_eq!(
        user_part["type"], "input_audio",
        "type_name must win as the part `type`"
    );
    assert_eq!(
        user_part["format"], "wav",
        "unknown keys must pass through verbatim"
    );
    assert_eq!(
        user_part["data"], "base64-audio",
        "unknown keys must pass through verbatim"
    );

    let output: serde_json::Value = serde_json::from_str(
        chat_span
            .fields
            .get("gen_ai.output.messages")
            .expect("missing output messages"),
    )
    .expect("output payload is JSON");
    let assistant_part = &output[0]["parts"][0];
    assert_eq!(assistant_part["type"], "citation");
    assert_eq!(assistant_part["citation_url"], "https://example.com");
    assert_eq!(assistant_part["confidence"], 0.9);
}

#[tokio::test]
async fn tracing_records_message_extensions_under_anyllm_namespace() {
    let (spans, _guard) = install_span_recorder();

    let model =
        TracingChatProvider::new(standard_provider()).with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: false,
            max_payload_chars: 4096,
        });

    let mut extensions = ExtraMap::new();
    extensions.insert(
        "cache_control".into(),
        serde_json::json!({"type": "ephemeral"}),
    );
    let request = ChatRequest::new("test-model").message(Message::User {
        content: anyllm::UserContent::Text("hi".into()),
        name: None,
        extensions: Some(extensions),
    });

    let _ = model.chat(&request).await.unwrap();

    let chat_span = find_chat_span(&spans);
    let input: serde_json::Value = serde_json::from_str(
        chat_span
            .fields
            .get("gen_ai.input.messages")
            .expect("missing input messages"),
    )
    .expect("input payload is JSON");
    let user = &input[0];
    assert!(
        user.get("extensions").is_none(),
        "bare `extensions` key must not be emitted at the message root"
    );
    assert_eq!(
        user["anyllm.extensions"]["cache_control"]["type"], "ephemeral",
        "vendor extensions must be namespaced under `anyllm.extensions`"
    );
}
