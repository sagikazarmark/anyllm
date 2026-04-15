use std::fmt;

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream, Error,
    ProviderIdentity, Result,
};

/// A [`ChatProvider`] wrapper that delegates to a fallback provider when the
/// primary provider returns an error that qualifies for built-in failover.
///
/// This wrapper is intentionally request-preserving: if the primary fails
/// before producing a response, the exact same [`ChatRequest`] is forwarded to
/// the fallback provider.
///
/// That makes it a good fit for same-family failover, or for higher-level code
/// that already abstracts provider-specific model IDs and options before the
/// request reaches `anyllm`.
///
/// If you need heterogeneous failover across providers with different model
/// identifiers or request conventions, translate the request before it reaches
/// this wrapper so both providers receive a shape they understand.
pub struct FallbackChatProvider<P, F> {
    primary: P,
    fallback: F,
}

impl<P, F> FallbackChatProvider<P, F> {
    /// Wrap a primary provider with a fallback provider.
    ///
    /// The same [`ChatRequest`] is forwarded unchanged to the fallback when the
    /// primary fails with a fallback-eligible error.
    #[must_use]
    pub fn new(primary: P, fallback: F) -> Self {
        Self { primary, fallback }
    }

    /// Consume the wrapper and return the `(primary, fallback)` providers.
    #[must_use]
    pub fn into_parts(self) -> (P, F) {
        (self.primary, self.fallback)
    }
}

impl<P, F> ChatProvider for FallbackChatProvider<P, F>
where
    P: ChatProvider,
    F: ChatProvider,
    P::Stream: 'static,
    F::Stream: 'static,
{
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        match self.primary.chat(request).await {
            Ok(response) => Ok(response),
            Err(error) if should_fallback(&error) => {
                let primary_error = error;

                #[cfg(feature = "tracing")]
                record_fallback_activation(self.fallback.provider_name(), &primary_error);

                match self.fallback.chat(request).await {
                    Ok(response) => Ok(response),
                    Err(fallback_error) => Err(Error::Fallback {
                        primary: Box::new(primary_error),
                        fallback: Box::new(fallback_error),
                    }),
                }
            }
            Err(error) => Err(error),
        }
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        match self.primary.chat_stream(request).await {
            Ok(stream) => Ok(Box::pin(stream) as ChatStream),
            Err(error) if should_fallback(&error) => {
                let primary_error = error;

                #[cfg(feature = "tracing")]
                record_fallback_activation(self.fallback.provider_name(), &primary_error);

                match self.fallback.chat_stream(request).await {
                    Ok(stream) => Ok(Box::pin(stream) as ChatStream),
                    Err(fallback_error) => Err(Error::Fallback {
                        primary: Box::new(primary_error),
                        fallback: Box::new(fallback_error),
                    }),
                }
            }
            Err(error) => Err(error),
        }
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        merge_capability_support(
            self.primary.chat_capability(model, capability),
            self.fallback.chat_capability(model, capability),
        )
    }
}

impl<P, F> ProviderIdentity for FallbackChatProvider<P, F>
where
    P: ProviderIdentity,
    F: ProviderIdentity,
{
    fn provider_name(&self) -> &'static str {
        self.primary.provider_name()
    }
}

#[cfg(feature = "extract")]
impl<P, F> crate::ExtractExt for FallbackChatProvider<P, F>
where
    P: ChatProvider + Sync,
    F: ChatProvider + Sync,
    P::Stream: 'static,
    F::Stream: 'static,
{
}

impl<P, F> fmt::Debug for FallbackChatProvider<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FallbackChatProvider")
            .finish_non_exhaustive()
    }
}

fn should_fallback(error: &Error) -> bool {
    match error {
        Error::Auth(_)
        | Error::InvalidRequest(_)
        | Error::ModelNotFound(_)
        | Error::UnexpectedResponse(_)
        | Error::Serialization(_) => false,
        Error::Fallback { fallback, .. } => should_fallback(fallback),
        #[cfg(feature = "extract")]
        Error::Extract(_) => false,
        _ => true,
    }
}

fn merge_capability_support(
    primary: CapabilitySupport,
    fallback: CapabilitySupport,
) -> CapabilitySupport {
    match (primary, fallback) {
        // This wrapper is request-preserving and always tries the primary first,
        // so capability answers must stay conservative. If the primary knows it
        // cannot handle a feature, callers should not rely on the fallback path
        // to make that request shape valid.
        (CapabilitySupport::Unsupported, _) => CapabilitySupport::Unsupported,
        (CapabilitySupport::Supported, CapabilitySupport::Supported) => {
            CapabilitySupport::Supported
        }
        (CapabilitySupport::Supported, CapabilitySupport::Unknown)
        | (CapabilitySupport::Unknown, CapabilitySupport::Supported)
        | (CapabilitySupport::Unknown, CapabilitySupport::Unknown)
        | (CapabilitySupport::Unknown, CapabilitySupport::Unsupported)
        | (CapabilitySupport::Supported, CapabilitySupport::Unsupported) => {
            CapabilitySupport::Unknown
        }
    }
}

#[cfg(feature = "tracing")]
fn record_fallback_activation(provider_name: &str, error: &Error) {
    let span = tracing::Span::current();
    span.record("gen_ai.fallback.used", true);
    span.record("gen_ai.fallback.provider", provider_name);
    span.record("gen_ai.fallback.error_type", error.telemetry_type());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CapabilitySupport, ChatCapability, ChatRequestRecord, ChatStreamExt, Message, MockProvider,
        MockStreamEvent, MockStreamingProvider, StreamBlockType, StreamEvent,
    };

    #[tokio::test]
    async fn returns_primary_response_without_using_fallback() {
        let primary = MockProvider::with_text("primary").with_provider_name("primary");
        let fallback = MockProvider::with_text("fallback").with_provider_name("fallback");

        let wrapper = FallbackChatProvider::new(primary, fallback);
        let response = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap();

        assert_eq!(response.text().as_deref(), Some("primary"));
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 0);
    }

    #[tokio::test]
    async fn falls_back_when_default_policy_allows_chat_error() {
        let primary = MockProvider::with_error(Error::Overloaded {
            message: "busy".into(),
            retry_after: None,
            request_id: None,
        })
        .with_provider_name("primary");
        let fallback = MockProvider::with_text("fallback").with_provider_name("fallback");

        let wrapper = FallbackChatProvider::new(primary, fallback);
        let response = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap();

        assert_eq!(response.text().as_deref(), Some("fallback"));
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn forwards_same_request_to_fallback_provider() {
        let primary = MockProvider::with_error(Error::Timeout("slow".into()));
        let fallback = MockProvider::with_text("fallback");
        let wrapper = FallbackChatProvider::new(primary, fallback);
        let request = ChatRequest::new("test-model")
            .message(Message::system("follow instructions"))
            .message(Message::user("hi"))
            .temperature(0.2)
            .max_tokens(128)
            .stop(["done"])
            .parallel_tool_calls(true);

        let response = wrapper.chat(&request).await.unwrap();

        assert_eq!(response.text().as_deref(), Some("fallback"));

        let expected = ChatRequestRecord::from(&request);
        let primary_requests = wrapper.primary.requests();
        let fallback_requests = wrapper.fallback.requests();

        assert_eq!(primary_requests.len(), 1);
        assert_eq!(fallback_requests.len(), 1);
        assert_eq!(ChatRequestRecord::from(&primary_requests[0]), expected);
        assert_eq!(ChatRequestRecord::from(&fallback_requests[0]), expected);
    }

    #[tokio::test]
    async fn does_not_fallback_on_non_eligible_chat_error() {
        let primary =
            MockProvider::with_error(Error::Auth("bad key".into())).with_provider_name("primary");
        let fallback = MockProvider::with_text("fallback").with_provider_name("fallback");

        let wrapper = FallbackChatProvider::new(primary, fallback);
        let err = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Auth(_)));
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 0);
    }

    #[tokio::test]
    async fn eligible_chat_fallback_propagates_fallback_error() {
        let primary = MockProvider::with_error(Error::Timeout("primary timeout".into()));
        let fallback = MockProvider::with_error(Error::RateLimited {
            message: "fallback rate limited".into(),
            retry_after: None,
            request_id: None,
        });

        let wrapper = FallbackChatProvider::new(primary, fallback);
        let err = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap_err();

        match err {
            Error::Fallback { primary, fallback } => {
                assert!(
                    matches!(*primary, Error::Timeout(ref message) if message == "primary timeout")
                );
                match *fallback {
                    Error::RateLimited { message, .. } => {
                        assert_eq!(message, "fallback rate limited");
                    }
                    other => panic!("expected rate-limited fallback error, got {other:?}"),
                }
            }
            other => panic!("expected fallback chain error, got {other:?}"),
        }
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn returns_fallback_stream_for_eligible_stream_setup_error() {
        let primary = MockProvider::with_error(Error::Timeout("stream setup timeout".into()))
            .with_provider_name("primary");
        let fallback = MockStreamingProvider::with_text("fallback").with_provider_name("fallback");

        let wrapper = FallbackChatProvider::new(primary, fallback);
        let stream = wrapper
            .chat_stream(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap();
        let response = stream.collect_response().await.unwrap();

        assert_eq!(response.text().as_deref(), Some("fallback"));
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn eligible_stream_setup_fallback_propagates_fallback_error() {
        let primary = MockProvider::with_error(Error::Timeout("stream setup timeout".into()));
        let fallback = MockProvider::with_error(Error::Auth("fallback denied".into()));

        let wrapper = FallbackChatProvider::new(primary, fallback);
        let err = match wrapper
            .chat_stream(&ChatRequest::new("test").message(Message::user("hi")))
            .await
        {
            Ok(_) => panic!("expected chat_stream to return a fallback setup error"),
            Err(err) => err,
        };

        match err {
            Error::Fallback { primary, fallback } => {
                assert!(
                    matches!(*primary, Error::Timeout(ref message) if message == "stream setup timeout")
                );
                match *fallback {
                    Error::Auth(message) => assert_eq!(message, "fallback denied"),
                    other => panic!("expected auth fallback error, got {other:?}"),
                }
            }
            other => panic!("expected fallback chain error, got {other:?}"),
        }
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 1);
    }

    #[tokio::test]
    async fn does_not_fallback_for_non_eligible_stream_setup_error() {
        let primary =
            MockProvider::with_error(Error::Auth("bad key".into())).with_provider_name("primary");
        let fallback = MockProvider::with_text("fallback").with_provider_name("fallback");

        let wrapper = FallbackChatProvider::new(primary, fallback);
        match wrapper
            .chat_stream(&ChatRequest::new("test").message(Message::user("hi")))
            .await
        {
            Err(Error::Auth(message)) => assert_eq!(message, "bad key"),
            Err(other) => panic!("expected auth error, got {other:?}"),
            Ok(_) => panic!("expected chat_stream to return an error"),
        }
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 0);
    }

    #[tokio::test]
    async fn midstream_errors_do_not_trigger_fallback() {
        let primary = MockStreamingProvider::with_stream([
            MockStreamEvent::Event(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("primary-model".into()),
            }),
            MockStreamEvent::Event(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            }),
            MockStreamEvent::Error(Error::Stream("broken pipe".into())),
        ]);
        let fallback = MockStreamingProvider::with_text("fallback");

        let wrapper = FallbackChatProvider::new(primary, fallback);
        let stream = wrapper
            .chat_stream(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap();
        let err = stream.collect_response().await.unwrap_err();

        match err {
            Error::Stream(message) => assert_eq!(message, "broken pipe"),
            other => panic!("expected mid-stream error, got {other:?}"),
        }
        assert_eq!(wrapper.primary.call_count(), 1);
        assert_eq!(wrapper.fallback.call_count(), 0);
    }

    #[test]
    fn reports_primary_identity_and_capabilities() {
        let primary = MockProvider::with_text("primary")
            .with_provider_name("primary")
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);
        let fallback = MockProvider::with_text("fallback")
            .with_provider_name("fallback")
            .with_supported_chat_capabilities([
                ChatCapability::Streaming,
                ChatCapability::ReasoningOutput,
            ]);

        let wrapper = FallbackChatProvider::new(primary, fallback);

        assert_eq!(wrapper.provider_name(), "primary");
        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::ToolCalls),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::Streaming),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::ReasoningOutput),
            CapabilitySupport::Unknown
        );
    }

    #[test]
    fn primary_unsupported_capability_dominates_fallback_support() {
        let primary = MockProvider::with_text("primary").with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        );
        let fallback = MockProvider::with_text("fallback").with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Supported,
        );

        let wrapper = FallbackChatProvider::new(primary, fallback);

        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::StructuredOutput),
            CapabilitySupport::Unsupported
        );
    }

    #[test]
    fn reports_supported_only_when_both_providers_support_capability() {
        let primary = MockProvider::with_text("primary")
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);
        let fallback = MockProvider::with_text("fallback")
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);

        let wrapper = FallbackChatProvider::new(primary, fallback);

        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn default_policy_falls_back_for_transient_and_alternate_provider_cases() {
        let cases = [
            Error::Provider {
                status: None,
                message: "conn refused".into(),
                body: None,
                request_id: None,
            },
            Error::Timeout("elapsed".into()),
            Error::RateLimited {
                message: "too fast".into(),
                retry_after: None,
                request_id: None,
            },
            Error::Overloaded {
                message: "degraded".into(),
                retry_after: None,
                request_id: None,
            },
            Error::Provider {
                status: Some(500),
                message: "ise".into(),
                body: None,
                request_id: None,
            },
            Error::Unsupported("no vision".into()),
            Error::ContextLengthExceeded {
                message: "too long".into(),
                max_tokens: Some(4096),
            },
            Error::ContentFiltered("blocked".into()),
            Error::Stream("broken pipe".into()),
        ];

        for error in cases {
            assert!(should_fallback(&error), "expected {error:?} to fallback");
        }
    }

    #[test]
    fn default_policy_rejects_non_fallback_errors() {
        let serialization = serde_json::from_str::<serde_json::Value>("parse error")
            .map(|_| unreachable!())
            .map_err(Error::from)
            .unwrap_err();

        assert!(!should_fallback(&Error::Auth("bad key".into())));
        assert!(!should_fallback(&Error::InvalidRequest(
            "missing field".into()
        )));
        assert!(!should_fallback(&Error::ModelNotFound(
            "gpt-unknown does not exist".into()
        )));
        assert!(!should_fallback(&Error::UnexpectedResponse(
            "missing text block".into()
        )));
        assert!(!should_fallback(&serialization));
    }

    #[cfg(feature = "extract")]
    #[test]
    fn default_policy_rejects_extract_errors() {
        let error = Error::Extract(Box::new(crate::ExtractError::MissingStructuredText {
            mode: crate::ExtractionMode::Native,
            provider: "mock".into(),
        }));

        assert!(!should_fallback(&error));
    }
}
