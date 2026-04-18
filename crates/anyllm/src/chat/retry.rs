use std::fmt;
use std::time::Duration;

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, Error,
    ProviderIdentity, Result,
};
use futures_timer::Delay;
#[cfg(feature = "tracing")]
use tracing::Span;

/// A [`ChatProvider`] wrapper that retries non-streaming `chat()` calls on
/// retryable errors with exponential backoff.
pub struct RetryingChatProvider<T> {
    inner: T,
    policy: RetryPolicy,
}

impl<T> fmt::Debug for RetryingChatProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryingChatProvider")
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl<T> RetryingChatProvider<T> {
    /// Wrap a provider with the default retry policy.
    #[must_use]
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            policy: RetryPolicy::default(),
        }
    }

    /// Replace the retry policy for this wrapper
    #[must_use]
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Borrow the currently configured retry policy.
    #[must_use]
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// Consume the wrapper and return the wrapped provider.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Consume the wrapper and return the wrapped provider plus retry policy.
    #[must_use]
    pub fn into_parts(self) -> (T, RetryPolicy) {
        (self.inner, self.policy)
    }
}

impl<T: ChatProvider> ChatProvider for RetryingChatProvider<T> {
    type Stream = T::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let max_attempts = self.policy.normalized_max_attempts();
        #[cfg(feature = "tracing")]
        {
            let span = Span::current();
            span.record("anyllm.retry.max_attempts", max_attempts as u64);
        }

        for attempt in 1..=max_attempts {
            match self.inner.chat(request).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    let should_retry =
                        attempt < max_attempts && self.policy.should_retry_error(&error);
                    if !should_retry {
                        return Err(error);
                    }

                    let delay = self.policy.delay_for_retry(&error, attempt);
                    #[cfg(feature = "tracing")]
                    record_retry_attempt(attempt + 1, &error, delay);
                    if !delay.is_zero() {
                        Delay::new(delay).await;
                    }
                }
            }
        }

        unreachable!("retry loop always returns before exhausting attempts")
    }

    /// Streaming calls are **not** retried and are forwarded directly to the
    /// inner provider. Retrying a stream is complex: a mid-stream failure would
    /// require discarding partial data and restarting from scratch, and the
    /// caller is better positioned to decide whether that is acceptable.
    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        self.inner.chat_stream(request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner.chat_capability(model, capability)
    }
}

impl<T: ProviderIdentity> ProviderIdentity for RetryingChatProvider<T> {
    fn provider_name(&self) -> &'static str {
        self.inner.provider_name()
    }
}

#[cfg(feature = "extract")]
impl<T: ChatProvider + Sync> crate::ExtractExt for RetryingChatProvider<T> {}

/// Policy controlling retry behavior for transient errors.
///
/// This is intentionally a plain configuration struct with public fields rather
/// than a builder-only type. Applications often tune retry behavior directly,
/// and the current fields do not carry invariants that justify hiding them.
///
/// Uses a `fn` pointer for `should_retry` (not a closure) so the type
/// remains `Copy`. For stateful retry logic, implement a custom
/// [`ChatProvider`] wrapper.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum total attempts including the first try
    pub max_attempts: usize,
    /// Initial backoff delay before retries
    pub base_delay: Duration,
    /// Upper bound for retry delays
    pub max_delay: Duration,
    /// Whether provider retry-after hints should override exponential backoff
    pub respect_retry_after: bool,
    /// Predicate used to decide whether an error should be retried
    pub should_retry: fn(&Error) -> bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(5),
            respect_retry_after: true,
            should_retry: default_should_retry,
        }
    }
}

impl RetryPolicy {
    #[must_use]
    fn normalized_max_attempts(&self) -> usize {
        self.max_attempts.max(1)
    }

    /// Returns whether the given error should be retried.
    #[must_use]
    fn should_retry_error(&self, error: &Error) -> bool {
        (self.should_retry)(error)
    }

    /// Computes the delay before the next retry attempt.
    #[must_use]
    fn delay_for_retry(&self, error: &Error, retry_index: usize) -> Duration {
        if self.respect_retry_after
            && let Some(retry_after) = retry_after_hint(error)
        {
            return min_duration(retry_after, self.max_delay);
        }

        exponential_backoff(self.base_delay, self.max_delay, retry_index)
    }
}

fn default_should_retry(error: &Error) -> bool {
    error.is_retryable()
}

#[cfg(feature = "tracing")]
fn record_retry_attempt(attempt: usize, error: &Error, delay: Duration) {
    let span = Span::current();
    span.record("anyllm.retry.used", true);
    span.record("anyllm.retry.attempts", attempt as u64);
    span.record("anyllm.retry.last_delay_ms", delay.as_millis() as u64);
    span.record("anyllm.retry.last_error_type", error.telemetry_type());
}

fn retry_after_hint(error: &Error) -> Option<Duration> {
    match error {
        Error::RateLimited { retry_after, .. } | Error::Overloaded { retry_after, .. } => {
            *retry_after
        }
        _ => None,
    }
}

fn exponential_backoff(base_delay: Duration, max_delay: Duration, retry_index: usize) -> Duration {
    if base_delay.is_zero() || max_delay.is_zero() {
        return Duration::ZERO;
    }

    let exponent = retry_index.saturating_sub(1).min(20) as u32;
    let multiplier = 1u128 << exponent;
    let base_ms = base_delay.as_millis();
    let max_ms = max_delay.as_millis();
    let delay_ms = base_ms.saturating_mul(multiplier).min(max_ms);
    Duration::from_millis(delay_ms as u64)
}

fn min_duration(a: Duration, b: Duration) -> Duration {
    if a <= b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapabilitySupport, ChatCapability, Message, MockProvider};

    #[tokio::test]
    async fn retries_retryable_error_until_success() {
        let model =
            MockProvider::build(|builder| builder.error(Error::Timeout("slow".into())).text("ok"));
        let model_handle = model.clone();

        let wrapper = RetryingChatProvider::new(model).with_policy(RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            ..RetryPolicy::default()
        });

        let response = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("ok"));
        assert_eq!(model_handle.call_count(), 2);
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable_error() {
        let model = MockProvider::with_error(Error::Auth("bad key".into()));
        let model_handle = model.clone();

        let wrapper = RetryingChatProvider::new(model).with_policy(RetryPolicy {
            max_attempts: 4,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            ..RetryPolicy::default()
        });

        let err = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Auth(_)));
        assert_eq!(model_handle.call_count(), 1);
    }

    #[tokio::test]
    async fn stops_after_max_attempts() {
        let model = MockProvider::new([Error::Timeout("one".into()), Error::Timeout("two".into())]);
        let model_handle = model.clone();

        let wrapper = RetryingChatProvider::new(model).with_policy(RetryPolicy {
            max_attempts: 2,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            ..RetryPolicy::default()
        });

        let err = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Timeout(ref msg) if msg == "two"));
        assert_eq!(model_handle.call_count(), 2);
    }

    #[tokio::test]
    async fn zero_max_attempts_is_normalized_to_one_attempt() {
        let model = MockProvider::with_error(Error::Timeout("slow".into()));
        let model_handle = model.clone();

        let wrapper = RetryingChatProvider::new(model).with_policy(RetryPolicy {
            max_attempts: 0,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            ..RetryPolicy::default()
        });

        let err = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Timeout(ref msg) if msg == "slow"));
        assert_eq!(model_handle.call_count(), 1);
    }

    #[tokio::test]
    async fn custom_should_retry_can_disable_retrying() {
        let model =
            MockProvider::build(|builder| builder.error(Error::Timeout("slow".into())).text("ok"));
        let model_handle = model.clone();

        let wrapper = RetryingChatProvider::new(model).with_policy(RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            should_retry: |_| false,
            ..RetryPolicy::default()
        });

        let err = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Timeout(ref msg) if msg == "slow"));
        assert_eq!(model_handle.call_count(), 1);
    }

    #[tokio::test]
    async fn custom_should_retry_can_enable_retrying_for_non_retryable_error() {
        let model =
            MockProvider::build(|builder| builder.error(Error::Auth("bad key".into())).text("ok"));
        let model_handle = model.clone();

        let wrapper = RetryingChatProvider::new(model).with_policy(RetryPolicy {
            max_attempts: 2,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            should_retry: |_| true,
            ..RetryPolicy::default()
        });

        let response = wrapper
            .chat(&ChatRequest::new("test").message(Message::user("hi")))
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("ok"));
        assert_eq!(model_handle.call_count(), 2);
    }

    #[tokio::test]
    async fn chat_stream_is_not_retried() {
        let model = MockProvider::with_error(Error::Timeout("stream timeout".into()));
        let model_handle = model.clone();

        let wrapper = RetryingChatProvider::new(model);
        let stream_result = wrapper
            .chat_stream(&ChatRequest::new("test").message(Message::user("hi")))
            .await;
        let err = match stream_result {
            Ok(_) => panic!("expected stream error"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Timeout(_)));
        assert_eq!(model_handle.call_count(), 1);
    }

    #[test]
    fn wrapper_accessors_expose_policy_and_ownership_recovery() {
        let wrapper =
            RetryingChatProvider::new(MockProvider::with_text("ok")).with_policy(RetryPolicy {
                max_attempts: 5,
                ..RetryPolicy::default()
            });

        assert_eq!(wrapper.policy().max_attempts, 5);

        let (inner, policy) = wrapper.into_parts();
        assert_eq!(inner.provider_name(), "mock");
        assert_eq!(policy.max_attempts, 5);
    }

    #[test]
    fn reports_inner_identity_and_capabilities() {
        let inner = MockProvider::with_text("ok")
            .with_provider_name("primary")
            .with_supported_chat_capabilities([
                ChatCapability::Streaming,
                ChatCapability::ToolCalls,
            ]);
        let wrapper = RetryingChatProvider::new(inner);

        assert_eq!(wrapper.provider_name(), "primary");
        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::Streaming),
            CapabilitySupport::Supported
        );
        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(
            wrapper.chat_capability("test", ChatCapability::ReasoningOutput),
            CapabilitySupport::Unknown
        );
    }

    #[test]
    fn rate_limit_retry_after_is_clamped_by_max_delay() {
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(2),
            respect_retry_after: true,
            should_retry: default_should_retry,
        };

        let err = Error::RateLimited {
            message: "slow down".into(),
            retry_after: Some(Duration::from_secs(10)),
            request_id: None,
        };

        assert_eq!(policy.delay_for_retry(&err, 1), Duration::from_secs(2));
    }

    #[test]
    fn retry_after_hint_can_be_ignored() {
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(2),
            respect_retry_after: false,
            should_retry: default_should_retry,
        };

        let err = Error::RateLimited {
            message: "slow down".into(),
            retry_after: Some(Duration::from_secs(10)),
            request_id: None,
        };

        assert_eq!(policy.delay_for_retry(&err, 1), Duration::from_millis(10));
    }

    #[test]
    fn exponential_backoff_grows_and_caps() {
        let policy = RetryPolicy {
            max_attempts: 4,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(250),
            respect_retry_after: false,
            should_retry: default_should_retry,
        };

        let err = Error::Timeout("slow".into());

        assert_eq!(policy.delay_for_retry(&err, 1), Duration::from_millis(100));
        assert_eq!(policy.delay_for_retry(&err, 2), Duration::from_millis(200));
        assert_eq!(policy.delay_for_retry(&err, 3), Duration::from_millis(250));
    }
}
