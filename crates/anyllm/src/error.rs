use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Convenience alias for `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for all LLM provider operations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Error {
    /// Request timed out while talking to the provider.
    ///
    /// Timeouts remain distinct because callers almost always handle them as
    /// transient retry candidates regardless of transport or provider details.
    Timeout(String),

    /// Authentication or authorization failure.
    Auth(String),

    /// Rate limited. Includes retry-after hint if the provider supplies one.
    RateLimited {
        /// Human-readable error message from the provider.
        message: String,
        /// Suggested delay before retrying, when reported.
        retry_after: Option<Duration>,
        /// Provider-assigned request ID for debugging with support teams.
        request_id: Option<String>,
    },

    /// Provider is overloaded or temporarily unavailable (503, Anthropic 529).
    /// Distinct from RateLimited: rate limits mean "you're sending too much,
    /// back off same provider." Overloaded means "the service is degraded,
    /// consider failing over to another provider."
    Overloaded {
        /// Human-readable error message from the provider.
        message: String,
        /// Suggested delay before retrying or failing over, when reported.
        retry_after: Option<Duration>,
        /// Provider-assigned request ID for debugging with support teams.
        request_id: Option<String>,
    },

    /// Provider communication failed outside a more specific category.
    ///
    /// `status: Some(_)` represents an HTTP error response from the provider.
    /// `status: None` represents failures where no HTTP status was available,
    /// such as connection-level transport errors or provider-native failures
    /// surfaced outside an HTTP response envelope.
    ///
    /// The HTTP-shaped fields here are best-effort diagnostics, not a promise
    /// that every provider is fundamentally HTTP-native. They exist because
    /// status codes, request IDs, and raw bodies are often the most useful data
    /// when debugging real integrations.
    Provider {
        /// HTTP status code when the failure came from an HTTP response.
        status: Option<u16>,
        /// Human-readable error message from the provider or transport layer.
        message: String,
        /// Raw response body when available.
        body: Option<String>,
        /// Provider-assigned request ID for debugging with support teams.
        request_id: Option<String>,
    },

    /// Serialization or deserialization failure.
    Serialization(SerializationError),

    /// The requested feature is not supported by this provider.
    Unsupported(String),

    /// The request was invalid.
    InvalidRequest(String),

    /// The requested model does not exist for this provider.
    ModelNotFound(String),

    /// The provider returned a response shape the caller did not expect.
    UnexpectedResponse(String),

    /// A fallback provider also failed after the primary provider errored.
    Fallback {
        /// Error returned by the primary provider.
        primary: Box<Error>,
        /// Error returned by the fallback provider.
        fallback: Box<Error>,
    },

    /// The input exceeds the model's context window.
    ///
    /// Distinct from InvalidRequest because the correct response is to
    /// truncate/summarize the input and retry, not fix the request shape.
    /// Critical for agent loops that manage conversation history.
    ContextLengthExceeded {
        /// Human-readable error message from the provider.
        message: String,
        /// Maximum tokens the model supports, if reported.
        max_tokens: Option<u64>,
    },

    /// Content filtered by provider safety systems.
    ContentFiltered(String),

    /// Error during streaming.
    Stream(String),

    /// Structured extraction failed after request execution.
    #[cfg(feature = "extract")]
    Extract(Box<crate::extract::ExtractError>),
}

impl Error {
    /// Build an [`Error::Serialization`] from any compatible source
    #[must_use]
    pub fn serialization(err: impl Into<SerializationError>) -> Self {
        Self::Serialization(err.into())
    }

    #[must_use]
    pub(crate) fn telemetry_type(&self) -> &'static str {
        match self {
            Error::Timeout(_) => "timeout",
            Error::Auth(_) => "auth",
            Error::RateLimited { .. } => "rate_limited",
            Error::Overloaded { .. } => "overloaded",
            Error::Provider { .. } => "provider",
            Error::Serialization(_) => "serialization",
            Error::Unsupported(_) => "unsupported",
            Error::InvalidRequest(_) => "invalid_request",
            Error::ModelNotFound(_) => "model_not_found",
            Error::UnexpectedResponse(_) => "unexpected_response",
            Error::Fallback { .. } => "fallback",
            Error::ContextLengthExceeded { .. } => "context_length_exceeded",
            Error::ContentFiltered(_) => "content_filtered",
            Error::Stream(_) => "stream",
            #[cfg(feature = "extract")]
            Error::Extract(_) => "extract",
        }
    }

    /// Returns true if this error is likely transient and worth retrying
    /// against the SAME provider.
    ///
    /// Covers: `RateLimited`, `Overloaded`, `Timeout`, provider failures with no
    /// HTTP status, and provider failures with 5xx status codes.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::RateLimited { .. } | Error::Overloaded { .. } | Error::Timeout(_) => true,
            Error::Provider { status: None, .. } => true,
            Error::Provider {
                status: Some(s), ..
            } => *s >= 500,
            #[cfg(feature = "extract")]
            Error::Extract(_) => false,
            Error::Fallback { fallback, .. } => fallback.is_retryable(),
            _ => false,
        }
    }

    /// Borrow a redacted logging view of this error
    #[must_use]
    pub fn as_log(&self) -> ErrorLog<'_> {
        ErrorLog::new(self)
    }
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr = match self {
            Error::Timeout(message) => SerializedError::Timeout {
                message: message.clone(),
            },
            Error::Auth(message) => SerializedError::Auth {
                message: message.clone(),
            },
            Error::RateLimited {
                message,
                retry_after,
                request_id,
            } => SerializedError::RateLimited {
                message: message.clone(),
                retry_after_secs: *retry_after,
                request_id: request_id.clone(),
            },
            Error::Overloaded {
                message,
                retry_after,
                request_id,
            } => SerializedError::Overloaded {
                message: message.clone(),
                retry_after_secs: *retry_after,
                request_id: request_id.clone(),
            },
            Error::Provider {
                status,
                message,
                body,
                request_id,
            } => SerializedError::Provider {
                status: *status,
                message: message.clone(),
                body: body.clone(),
                request_id: request_id.clone(),
            },
            Error::Serialization(err) => SerializedError::Serialization {
                message: err.to_string(),
            },
            Error::Unsupported(message) => SerializedError::Unsupported {
                message: message.clone(),
            },
            Error::InvalidRequest(message) => SerializedError::InvalidRequest {
                message: message.clone(),
            },
            Error::ModelNotFound(message) => SerializedError::ModelNotFound {
                message: message.clone(),
            },
            Error::UnexpectedResponse(message) => SerializedError::UnexpectedResponse {
                message: message.clone(),
            },
            Error::Fallback { primary, fallback } => SerializedError::Fallback {
                primary: primary.clone(),
                fallback: fallback.clone(),
            },
            Error::ContextLengthExceeded {
                message,
                max_tokens,
            } => SerializedError::ContextLengthExceeded {
                message: message.clone(),
                max_tokens: *max_tokens,
            },
            Error::ContentFiltered(message) => SerializedError::ContentFiltered {
                message: message.clone(),
            },
            Error::Stream(message) => SerializedError::Stream {
                message: message.clone(),
            },
            #[cfg(feature = "extract")]
            Error::Extract(error) => SerializedError::Extract {
                error: error.clone(),
            },
        };

        repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Error {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = SerializedError::deserialize(deserializer)?;
        Ok(match repr {
            SerializedError::Timeout { message } => Error::Timeout(message),
            SerializedError::Auth { message } => Error::Auth(message),
            SerializedError::RateLimited {
                message,
                retry_after_secs,
                request_id,
            } => Error::RateLimited {
                message,
                retry_after: retry_after_secs,
                request_id,
            },
            SerializedError::Overloaded {
                message,
                retry_after_secs,
                request_id,
            } => Error::Overloaded {
                message,
                retry_after: retry_after_secs,
                request_id,
            },
            SerializedError::Provider {
                status,
                message,
                body,
                request_id,
            } => Error::Provider {
                status,
                message,
                body,
                request_id,
            },
            SerializedError::Serialization { message } => Error::serialization(message),
            SerializedError::Unsupported { message } => Error::Unsupported(message),
            SerializedError::InvalidRequest { message } => Error::InvalidRequest(message),
            SerializedError::ModelNotFound { message } => Error::ModelNotFound(message),
            SerializedError::UnexpectedResponse { message } => Error::UnexpectedResponse(message),
            SerializedError::Fallback { primary, fallback } => {
                Error::Fallback { primary, fallback }
            }
            SerializedError::ContextLengthExceeded {
                message,
                max_tokens,
            } => Error::ContextLengthExceeded {
                message,
                max_tokens,
            },
            SerializedError::ContentFiltered { message } => Error::ContentFiltered(message),
            SerializedError::Stream { message } => Error::Stream(message),
            #[cfg(feature = "extract")]
            SerializedError::Extract { error } => Error::Extract(error),
        })
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Timeout(msg) => write!(f, "timeout: {msg}"),
            Error::Auth(msg) => write!(f, "authentication error: {msg}"),
            Error::RateLimited { message, .. } => write!(f, "rate limited: {message}"),
            Error::Overloaded { message, .. } => write!(f, "overloaded: {message}"),
            Error::Provider {
                message, status, ..
            } => {
                if let Some(status) = status {
                    write!(f, "provider error ({status}): {message}")
                } else {
                    write!(f, "provider error: {message}")
                }
            }
            Error::Serialization(err) => write!(f, "serialization error: {err}"),
            Error::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Error::InvalidRequest(msg) => write!(f, "invalid request: {msg}"),
            Error::ModelNotFound(msg) => write!(f, "model not found: {msg}"),
            Error::UnexpectedResponse(msg) => write!(f, "unexpected response: {msg}"),
            Error::Fallback { primary, fallback } => {
                write!(
                    f,
                    "fallback error: primary failed with {primary}; fallback failed with {fallback}"
                )
            }
            Error::ContextLengthExceeded { message, .. } => {
                write!(f, "context length exceeded: {message}")
            }
            Error::ContentFiltered(msg) => write!(f, "content filtered: {msg}"),
            Error::Stream(msg) => write!(f, "stream error: {msg}"),
            #[cfg(feature = "extract")]
            Error::Extract(err) => write!(f, "extraction error: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(err) => Some(err),
            Self::Fallback { fallback, .. } => Some(fallback.as_ref()),
            #[cfg(feature = "extract")]
            Self::Extract(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.into())
    }
}

/// Wrapper for serialization/deserialization failures surfaced through [`Error`].
///
/// This keeps the public error type cloneable and serializable while still
/// preserving the original error as the source when available.
#[derive(Debug)]
pub struct SerializationError(BoxError);

impl Clone for SerializationError {
    fn clone(&self) -> Self {
        self.to_string().into()
    }
}

impl Serialize for SerializationError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SerializationError {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let message = String::deserialize(deserializer)?;
        Ok(message.into())
    }
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

impl From<serde_json::Error> for SerializationError {
    fn from(err: serde_json::Error) -> Self {
        Self(Box::new(err))
    }
}

impl From<String> for SerializationError {
    fn from(message: String) -> Self {
        Self(Box::new(StringError(message)))
    }
}

impl From<&str> for SerializationError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

#[derive(Debug)]
struct StringError(String);

impl std::fmt::Display for StringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StringError {}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SerializedError {
    Timeout {
        message: String,
    },
    Auth {
        message: String,
    },
    RateLimited {
        message: String,
        #[serde(default, with = "option_duration_secs_f64")]
        retry_after_secs: Option<Duration>,
        #[serde(default)]
        request_id: Option<String>,
    },
    Overloaded {
        message: String,
        #[serde(default, with = "option_duration_secs_f64")]
        retry_after_secs: Option<Duration>,
        #[serde(default)]
        request_id: Option<String>,
    },
    Provider {
        #[serde(default)]
        status: Option<u16>,
        message: String,
        #[serde(default)]
        body: Option<String>,
        #[serde(default)]
        request_id: Option<String>,
    },
    Serialization {
        message: String,
    },
    Unsupported {
        message: String,
    },
    InvalidRequest {
        message: String,
    },
    ModelNotFound {
        message: String,
    },
    UnexpectedResponse {
        message: String,
    },
    Fallback {
        primary: Box<Error>,
        fallback: Box<Error>,
    },
    ContextLengthExceeded {
        message: String,
        #[serde(default)]
        max_tokens: Option<u64>,
    },
    ContentFiltered {
        message: String,
    },
    Stream {
        message: String,
    },
    #[cfg(feature = "extract")]
    Extract {
        error: Box<crate::extract::ExtractError>,
    },
}

/// Redacted logging/telemetry view of an [`Error`].
///
/// Unlike serde serialization of `Error` itself, this intentionally excludes raw
/// provider response bodies because they may contain prompts, model output,
/// credentials, or other sensitive data.
#[derive(Debug, Clone, Copy)]
pub struct ErrorLog<'a>(&'a Error);

impl<'a> ErrorLog<'a> {
    #[must_use]
    /// Wrap an error in its redacted logging representation.
    pub const fn new(error: &'a Error) -> Self {
        Self(error)
    }

    #[must_use]
    /// Return the original error referenced by this logging view.
    pub const fn error(self) -> &'a Error {
        self.0
    }
}

impl Serialize for ErrorLog<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let error = self.0;
        match error {
            Error::Timeout(msg) => telemetry_value(error.telemetry_type(), msg),
            Error::Auth(msg) => telemetry_value(error.telemetry_type(), msg),
            Error::RateLimited {
                message,
                retry_after,
                request_id,
            } => {
                let mut map = telemetry_map(error.telemetry_type(), message);
                insert_retry_after_secs(&mut map, *retry_after);
                insert_request_id(&mut map, request_id.as_deref());
                serde_json::Value::Object(map)
            }
            Error::Overloaded {
                message,
                retry_after,
                request_id,
            } => {
                let mut map = telemetry_map(error.telemetry_type(), message);
                insert_retry_after_secs(&mut map, *retry_after);
                insert_request_id(&mut map, request_id.as_deref());
                serde_json::Value::Object(map)
            }
            Error::Provider {
                status,
                message,
                body,
                request_id,
            } => {
                let mut map = telemetry_map(error.telemetry_type(), message);
                insert_optional(&mut map, "status", status.map(serde_json::Value::from));
                if let Some(b) = body {
                    map.insert("body_present".to_owned(), serde_json::Value::Bool(true));
                    map.insert("body_len".to_owned(), serde_json::Value::from(b.len()));
                }
                insert_request_id(&mut map, request_id.as_deref());
                serde_json::Value::Object(map)
            }
            Error::Serialization(err) => telemetry_value(error.telemetry_type(), &err.to_string()),
            Error::Unsupported(msg) => telemetry_value(error.telemetry_type(), msg),
            Error::InvalidRequest(msg) => telemetry_value(error.telemetry_type(), msg),
            Error::ModelNotFound(msg) => telemetry_value(error.telemetry_type(), msg),
            Error::UnexpectedResponse(msg) => telemetry_value(error.telemetry_type(), msg),
            Error::Fallback { primary, fallback } => serde_json::json!({
                "type": error.telemetry_type(),
                "message": fallback.to_string(),
                "primary": primary.as_log(),
                "fallback": fallback.as_log(),
            }),
            Error::ContextLengthExceeded {
                message,
                max_tokens,
            } => {
                let mut map = telemetry_map(error.telemetry_type(), message);
                insert_optional(
                    &mut map,
                    "max_tokens",
                    max_tokens.map(serde_json::Value::from),
                );
                serde_json::Value::Object(map)
            }
            Error::ContentFiltered(msg) => telemetry_value(error.telemetry_type(), msg),
            Error::Stream(msg) => telemetry_value(error.telemetry_type(), msg),
            #[cfg(feature = "extract")]
            Error::Extract(err) => telemetry_value(error.telemetry_type(), &err.to_string()),
        }
        .serialize(serializer)
    }
}

mod option_duration_secs_f64 {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(
        value: &Option<Duration>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value
            .map(|duration| duration.as_secs_f64())
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = Option::<f64>::deserialize(deserializer)?;
        Ok(secs.map(Duration::from_secs_f64))
    }
}

fn telemetry_value(kind: &'static str, message: &str) -> serde_json::Value {
    serde_json::Value::Object(telemetry_map(kind, message))
}

fn telemetry_map(kind: &'static str, message: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    map.insert("type".to_owned(), serde_json::Value::from(kind));
    map.insert("message".to_owned(), serde_json::Value::from(message));
    map
}

fn insert_optional(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<serde_json::Value>,
) {
    if let Some(value) = value {
        map.insert(key.to_owned(), value);
    }
}

fn insert_request_id(
    map: &mut serde_json::Map<String, serde_json::Value>,
    request_id: Option<&str>,
) {
    insert_optional(map, "request_id", request_id.map(serde_json::Value::from));
}

fn insert_retry_after_secs(
    map: &mut serde_json::Map<String, serde_json::Value>,
    retry_after: Option<Duration>,
) {
    insert_optional(
        map,
        "retry_after_secs",
        retry_after.map(|duration| serde_json::Value::from(duration.as_secs_f64())),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn serialization_error() -> Error {
        serde_json::from_str::<serde_json::Value>("invalid JSON")
            .map(|_| unreachable!())
            .map_err(Error::from)
            .unwrap_err()
    }

    #[test]
    fn simple_error_variants_have_expected_behavior() {
        let cases = vec![
            (
                Error::Provider {
                    status: None,
                    message: "connection refused".into(),
                    body: None,
                    request_id: None,
                },
                "provider error: connection refused",
                "provider",
                "connection refused",
                true,
            ),
            (
                Error::Timeout("30s elapsed".into()),
                "timeout: 30s elapsed",
                "timeout",
                "30s elapsed",
                true,
            ),
            (
                Error::Auth("invalid API key".into()),
                "authentication error: invalid API key",
                "auth",
                "invalid API key",
                false,
            ),
            (
                Error::Unsupported("vision not available".into()),
                "unsupported: vision not available",
                "unsupported",
                "vision not available",
                false,
            ),
            (
                Error::InvalidRequest("missing model field".into()),
                "invalid request: missing model field",
                "invalid_request",
                "missing model field",
                false,
            ),
            (
                Error::ModelNotFound("gpt-unknown does not exist".into()),
                "model not found: gpt-unknown does not exist",
                "model_not_found",
                "gpt-unknown does not exist",
                false,
            ),
            (
                Error::UnexpectedResponse("missing text block".into()),
                "unexpected response: missing text block",
                "unexpected_response",
                "missing text block",
                false,
            ),
            (
                Error::Fallback {
                    primary: Box::new(Error::Timeout("primary timed out".into())),
                    fallback: Box::new(Error::Auth("fallback rejected".into())),
                },
                "fallback error: primary failed with timeout: primary timed out; fallback failed with authentication error: fallback rejected",
                "fallback",
                "authentication error: fallback rejected",
                false,
            ),
            (
                Error::ContentFiltered("unsafe content detected".into()),
                "content filtered: unsafe content detected",
                "content_filtered",
                "unsafe content detected",
                false,
            ),
            (
                Error::Stream("unexpected EOF".into()),
                "stream error: unexpected EOF",
                "stream",
                "unexpected EOF",
                false,
            ),
        ];

        for (err, expected_display, expected_type, expected_message, retryable) in cases {
            assert_eq!(err.to_string(), expected_display);
            assert_eq!(err.is_retryable(), retryable);
            match &err {
                Error::Fallback { fallback, .. } => {
                    let source = std::error::Error::source(&err).unwrap();
                    assert_eq!(source.to_string(), fallback.to_string());
                }
                _ => assert!(std::error::Error::source(&err).is_none()),
            }

            let log = serde_json::to_value(err.as_log()).unwrap();
            assert_eq!(log["type"], expected_type);
            assert_eq!(log["message"], expected_message);
        }
    }

    #[test]
    fn fallback_error_logs_both_primary_and_fallback_failures() {
        let err = Error::Fallback {
            primary: Box::new(Error::Timeout("primary timed out".into())),
            fallback: Box::new(Error::RateLimited {
                message: "fallback rate limited".into(),
                retry_after: Some(Duration::from_secs(3)),
                request_id: Some("req-fallback".into()),
            }),
        };

        assert!(err.is_retryable());

        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["type"], "fallback");
        assert_eq!(log["primary"]["type"], "timeout");
        assert_eq!(log["primary"]["message"], "primary timed out");
        assert_eq!(log["fallback"]["type"], "rate_limited");
        assert_eq!(log["fallback"]["message"], "fallback rate limited");
        assert_eq!(log["fallback"]["retry_after_secs"], 3.0);
        assert_eq!(log["fallback"]["request_id"], "req-fallback");
    }

    #[test]
    fn rate_limit_and_overload_errors_include_metadata() {
        let rate_limited = Error::RateLimited {
            message: "too many requests".into(),
            retry_after: Some(Duration::from_secs(5)),
            request_id: Some("req-rate".into()),
        };
        assert!(rate_limited.is_retryable());
        assert_eq!(rate_limited.to_string(), "rate limited: too many requests");
        assert!(std::error::Error::source(&rate_limited).is_none());

        let rate_limited_log = serde_json::to_value(rate_limited.as_log()).unwrap();
        assert_eq!(rate_limited_log["type"], "rate_limited");
        assert_eq!(rate_limited_log["message"], "too many requests");
        assert_eq!(rate_limited_log["retry_after_secs"], 5.0);
        assert_eq!(rate_limited_log["request_id"], "req-rate");

        let overloaded = Error::Overloaded {
            message: "service degraded".into(),
            retry_after: None,
            request_id: Some("req-overload".into()),
        };
        assert!(overloaded.is_retryable());
        assert_eq!(overloaded.to_string(), "overloaded: service degraded");
        assert!(std::error::Error::source(&overloaded).is_none());

        let overloaded_log = serde_json::to_value(overloaded.as_log()).unwrap();
        assert_eq!(overloaded_log["type"], "overloaded");
        assert_eq!(overloaded_log["message"], "service degraded");
        assert_eq!(overloaded_log["request_id"], "req-overload");
        assert!(overloaded_log.get("retry_after_secs").is_none());
    }

    #[test]
    fn provider_errors_handle_status_retryability_and_redaction() {
        let retryable = Error::Provider {
            status: Some(500),
            message: "internal server error".into(),
            body: None,
            request_id: None,
        };
        assert!(retryable.is_retryable());
        assert_eq!(
            retryable.to_string(),
            "provider error (500): internal server error"
        );

        let non_retryable = Error::Provider {
            status: Some(400),
            message: "bad request".into(),
            body: Some("secret payload".into()),
            request_id: Some("req-123".into()),
        };
        assert!(!non_retryable.is_retryable());
        assert_eq!(
            non_retryable.to_string(),
            "provider error (400): bad request"
        );
        assert!(std::error::Error::source(&non_retryable).is_none());

        let no_status = Error::Provider {
            status: None,
            message: "unknown error".into(),
            body: None,
            request_id: None,
        };
        assert!(no_status.is_retryable());
        assert_eq!(no_status.to_string(), "provider error: unknown error");

        let log = serde_json::to_value(non_retryable.as_log()).unwrap();
        assert_eq!(log["type"], "provider");
        assert_eq!(log["message"], "bad request");
        assert_eq!(log["status"], 400);
        assert_eq!(log["request_id"], "req-123");
        assert_eq!(log["body_present"], true);
        assert_eq!(log["body_len"], "secret payload".len());
        assert!(log.get("body").is_none());
    }

    #[test]
    fn serialization_errors_wrap_source_and_log_message() {
        let err = serialization_error();
        assert!(!err.is_retryable());
        assert!(err.to_string().starts_with("serialization error:"));
        assert!(std::error::Error::source(&err).is_some());

        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["type"], "serialization");
        assert!(log["message"].as_str().unwrap().contains("expected value"));
    }

    #[test]
    fn context_length_exceeded_logs_max_tokens() {
        let err = Error::ContextLengthExceeded {
            message: "input too long".into(),
            max_tokens: Some(128_000),
        };
        assert!(!err.is_retryable());
        assert_eq!(err.to_string(), "context length exceeded: input too long");
        assert!(std::error::Error::source(&err).is_none());

        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["type"], "context_length_exceeded");
        assert_eq!(log["message"], "input too long");
        assert_eq!(log["max_tokens"], 128_000);
    }

    #[cfg(feature = "extract")]
    #[test]
    fn extract_errors_are_non_retryable_and_expose_source() {
        let err = Error::Extract(Box::new(
            crate::extract::ExtractError::MissingStructuredText {
                mode: crate::extract::ExtractionMode::Native,
                provider: "mock".into(),
            },
        ));
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("extraction error"));
        assert!(std::error::Error::source(&err).is_some());

        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["type"], "extract");
        assert!(
            log["message"]
                .as_str()
                .unwrap()
                .contains("did not return structured text")
        );
    }
}
