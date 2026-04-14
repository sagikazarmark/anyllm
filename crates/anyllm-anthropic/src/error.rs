use std::error::Error as StdError;
use std::io::ErrorKind;

use crate::wire;

pub(crate) fn is_timeout_error(err: &(dyn StdError + 'static)) -> bool {
    let mut current: Option<&(dyn StdError + 'static)> = Some(err);
    while let Some(err) = current {
        if let Some(reqwest_err) = err.downcast_ref::<reqwest::Error>()
            && reqwest_err.is_timeout()
        {
            return true;
        }

        if let Some(io_err) = err.downcast_ref::<std::io::Error>()
            && io_err.kind() == ErrorKind::TimedOut
        {
            return true;
        }

        let lower = err.to_string().to_ascii_lowercase();
        if lower.contains("timed out") || lower.contains("timeout") {
            return true;
        }

        current = err.source();
    }

    false
}

pub(crate) fn map_transport_error<E>(err: E) -> anyllm::Error
where
    E: StdError + Send + Sync + 'static,
{
    if is_timeout_error(&err) {
        anyllm::Error::Timeout(err.to_string())
    } else {
        anyllm::Error::Provider {
            status: None,
            message: err.to_string(),
            body: None,
            request_id: None,
        }
    }
}

pub(crate) fn map_response_deserialize_error<E>(err: E) -> anyllm::Error
where
    E: StdError + Send + Sync + 'static,
{
    if is_timeout_error(&err) {
        anyllm::Error::Timeout(err.to_string())
    } else {
        anyllm::Error::serialization(format!("Failed to deserialize response: {err}"))
    }
}

pub(crate) fn map_stream_error<E>(err: E) -> anyllm::Error
where
    E: StdError + Send + Sync + 'static,
{
    if is_timeout_error(&err) {
        anyllm::Error::Timeout(err.to_string())
    } else {
        anyllm::Error::Stream(format!("SSE error: {err}"))
    }
}

/// Map an Anthropic `error_type` string to a typed `anyllm::Error`.
///
/// Used for both HTTP error responses and mid-stream SSE error events,
/// so that retry/fallback policy works regardless of when the error arrives.
pub(crate) fn map_api_error(error: &wire::Error) -> anyllm::Error {
    match error.error_type.as_str() {
        "authentication_error" | "permission_error" => anyllm::Error::Auth(error.message.clone()),
        "rate_limit_error" => anyllm::Error::RateLimited {
            message: error.message.clone(),
            retry_after: None,
            request_id: None,
        },
        "overloaded_error" => anyllm::Error::Overloaded {
            message: error.message.clone(),
            retry_after: None,
            request_id: None,
        },
        "invalid_request_error" => {
            if error.message.contains("prompt is too long")
                || error.message.contains("too many tokens")
            {
                anyllm::Error::ContextLengthExceeded {
                    message: error.message.clone(),
                    max_tokens: None,
                }
            } else if is_model_not_found_message(&error.message) {
                anyllm::Error::ModelNotFound(error.message.clone())
            } else if error.message.contains("content filtering") {
                anyllm::Error::ContentFiltered(error.message.clone())
            } else {
                anyllm::Error::InvalidRequest(error.message.clone())
            }
        }
        "api_error" | "server_error" => anyllm::Error::Provider {
            status: None,
            message: error.message.clone(),
            body: None,
            request_id: None,
        },
        _ => anyllm::Error::Stream(format!("{}: {}", error.error_type, error.message)),
    }
}

/// Map an HTTP error response to an anyllm::Error.
pub(crate) fn map_http_error(
    status: u16,
    body: &str,
    request_id: Option<String>,
    retry_after: Option<std::time::Duration>,
) -> anyllm::Error {
    // Try to parse structured error
    let api_error = serde_json::from_str::<wire::ErrorResponse>(body).ok();
    let error_message = api_error
        .as_ref()
        .map(|e| e.error.message.clone())
        .unwrap_or_else(|| body.to_string());

    if is_model_not_found_message(&error_message) {
        return anyllm::Error::ModelNotFound(error_message);
    }

    // Check for context length exceeded (400 + "prompt is too long")
    if status == 400
        && let Some(ref err) = api_error
    {
        if err.error.message.contains("prompt is too long")
            || err.error.message.contains("too many tokens")
        {
            return anyllm::Error::ContextLengthExceeded {
                message: error_message,
                max_tokens: None,
            };
        }
        if err.error.error_type == "invalid_request_error"
            && err.error.message.contains("content filtering")
        {
            return anyllm::Error::ContentFiltered(error_message);
        }
    }

    match status {
        401 | 403 => anyllm::Error::Auth(error_message),
        429 => anyllm::Error::RateLimited {
            message: error_message,
            retry_after,
            request_id,
        },
        503 | 529 => anyllm::Error::Overloaded {
            message: error_message,
            retry_after,
            request_id,
        },
        400..=499 => anyllm::Error::InvalidRequest(error_message),
        500..=599 => anyllm::Error::Provider {
            status: Some(status),
            message: error_message,
            body: Some(body.to_string()),
            request_id,
        },
        _ => anyllm::Error::Provider {
            status: Some(status),
            message: error_message,
            body: Some(body.to_string()),
            request_id,
        },
    }
}

fn is_model_not_found_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("model") && (lower.contains("not found") || lower.contains("does not exist"))
}

#[cfg(test)]
pub(crate) fn conformance_map_http_error(
    status: u16,
    body: &str,
    request_id: Option<String>,
    retry_after: Option<std::time::Duration>,
) -> anyllm::Error {
    map_http_error(status, body, request_id, retry_after)
}

#[cfg(test)]
pub(crate) fn conformance_map_timeout_transport_error() -> anyllm::Error {
    map_transport_error(std::io::Error::new(
        ErrorKind::TimedOut,
        "request timed out",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct WrappedIoError(std::io::Error);

    impl std::fmt::Display for WrappedIoError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "wrapped: {}", self.0)
        }
    }

    impl StdError for WrappedIoError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            Some(&self.0)
        }
    }

    fn structured_error_body(error_type: &str, message: &str) -> String {
        serde_json::json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })
        .to_string()
    }

    #[test]
    fn maps_401_to_auth() {
        let body = structured_error_body("authentication_error", "invalid x-api-key");
        let err = map_http_error(401, &body, None, None);
        match err {
            anyllm::Error::Auth(msg) => assert_eq!(msg, "invalid x-api-key"),
            other => panic!("Expected Auth, got {other:?}"),
        }
    }

    #[test]
    fn maps_403_to_auth() {
        let body = structured_error_body("permission_error", "forbidden");
        let err = map_http_error(403, &body, None, None);
        match err {
            anyllm::Error::Auth(msg) => assert_eq!(msg, "forbidden"),
            other => panic!("Expected Auth, got {other:?}"),
        }
    }

    #[test]
    fn maps_429_to_rate_limited() {
        let body = structured_error_body("rate_limit_error", "too many requests");
        let err = map_http_error(429, &body, Some("req-123".to_string()), None);
        match err {
            anyllm::Error::RateLimited {
                message,
                retry_after,
                request_id,
            } => {
                assert_eq!(message, "too many requests");
                assert!(retry_after.is_none());
                assert_eq!(request_id, Some("req-123".to_string()));
            }
            other => panic!("Expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn maps_503_to_overloaded() {
        let body = structured_error_body("overloaded_error", "service unavailable");
        let err = map_http_error(503, &body, Some("req-456".to_string()), None);
        match err {
            anyllm::Error::Overloaded {
                message,
                retry_after,
                request_id,
            } => {
                assert_eq!(message, "service unavailable");
                assert!(retry_after.is_none());
                assert_eq!(request_id, Some("req-456".to_string()));
            }
            other => panic!("Expected Overloaded, got {other:?}"),
        }
    }

    #[test]
    fn maps_529_to_overloaded() {
        let body = structured_error_body("overloaded_error", "Overloaded");
        let err = map_http_error(529, &body, None, None);
        match err {
            anyllm::Error::Overloaded {
                message,
                retry_after,
                request_id,
            } => {
                assert_eq!(message, "Overloaded");
                assert!(retry_after.is_none());
                assert!(request_id.is_none());
            }
            other => panic!("Expected Overloaded, got {other:?}"),
        }
    }

    #[test]
    fn maps_400_prompt_too_long_to_context_length_exceeded() {
        let body =
            structured_error_body("invalid_request_error", "prompt is too long: 200000 tokens");
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::ContextLengthExceeded {
                message,
                max_tokens,
            } => {
                assert!(message.contains("prompt is too long"));
                assert!(max_tokens.is_none());
            }
            other => panic!("Expected ContextLengthExceeded, got {other:?}"),
        }
    }

    #[test]
    fn maps_400_too_many_tokens_to_context_length_exceeded() {
        let body = structured_error_body(
            "invalid_request_error",
            "too many tokens: your request had 200000",
        );
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::ContextLengthExceeded { message, .. } => {
                assert!(message.contains("too many tokens"));
            }
            other => panic!("Expected ContextLengthExceeded, got {other:?}"),
        }
    }

    #[test]
    fn maps_400_content_filtering_to_content_filtered() {
        let body = structured_error_body(
            "invalid_request_error",
            "Your request was blocked due to content filtering",
        );
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::ContentFiltered(msg) => {
                assert!(msg.contains("content filtering"));
            }
            other => panic!("Expected ContentFiltered, got {other:?}"),
        }
    }

    #[test]
    fn maps_400_other_to_invalid_request() {
        let body = structured_error_body(
            "invalid_request_error",
            "max_tokens must be a positive integer",
        );
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::InvalidRequest(msg) => {
                assert_eq!(msg, "max_tokens must be a positive integer");
            }
            other => panic!("Expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn maps_500_to_provider() {
        let body = structured_error_body("api_error", "internal server error");
        let err = map_http_error(500, &body, Some("req-789".to_string()), None);
        match err {
            anyllm::Error::Provider {
                status,
                message,
                body: err_body,
                request_id,
            } => {
                assert_eq!(status, Some(500));
                assert_eq!(message, "internal server error");
                assert!(err_body.is_some());
                assert_eq!(request_id, Some("req-789".to_string()));
            }
            other => panic!("Expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn maps_structured_model_not_found_message() {
        let body = structured_error_body("invalid_request_error", "model not found");
        let err = map_http_error(404, &body, None, None);
        match err {
            anyllm::Error::ModelNotFound(msg) => {
                // Should use the structured message, not the raw body
                assert_eq!(msg, "model not found");
            }
            other => panic!("Expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn falls_back_to_raw_body_when_not_structured() {
        let body = "Service Unavailable";
        let err = map_http_error(503, body, None, None);
        match err {
            anyllm::Error::Overloaded { message, .. } => {
                assert_eq!(message, "Service Unavailable");
            }
            other => panic!("Expected Overloaded, got {other:?}"),
        }
    }

    #[test]
    fn maps_unknown_status_to_provider() {
        let body = "something weird";
        let err = map_http_error(600, body, None, None);
        match err {
            anyllm::Error::Provider {
                status,
                message,
                body: err_body,
                ..
            } => {
                assert_eq!(status, Some(600));
                assert_eq!(message, "something weird");
                assert_eq!(err_body, Some("something weird".to_string()));
            }
            other => panic!("Expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn maps_502_to_provider() {
        let body = structured_error_body("api_error", "bad gateway");
        let err = map_http_error(502, &body, None, None);
        match err {
            anyllm::Error::Provider {
                status, message, ..
            } => {
                assert_eq!(status, Some(502));
                assert_eq!(message, "bad gateway");
            }
            other => panic!("Expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn maps_404_to_invalid_request() {
        let body = "Not Found";
        let err = map_http_error(404, body, None, None);
        match err {
            anyllm::Error::InvalidRequest(msg) => {
                assert_eq!(msg, "Not Found");
            }
            other => panic!("Expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn api_error_overloaded_maps_to_overloaded() {
        let err = map_api_error(&wire::Error {
            error_type: "overloaded_error".into(),
            message: "Overloaded".into(),
        });
        assert!(
            matches!(err, anyllm::Error::Overloaded { .. }),
            "expected Overloaded, got {err:?}"
        );
    }

    #[test]
    fn api_error_rate_limit_maps_to_rate_limited() {
        let err = map_api_error(&wire::Error {
            error_type: "rate_limit_error".into(),
            message: "slow down".into(),
        });
        assert!(
            matches!(err, anyllm::Error::RateLimited { .. }),
            "expected RateLimited, got {err:?}"
        );
    }

    #[test]
    fn api_error_auth_maps_to_auth() {
        let err = map_api_error(&wire::Error {
            error_type: "authentication_error".into(),
            message: "bad key".into(),
        });
        assert!(
            matches!(err, anyllm::Error::Auth(_)),
            "expected Auth, got {err:?}"
        );
    }

    #[test]
    fn api_error_permission_maps_to_auth() {
        let err = map_api_error(&wire::Error {
            error_type: "permission_error".into(),
            message: "forbidden".into(),
        });
        assert!(
            matches!(err, anyllm::Error::Auth(_)),
            "expected Auth, got {err:?}"
        );
    }

    #[test]
    fn api_error_invalid_request_maps_to_invalid_request() {
        let err = map_api_error(&wire::Error {
            error_type: "invalid_request_error".into(),
            message: "max_tokens must be positive".into(),
        });
        assert!(
            matches!(err, anyllm::Error::InvalidRequest(_)),
            "expected InvalidRequest, got {err:?}"
        );
    }

    #[test]
    fn api_error_context_length_maps_to_context_length_exceeded() {
        let err = map_api_error(&wire::Error {
            error_type: "invalid_request_error".into(),
            message: "prompt is too long: 300000 tokens".into(),
        });
        assert!(
            matches!(err, anyllm::Error::ContextLengthExceeded { .. }),
            "expected ContextLengthExceeded, got {err:?}"
        );
    }

    #[test]
    fn api_error_server_error_maps_to_provider() {
        let err = map_api_error(&wire::Error {
            error_type: "api_error".into(),
            message: "internal failure".into(),
        });
        assert!(
            matches!(err, anyllm::Error::Provider { .. }),
            "expected Provider, got {err:?}"
        );
    }

    #[test]
    fn api_error_unknown_type_falls_back_to_stream() {
        let err = map_api_error(&wire::Error {
            error_type: "totally_new_error".into(),
            message: "something new".into(),
        });
        assert!(
            matches!(err, anyllm::Error::Stream(_)),
            "expected Stream fallback for unknown type, got {err:?}"
        );
    }

    #[test]
    fn transport_timeout_maps_to_timeout() {
        let err = WrappedIoError(std::io::Error::new(
            ErrorKind::TimedOut,
            "request timed out",
        ));
        assert!(matches!(
            map_transport_error(err),
            anyllm::Error::Timeout(_)
        ));
    }

    #[test]
    fn deserialize_timeout_maps_to_timeout() {
        let err = WrappedIoError(std::io::Error::new(ErrorKind::TimedOut, "body timed out"));
        assert!(matches!(
            map_response_deserialize_error(err),
            anyllm::Error::Timeout(_)
        ));
    }

    #[test]
    fn stream_timeout_maps_to_timeout() {
        let err = WrappedIoError(std::io::Error::new(ErrorKind::TimedOut, "stream timed out"));
        assert!(matches!(map_stream_error(err), anyllm::Error::Timeout(_)));
    }
}
