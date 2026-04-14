use std::error::Error as StdError;
use std::io::ErrorKind;

use crate::wire::GeminiErrorResponse;

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

/// Map an HTTP error response to an `anyllm::Error`.
///
/// Uses both the HTTP status code and the structured Gemini error body
/// (when parseable) to produce the most specific error variant.
pub(crate) fn map_http_error(
    status: u16,
    body: &str,
    retry_after: Option<std::time::Duration>,
) -> anyllm::Error {
    let api_error = serde_json::from_str::<GeminiErrorResponse>(body).ok();
    let error_message = api_error
        .as_ref()
        .map(|e| e.error.message.clone())
        .unwrap_or_else(|| body.to_string());
    let error_status = api_error.as_ref().and_then(|e| e.error.status.clone());
    let lower = error_message.to_ascii_lowercase();

    // Check Gemini status string for safety/content filter.
    if let Some(ref gs) = error_status
        && gs.as_str() == "PERMISSION_DENIED"
    {
        return anyllm::Error::Auth(error_message);
    }

    if lower.contains("model") && (lower.contains("not found") || lower.contains("does not exist"))
    {
        return anyllm::Error::ModelNotFound(error_message);
    }

    // Check message text for specific conditions.
    if status == 400 {
        if lower.contains("context") && (lower.contains("token") || lower.contains("length")) {
            return anyllm::Error::ContextLengthExceeded {
                message: error_message,
                max_tokens: None,
            };
        }
        if lower.contains("safety") || lower.contains("content filter") {
            return anyllm::Error::ContentFiltered(error_message);
        }
    }

    match status {
        401 | 403 => anyllm::Error::Auth(error_message),
        429 => anyllm::Error::RateLimited {
            message: error_message,
            retry_after,
            request_id: None,
        },
        503 => anyllm::Error::Overloaded {
            message: error_message,
            retry_after,
            request_id: None,
        },
        400..=499 => anyllm::Error::InvalidRequest(error_message),
        500..=599 => anyllm::Error::Provider {
            status: Some(status),
            message: error_message,
            body: Some(body.to_string()),
            request_id: None,
        },
        _ => anyllm::Error::Provider {
            status: Some(status),
            message: error_message,
            body: Some(body.to_string()),
            request_id: None,
        },
    }
}

#[cfg(test)]
pub(crate) fn conformance_map_http_error(
    status: u16,
    body: &str,
    retry_after: Option<std::time::Duration>,
) -> anyllm::Error {
    map_http_error(status, body, retry_after)
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

    fn gemini_error_body(code: u16, message: &str, status: &str) -> String {
        serde_json::json!({
            "error": {
                "code": code,
                "message": message,
                "status": status
            }
        })
        .to_string()
    }

    #[test]
    fn maps_401_to_auth() {
        let body = gemini_error_body(401, "API key not valid", "UNAUTHENTICATED");
        let err = map_http_error(401, &body, None);
        match err {
            anyllm::Error::Auth(msg) => assert_eq!(msg, "API key not valid"),
            other => panic!("Expected Auth, got {other:?}"),
        }
    }

    #[test]
    fn maps_403_to_auth() {
        let body = gemini_error_body(403, "Permission denied", "PERMISSION_DENIED");
        let err = map_http_error(403, &body, None);
        match err {
            anyllm::Error::Auth(msg) => assert_eq!(msg, "Permission denied"),
            other => panic!("Expected Auth, got {other:?}"),
        }
    }

    #[test]
    fn maps_permission_denied_status_to_auth() {
        // Even on non-403 status, PERMISSION_DENIED → Auth
        let body = gemini_error_body(400, "Caller does not have permission", "PERMISSION_DENIED");
        let err = map_http_error(400, &body, None);
        match err {
            anyllm::Error::Auth(msg) => assert!(msg.contains("permission")),
            other => panic!("Expected Auth, got {other:?}"),
        }
    }

    #[test]
    fn maps_429_to_rate_limited() {
        let body = gemini_error_body(429, "Quota exceeded", "RESOURCE_EXHAUSTED");
        let retry_after = Some(std::time::Duration::from_secs(30));
        let err = map_http_error(429, &body, retry_after);
        match err {
            anyllm::Error::RateLimited {
                message,
                retry_after: ra,
                ..
            } => {
                assert_eq!(message, "Quota exceeded");
                assert_eq!(ra, Some(std::time::Duration::from_secs(30)));
            }
            other => panic!("Expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn maps_503_to_overloaded() {
        let body = "Service Unavailable";
        let err = map_http_error(503, body, None);
        match err {
            anyllm::Error::Overloaded { message, .. } => {
                assert_eq!(message, "Service Unavailable");
            }
            other => panic!("Expected Overloaded, got {other:?}"),
        }
    }

    #[test]
    fn maps_context_length_message_to_context_length_exceeded() {
        let body = gemini_error_body(
            400,
            "This model's maximum context length is exceeded",
            "INVALID_ARGUMENT",
        );
        let err = map_http_error(400, &body, None);
        match err {
            anyllm::Error::ContextLengthExceeded { .. } => {}
            other => panic!("Expected ContextLengthExceeded, got {other:?}"),
        }
    }

    #[test]
    fn maps_safety_message_to_content_filtered() {
        let body = gemini_error_body(
            400,
            "Response was blocked due to safety settings",
            "INVALID_ARGUMENT",
        );
        let err = map_http_error(400, &body, None);
        match err {
            anyllm::Error::ContentFiltered(_) => {}
            other => panic!("Expected ContentFiltered, got {other:?}"),
        }
    }

    #[test]
    fn maps_400_model_missing_message_to_model_not_found() {
        let body = gemini_error_body(400, "model does not exist", "INVALID_ARGUMENT");
        let err = map_http_error(400, &body, None);
        match err {
            anyllm::Error::ModelNotFound(msg) => {
                assert_eq!(msg, "model does not exist");
            }
            other => panic!("Expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn maps_500_to_provider() {
        let body = gemini_error_body(500, "Internal server error", "INTERNAL");
        let err = map_http_error(500, &body, None);
        match err {
            anyllm::Error::Provider {
                status, message, ..
            } => {
                assert_eq!(status, Some(500));
                assert_eq!(message, "Internal server error");
            }
            other => panic!("Expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn falls_back_to_raw_body_when_not_structured() {
        let body = "Bad Gateway";
        let err = map_http_error(502, body, None);
        match err {
            anyllm::Error::Provider { message, .. } => {
                assert_eq!(message, "Bad Gateway");
            }
            other => panic!("Expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn maps_404_to_model_not_found() {
        let body = gemini_error_body(404, "Model not found", "NOT_FOUND");
        let err = map_http_error(404, &body, None);
        match err {
            anyllm::Error::ModelNotFound(msg) => {
                assert_eq!(msg, "Model not found");
            }
            other => panic!("Expected ModelNotFound, got {other:?}"),
        }
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
