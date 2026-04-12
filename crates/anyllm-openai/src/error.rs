pub(crate) use anyllm_openai_compat::{
    map_http_error, map_response_deserialize_error, map_stream_error, map_transport_error,
};

#[cfg(test)]
use std::error::Error as StdError;
#[cfg(test)]
use std::io::ErrorKind;

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

    #[derive(Debug)]
    struct OtherError;

    impl std::fmt::Display for OtherError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "boom")
        }
    }

    impl StdError for OtherError {}

    fn structured_error_body(error_type: &str, message: &str, code: Option<&str>) -> String {
        let mut error = serde_json::json!({
            "error": {
                "type": error_type,
                "message": message,
            }
        });
        if let Some(c) = code {
            error["error"]["code"] = serde_json::json!(c);
        }
        error.to_string()
    }

    #[test]
    fn maps_401_to_auth() {
        let body = structured_error_body(
            "invalid_request_error",
            "Incorrect API key provided",
            Some("invalid_api_key"),
        );
        let err = map_http_error(401, &body, None, None);
        match err {
            anyllm::Error::Auth(msg) => assert_eq!(msg, "Incorrect API key provided"),
            other => panic!("Expected Auth, got {other:?}"),
        }
    }

    #[test]
    fn maps_403_to_auth() {
        let body = structured_error_body("insufficient_quota", "Quota exceeded", None);
        let err = map_http_error(403, &body, None, None);
        match err {
            anyllm::Error::Auth(msg) => assert_eq!(msg, "Quota exceeded"),
            other => panic!("Expected Auth, got {other:?}"),
        }
    }

    #[test]
    fn maps_429_to_rate_limited() {
        let body = structured_error_body("rate_limit_error", "Rate limit reached", None);
        let retry_after = Some(std::time::Duration::from_secs(30));
        let err = map_http_error(429, &body, Some("req-123".into()), retry_after);
        match err {
            anyllm::Error::RateLimited {
                message,
                retry_after: ra,
                request_id,
            } => {
                assert_eq!(message, "Rate limit reached");
                assert_eq!(ra, Some(std::time::Duration::from_secs(30)));
                assert_eq!(request_id, Some("req-123".to_string()));
            }
            other => panic!("Expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn maps_503_to_overloaded() {
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
    fn maps_context_length_code_to_context_length_exceeded() {
        let body = structured_error_body(
            "invalid_request_error",
            "This model's maximum context length is 128000 tokens",
            Some("context_length_exceeded"),
        );
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::ContextLengthExceeded { message, .. } => {
                assert!(message.contains("maximum context length"));
            }
            other => panic!("Expected ContextLengthExceeded, got {other:?}"),
        }
    }

    #[test]
    fn maps_context_length_message_to_context_length_exceeded() {
        let body = structured_error_body(
            "invalid_request_error",
            "This request exceeds the maximum context length of 128000",
            None,
        );
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::ContextLengthExceeded { message, .. } => {
                assert!(message.contains("maximum context length"));
            }
            other => panic!("Expected ContextLengthExceeded, got {other:?}"),
        }
    }

    #[test]
    fn maps_content_filter_code_to_content_filtered() {
        let body = structured_error_body(
            "invalid_request_error",
            "Your request was rejected by the content filter",
            Some("content_filter"),
        );
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::ContentFiltered(msg) => {
                assert!(msg.contains("content filter"));
            }
            other => panic!("Expected ContentFiltered, got {other:?}"),
        }
    }

    #[test]
    fn maps_400_model_not_found_code_to_model_not_found() {
        let body = structured_error_body(
            "invalid_request_error",
            "model does not exist",
            Some("model_not_found"),
        );
        let err = map_http_error(400, &body, None, None);
        match err {
            anyllm::Error::ModelNotFound(msg) => {
                assert_eq!(msg, "model does not exist");
            }
            other => panic!("Expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn maps_500_to_provider() {
        let body = structured_error_body("server_error", "Internal server error", None);
        let err = map_http_error(500, &body, Some("req-789".into()), None);
        match err {
            anyllm::Error::Provider {
                status,
                message,
                request_id,
                ..
            } => {
                assert_eq!(status, Some(500));
                assert_eq!(message, "Internal server error");
                assert_eq!(request_id, Some("req-789".to_string()));
            }
            other => panic!("Expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn falls_back_to_raw_body_when_not_structured() {
        let body = "Bad Gateway";
        let err = map_http_error(502, body, None, None);
        match err {
            anyllm::Error::Provider { message, .. } => {
                assert_eq!(message, "Bad Gateway");
            }
            other => panic!("Expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn maps_404_model_not_found_to_model_not_found() {
        let body = structured_error_body(
            "invalid_request_error",
            "That model does not exist",
            Some("model_not_found"),
        );
        let err = map_http_error(404, &body, None, None);
        match err {
            anyllm::Error::ModelNotFound(msg) => {
                assert_eq!(msg, "That model does not exist");
            }
            other => panic!("Expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn transport_timeout_maps_to_timeout() {
        let err = WrappedIoError(std::io::Error::new(
            ErrorKind::TimedOut,
            "network timed out",
        ));
        match map_transport_error(err) {
            anyllm::Error::Timeout(msg) => assert!(msg.contains("timed out")),
            other => panic!("Expected Timeout, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_timeout_maps_to_timeout() {
        let err = WrappedIoError(std::io::Error::new(ErrorKind::TimedOut, "body timed out"));
        match map_response_deserialize_error(err) {
            anyllm::Error::Timeout(msg) => assert!(msg.contains("timed out")),
            other => panic!("Expected Timeout, got {other:?}"),
        }
    }

    #[test]
    fn non_timeout_transport_maps_to_provider() {
        assert!(matches!(
            map_transport_error(OtherError),
            anyllm::Error::Provider { status: None, .. }
        ));
    }

    #[test]
    fn stream_timeout_maps_to_timeout() {
        let err = WrappedIoError(std::io::Error::new(ErrorKind::TimedOut, "stream timed out"));
        match map_stream_error(err) {
            anyllm::Error::Timeout(msg) => assert!(msg.contains("timed out")),
            other => panic!("Expected Timeout, got {other:?}"),
        }
    }
}
