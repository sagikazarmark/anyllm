//! Error mapping from `worker::Error` to `anyllm::Error`.

/// Convert a `worker::Error` to an `anyllm::Error`.
pub(crate) fn map_worker_error(err: worker::Error) -> anyllm::Error {
    let message = err.to_string();

    // workers-rs doesn't expose HTTP status codes through its Error type,
    // so we map based on error message patterns.
    let lower = message.to_ascii_lowercase();

    if lower.contains("auth") || lower.contains("unauthorized") || lower.contains("forbidden") {
        anyllm::Error::Auth(message)
    } else if lower.contains("timeout") || lower.contains("timed out") {
        anyllm::Error::Timeout(message)
    } else if lower.contains("rate limit") || lower.contains("too many requests") {
        anyllm::Error::RateLimited {
            message,
            retry_after: None,
            request_id: None,
        }
    } else if lower.contains("overloaded") || lower.contains("unavailable") {
        anyllm::Error::Overloaded {
            message,
            retry_after: None,
            request_id: None,
        }
    } else if lower.contains("model not found")
        || lower.contains("unknown model")
        || lower.contains("no such model")
    {
        anyllm::Error::ModelNotFound(message)
    } else if lower.contains("context length")
        || lower.contains("maximum context length")
        || lower.contains("prompt is too long")
        || lower.contains("too many tokens")
    {
        anyllm::Error::ContextLengthExceeded {
            message,
            max_tokens: None,
        }
    } else if lower.contains("content filtered")
        || lower.contains("unsafe content")
        || lower.contains("safety")
    {
        anyllm::Error::ContentFiltered(message)
    } else {
        anyllm::Error::Provider {
            status: None,
            message,
            body: None,
            request_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_error(message: &str) -> worker::Error {
        worker::Error::RustError(message.to_string())
    }

    #[test]
    fn maps_auth_errors() {
        let err = map_worker_error(provider_error("Unauthorized request"));
        assert!(matches!(err, anyllm::Error::Auth(message) if message.contains("Unauthorized")));
    }

    #[test]
    fn maps_rate_limit_errors() {
        let err = map_worker_error(provider_error("Rate limit exceeded"));
        assert!(
            matches!(err, anyllm::Error::RateLimited { message, .. } if message.contains("Rate limit"))
        );
    }

    #[test]
    fn maps_timeout_errors() {
        let err = map_worker_error(provider_error("Worker request timed out"));
        assert!(matches!(err, anyllm::Error::Timeout(message) if message.contains("timed out")));
    }

    #[test]
    fn maps_overloaded_errors() {
        let err = map_worker_error(provider_error("Service unavailable"));
        assert!(
            matches!(err, anyllm::Error::Overloaded { message, .. } if message.contains("unavailable"))
        );
    }

    #[test]
    fn maps_model_not_found_errors() {
        let err = map_worker_error(provider_error("Model not found: @cf/meta/missing"));
        assert!(
            matches!(err, anyllm::Error::ModelNotFound(message) if message.contains("Model not found"))
        );
    }

    #[test]
    fn maps_context_length_errors() {
        let err = map_worker_error(provider_error(
            "This model's maximum context length is 32768 tokens",
        ));
        assert!(matches!(
            err,
            anyllm::Error::ContextLengthExceeded { message, max_tokens: None }
            if message.contains("maximum context length")
        ));
    }

    #[test]
    fn maps_content_filtered_errors() {
        let err = map_worker_error(provider_error("Safety system blocked this response"));
        assert!(matches!(
            err,
            anyllm::Error::ContentFiltered(message) if message.contains("Safety system")
        ));
    }

    #[test]
    fn falls_back_to_provider_error() {
        let err = map_worker_error(provider_error("unexpected worker failure"));
        assert!(
            matches!(err, anyllm::Error::Provider { message, .. } if message.contains("unexpected worker failure"))
        );
    }
}
