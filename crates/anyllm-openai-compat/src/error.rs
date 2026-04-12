use std::error::Error as StdError;
use std::io::ErrorKind;

use crate::wire::ApiErrorResponse;

pub fn is_timeout_error(err: &(dyn StdError + 'static)) -> bool {
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

pub fn map_transport_error<E>(err: E) -> anyllm::Error
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

pub fn map_response_deserialize_error<E>(err: E) -> anyllm::Error
where
    E: StdError + Send + Sync + 'static,
{
    if is_timeout_error(&err) {
        anyllm::Error::Timeout(err.to_string())
    } else {
        anyllm::Error::serialization(format!("Failed to deserialize response: {err}"))
    }
}

pub fn map_stream_error<E>(err: E) -> anyllm::Error
where
    E: StdError + Send + Sync + 'static,
{
    if is_timeout_error(&err) {
        anyllm::Error::Timeout(err.to_string())
    } else {
        anyllm::Error::Stream(format!("SSE error: {err}"))
    }
}

pub fn map_http_error(
    status: u16,
    body: &str,
    request_id: Option<String>,
    retry_after: Option<std::time::Duration>,
) -> anyllm::Error {
    let api_error = serde_json::from_str::<ApiErrorResponse>(body).ok();
    let error_message = api_error
        .as_ref()
        .map(|e| e.error.message.clone())
        .unwrap_or_else(|| body.to_string());
    let error_code = api_error.as_ref().and_then(|e| e.error.code.clone());

    if let Some(ref code) = error_code {
        match code.as_str() {
            "context_length_exceeded" => {
                return anyllm::Error::ContextLengthExceeded {
                    message: error_message,
                    max_tokens: None,
                };
            }
            "content_filter" => {
                return anyllm::Error::ContentFiltered(error_message);
            }
            "model_not_found" => {
                return anyllm::Error::ModelNotFound(error_message);
            }
            "invalid_api_key" => {
                return anyllm::Error::Auth(error_message);
            }
            _ => {}
        }
    }

    if status == 400
        && (error_message.contains("maximum context length")
            || error_message.contains("context_length_exceeded"))
    {
        return anyllm::Error::ContextLengthExceeded {
            message: error_message,
            max_tokens: None,
        };
    }

    let lower = error_message.to_ascii_lowercase();
    if status == 404
        && lower.contains("model")
        && (lower.contains("not found") || lower.contains("does not exist"))
    {
        return anyllm::Error::ModelNotFound(error_message);
    }

    match status {
        401 | 403 => anyllm::Error::Auth(error_message),
        429 => anyllm::Error::RateLimited {
            message: error_message,
            retry_after,
            request_id,
        },
        503 => anyllm::Error::Overloaded {
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
