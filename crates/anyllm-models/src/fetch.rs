use std::collections::HashMap;

use reqwest::header::{HeaderMap, HeaderValue, IF_MODIFIED_SINCE, IF_NONE_MATCH};

use crate::Provider;

/// Options for fetching the registry, including conditional-request validators.
#[derive(Debug, Clone, Default)]
pub struct FetchOptions {
    /// ETag from a previous response, used for conditional requests.
    pub etag: Option<String>,

    /// Last-Modified value from a previous response, used for conditional requests.
    pub last_modified: Option<String>,
}

/// Result of a fetch operation.
#[derive(Debug)]
pub enum FetchResult {
    /// The registry was fetched and deserialized successfully.
    Modified {
        /// The deserialized registry.
        registry: HashMap<String, Provider>,

        /// ETag from the response, if present.
        etag: Option<String>,

        /// Last-Modified from the response, if present.
        last_modified: Option<String>,
    },

    /// The registry has not changed since the conditional-request validators.
    NotModified,
}

const API_URL: &str = "https://models.dev/api.json";

/// Fetch the registry from `models.dev` using the provided HTTP client.
///
/// Pass [`FetchOptions`] with `etag` or `last_modified` from a previous
/// [`FetchResult::Modified`] to make a conditional request. The server may
/// return [`FetchResult::NotModified`] if the data has not changed.
pub async fn fetch(
    client: &reqwest::Client,
    options: &FetchOptions,
) -> Result<FetchResult, Box<dyn std::error::Error + Send + Sync>> {
    let mut headers = HeaderMap::new();
    if let Some(etag) = &options.etag {
        headers.insert(IF_NONE_MATCH, HeaderValue::from_str(etag)?);
    }
    if let Some(last_modified) = &options.last_modified {
        headers.insert(IF_MODIFIED_SINCE, HeaderValue::from_str(last_modified)?);
    }

    let response = client.get(API_URL).headers(headers).send().await?;

    if response.status() == reqwest::StatusCode::NOT_MODIFIED {
        return Ok(FetchResult::NotModified);
    }

    let response = response.error_for_status()?;

    let etag = response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let last_modified = response
        .headers()
        .get("last-modified")
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let registry: HashMap<String, Provider> = response.json().await?;

    Ok(FetchResult::Modified {
        registry,
        etag,
        last_modified,
    })
}
