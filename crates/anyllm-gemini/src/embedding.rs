use anyllm::{
    CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    Error, Result,
};
use serde::{Deserialize, Serialize};

use crate::EmbeddingRequestOptions;
use crate::Provider;
use crate::error::{map_http_error, map_response_deserialize_error, map_transport_error};

#[derive(Debug, Serialize)]
pub(crate) struct BatchEmbedContentsRequest {
    pub requests: Vec<EmbedContentRequest>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EmbedContentRequest {
    pub model: String,
    pub content: EmbedContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbedContent {
    pub parts: Vec<EmbedPart>,
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbedPart {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BatchEmbedContentsResponse {
    pub embeddings: Vec<EmbeddingValues>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingValues {
    pub values: Vec<f32>,
}

fn to_api_request(
    request: &EmbeddingRequest,
    options: &EmbeddingRequestOptions,
) -> Result<BatchEmbedContentsRequest> {
    if request.inputs.is_empty() {
        return Err(Error::InvalidRequest(
            "embedding request must have at least one input".into(),
        ));
    }
    let model_path = if request.model.starts_with("models/") {
        request.model.clone()
    } else {
        format!("models/{}", request.model)
    };
    let requests = request
        .inputs
        .iter()
        .map(|input| EmbedContentRequest {
            model: model_path.clone(),
            content: EmbedContent {
                parts: vec![EmbedPart {
                    text: input.clone(),
                }],
            },
            task_type: options.task_type.clone(),
            title: options.title.clone(),
            output_dimensionality: request.dimensions,
        })
        .collect();
    Ok(BatchEmbedContentsRequest { requests })
}

pub(crate) fn from_api_response(
    response: BatchEmbedContentsResponse,
    model: &str,
) -> EmbeddingResponse {
    let embeddings: Vec<Vec<f32>> = response.embeddings.into_iter().map(|e| e.values).collect();
    EmbeddingResponse::new(embeddings).model(model.to_string())
}

impl Provider {
    async fn send_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<reqwest::Response> {
        let options = request
            .option::<EmbeddingRequestOptions>()
            .cloned()
            .unwrap_or_default();
        let api_request = to_api_request(request, &options)?;
        let model_path = api_request
            .requests
            .first()
            .map(|r| r.model.clone())
            .unwrap_or_else(|| format!("models/{}", request.model));

        let url = format!("{}/{}:batchEmbedContents", self.inner.base_url, model_path);
        let body = serde_json::to_string(&api_request).map_err(Error::from)?;

        let mut req = self
            .inner
            .client
            .post(&url)
            .header("x-goog-api-key", &self.inner.api_key)
            .header("content-type", "application/json");

        if let Some(timeout) = crate::request_timeout(false, self.inner.request_timeout) {
            req = req.timeout(timeout);
        }

        let response = req.body(body).send().await.map_err(map_transport_error)?;
        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|seconds| seconds.is_finite() && !seconds.is_sign_negative())
                .map(std::time::Duration::from_secs_f64);
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            return Err(map_http_error(status.as_u16(), &error_body, retry_after));
        }
        Ok(response)
    }
}

impl EmbeddingProvider for Provider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let response = self.send_embedding_request(request).await?;
        let api_response: BatchEmbedContentsResponse = response
            .json()
            .await
            .map_err(map_response_deserialize_error)?;
        // Gemini does not return usage on embedding responses.
        let converted = from_api_response(api_response, &request.model);
        Ok(converted)
    }

    fn embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        match capability {
            EmbeddingCapability::BatchInput | EmbeddingCapability::OutputDimensions => {
                CapabilitySupport::Supported
            }
            _ => CapabilitySupport::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_api_request_prefixes_model_when_missing() {
        let request = EmbeddingRequest::new("text-embedding-004")
            .inputs(["a", "b"])
            .dimensions(256);
        let options = EmbeddingRequestOptions {
            task_type: Some("RETRIEVAL_DOCUMENT".into()),
            title: None,
        };
        let api = to_api_request(&request, &options).unwrap();
        let json = serde_json::to_value(&api).unwrap();
        let requests = json["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["model"], "models/text-embedding-004");
        assert_eq!(requests[0]["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(requests[0]["outputDimensionality"], 256);
        assert_eq!(requests[0]["content"]["parts"][0]["text"], "a");
    }

    #[test]
    fn to_api_request_preserves_explicit_models_prefix() {
        let request = EmbeddingRequest::new("models/text-embedding-004").input("a");
        let api = to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap();
        assert_eq!(api.requests[0].model, "models/text-embedding-004");
    }

    #[test]
    fn to_api_request_rejects_empty_inputs() {
        let request = EmbeddingRequest::new("text-embedding-004");
        let err = to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn from_api_response_preserves_order_and_model() {
        let wire = BatchEmbedContentsResponse {
            embeddings: vec![
                EmbeddingValues {
                    values: vec![1.0, 2.0],
                },
                EmbeddingValues {
                    values: vec![3.0, 4.0],
                },
            ],
        };
        let converted = from_api_response(wire, "text-embedding-004");
        assert_eq!(converted.embeddings, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(converted.model.as_deref(), Some("text-embedding-004"));
    }
}
