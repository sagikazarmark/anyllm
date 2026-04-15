use anyllm::{
    EmbeddingRequest, EmbeddingResponse, Error, ExtraMap, ResponseMetadata, Result, Usage,
};
use serde::{Deserialize, Serialize};

/// OpenAI-style embeddings request wire shape.
///
/// Used by any OpenAI-compatible `/v1/embeddings` endpoint.
#[derive(Debug, Serialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(flatten, skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

/// Provider-specific extras applied to the OpenAI-style embedding request.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingRequestOptions {
    /// Optional `user` field forwarded for abuse-prevention tracking.
    pub user: Option<String>,
    /// Additional serializable JSON fields merged into the request body.
    pub extra: ExtraMap,
}

/// OpenAI-style embeddings response wire shape.
#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    #[serde(default)]
    pub usage: Option<EmbeddingsUsage>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
    pub index: u32,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
}

/// Convert an anyllm `EmbeddingRequest` into the OpenAI-style wire shape.
pub fn to_embeddings_request(
    request: &EmbeddingRequest,
    options: &EmbeddingRequestOptions,
) -> Result<EmbeddingsRequest> {
    if request.inputs.is_empty() {
        return Err(Error::InvalidRequest(
            "embedding request must have at least one input".into(),
        ));
    }
    Ok(EmbeddingsRequest {
        model: request.model.clone(),
        input: request.inputs.clone(),
        dimensions: request.dimensions,
        user: options.user.clone(),
        extra: options.extra.clone(),
    })
}

/// Convert an OpenAI-style embeddings response into the portable
/// `EmbeddingResponse`.
///
/// The caller may attach provider-specific typed metadata via `metadata_hook`.
pub fn from_embeddings_response<M>(
    response: EmbeddingsResponse,
    metadata_hook: M,
) -> Result<EmbeddingResponse>
where
    M: FnOnce(&EmbeddingsResponse, &mut ResponseMetadata),
{
    let mut metadata = ResponseMetadata::new();
    metadata_hook(&response, &mut metadata);

    let EmbeddingsResponse { data, model, usage } = response;

    let mut ordered: Vec<(u32, Vec<f32>)> =
        data.into_iter().map(|d| (d.index, d.embedding)).collect();
    ordered.sort_by_key(|(index, _)| *index);
    let embeddings: Vec<Vec<f32>> = ordered.into_iter().map(|(_, vector)| vector).collect();

    let usage = usage.map(|u| {
        let mut out = Usage::new();
        if let Some(p) = u.prompt_tokens {
            out.input_tokens = Some(p);
        }
        if let Some(t) = u.total_tokens {
            out.total_tokens = Some(t);
        }
        out
    });

    let mut response = EmbeddingResponse::new(embeddings);
    response.model = Some(model);
    response.usage = usage;
    response.metadata = metadata;
    Ok(response)
}

/// Generic HTTP dispatch for sending an OpenAI-style embeddings request body
/// via a caller-supplied `send` future. Mirrors
/// [`crate::send_chat_completion_request`].
pub async fn send_embeddings_request<E, Fut, F, M>(
    api_request: &EmbeddingsRequest,
    send: F,
    map_transport_error: M,
) -> Result<reqwest::Response>
where
    F: FnOnce(String) -> Fut,
    Fut: std::future::Future<Output = std::result::Result<reqwest::Response, E>>,
    M: Fn(E) -> Error,
{
    let body = serde_json::to_string(api_request).map_err(Error::from)?;
    send(body).await.map_err(map_transport_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_wire_request_serializes_inputs_and_dimensions() {
        let request = EmbeddingRequest::new("text-embedding-3-small")
            .inputs(["a", "b"])
            .dimensions(256);
        let options = EmbeddingRequestOptions {
            user: Some("user_1".into()),
            ..Default::default()
        };
        let wire = to_embeddings_request(&request, &options).unwrap();
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "model": "text-embedding-3-small",
                "input": ["a", "b"],
                "dimensions": 256,
                "user": "user_1"
            })
        );
    }

    #[test]
    fn to_wire_request_rejects_empty_inputs() {
        let request = EmbeddingRequest::new("m");
        let err = to_embeddings_request(&request, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn from_wire_response_orders_by_index_and_maps_usage() {
        let response = EmbeddingsResponse {
            data: vec![
                EmbeddingData {
                    embedding: vec![2.0, 2.0],
                    index: 1,
                },
                EmbeddingData {
                    embedding: vec![1.0, 1.0],
                    index: 0,
                },
            ],
            model: "text-embedding-3-small".into(),
            usage: Some(EmbeddingsUsage {
                prompt_tokens: Some(8),
                total_tokens: Some(8),
            }),
        };
        let converted = from_embeddings_response(response, |_, _| {}).unwrap();
        assert_eq!(converted.embeddings, vec![vec![1.0, 1.0], vec![2.0, 2.0]]);
        assert_eq!(converted.model.as_deref(), Some("text-embedding-3-small"));
        let usage = converted.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(8));
        assert_eq!(usage.total_tokens, Some(8));
    }
}
