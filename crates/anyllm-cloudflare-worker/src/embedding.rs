//! `EmbeddingProvider` implementation for Cloudflare Workers AI.
//!
//! Wraps the `worker::Ai::run()` binding for text-embedding models like
//! `@cf/baai/bge-base-en-v1.5`. Wire types mirror the native Workers AI
//! embedding schema: `{ "text": [...] }` in, `{ "shape": [N, D], "data": [...] }`
//! out.

use anyllm::{
    CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    Error, Result,
};
use serde::{Deserialize, Serialize};

use crate::Provider;
use crate::error::map_worker_error;

/// Workers AI embedding request body.
#[derive(Debug, Serialize)]
pub(crate) struct EmbedRequest {
    pub text: Vec<String>,
}

/// Workers AI embedding response body.
///
/// The native API returns `{ "shape": [batch, dimensions], "data": [...] }`.
/// `shape` is ignored — the portable response inspects `data` directly.
#[derive(Debug, Deserialize)]
pub(crate) struct EmbedResponse {
    pub data: Vec<Vec<f32>>,
}

impl TryFrom<&EmbeddingRequest> for EmbedRequest {
    type Error = Error;

    fn try_from(request: &EmbeddingRequest) -> Result<Self> {
        if request.dimensions.is_some() {
            return Err(Error::Unsupported(
                "cloudflare-worker embedding does not support output dimension selection".into(),
            ));
        }
        if request.inputs.is_empty() {
            return Err(Error::InvalidRequest(
                "embedding request has no inputs".into(),
            ));
        }
        Ok(Self {
            text: request.inputs.clone(),
        })
    }
}

impl From<EmbedResponse> for EmbeddingResponse {
    fn from(response: EmbedResponse) -> Self {
        EmbeddingResponse::new(response.data)
    }
}

impl EmbeddingProvider for Provider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let cf_request = EmbedRequest::try_from(request)?;

        let response: EmbedResponse = self
            .ai
            .run(&request.model, &cf_request)
            .await
            .map_err(map_worker_error)?;

        Ok(response.into())
    }

    fn embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        match capability {
            EmbeddingCapability::BatchInput => CapabilitySupport::Supported,
            EmbeddingCapability::OutputDimensions => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_conversion_forwards_inputs() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input("hello")
            .input("world");

        let cf = EmbedRequest::try_from(&request).unwrap();
        assert_eq!(cf.text, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn request_rejects_dimension_override() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input("hi")
            .dimensions(256);
        let err = EmbedRequest::try_from(&request).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn request_rejects_empty_inputs() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5");
        let err = EmbedRequest::try_from(&request).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn response_conversion_preserves_vectors() {
        let wire = EmbedResponse {
            data: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        };
        let response: EmbeddingResponse = wire.into();
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.embeddings[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(response.embeddings[1], vec![0.4, 0.5, 0.6]);
    }
}
