use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::{
    CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    Error, ProviderIdentity, Result,
};

/// Deterministic embedding provider for tests.
///
/// Returns canned responses (or errors) in sequence, records every request,
/// and exposes provider identity plus embedding capability overrides.
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    state: Arc<Mutex<MockEmbeddingState>>,
    embedding_capabilities: HashMap<EmbeddingCapability, CapabilitySupport>,
    provider_name: &'static str,
}

#[derive(Debug)]
struct MockEmbeddingState {
    responses: VecDeque<Result<EmbeddingResponse>>,
    requests: Vec<EmbeddingRequest>,
}

impl MockEmbeddingProvider {
    /// Create a mock embedding provider with no queued responses.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(std::iter::empty::<Result<EmbeddingResponse>>())
    }

    /// Create a mock embedding provider from an ordered queue of responses.
    #[must_use]
    pub fn new<I>(responses: I) -> Self
    where
        I: IntoIterator<Item = Result<EmbeddingResponse>>,
    {
        Self {
            state: Arc::new(Mutex::new(MockEmbeddingState {
                responses: responses.into_iter().collect(),
                requests: Vec::new(),
            })),
            embedding_capabilities: HashMap::new(),
            provider_name: "mock",
        }
    }

    /// Convenience for a single successful response with the given vectors.
    #[must_use]
    pub fn with_vectors(vectors: Vec<Vec<f32>>) -> Self {
        Self::new([Ok(EmbeddingResponse::new(vectors))])
    }

    /// Convenience for a single error response.
    #[must_use]
    pub fn with_error(error: Error) -> Self {
        Self::new([Err(error)])
    }

    /// Queue an additional response without mutating existing queue entries.
    pub fn push_response(&self, response: Result<EmbeddingResponse>) {
        self.state.lock().unwrap().responses.push_back(response);
    }

    /// Override the provider identity name reported by this mock.
    #[must_use]
    pub fn with_provider_name(mut self, name: &'static str) -> Self {
        self.provider_name = name;
        self
    }

    /// Override support for an embedding capability.
    #[must_use]
    pub fn with_embedding_capability(
        mut self,
        capability: EmbeddingCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.embedding_capabilities.insert(capability, support);
        self
    }

    /// Snapshot of every request the mock has received.
    #[must_use]
    pub fn requests(&self) -> Vec<EmbeddingRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Number of queued responses remaining.
    #[must_use]
    pub fn pending_responses(&self) -> usize {
        self.state.lock().unwrap().responses.len()
    }
}

impl ProviderIdentity for MockEmbeddingProvider {
    fn provider_name(&self) -> &'static str {
        self.provider_name
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let mut state = self.state.lock().unwrap();
        state.requests.push(request.clone());
        match state.responses.pop_front() {
            Some(response) => response,
            None => Err(Error::UnexpectedResponse(format!(
                "mock embedding provider '{}' has no queued responses",
                self.provider_name
            ))),
        }
    }

    fn embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        self.embedding_capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }
}

#[cfg(test)]
mod embedding_mock_tests {
    use super::*;
    use crate::{
        CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest,
        EmbeddingResponse, Error, ProviderIdentity,
    };

    #[tokio::test]
    async fn mock_embedding_provider_returns_queued_response() {
        let provider = MockEmbeddingProvider::with_vectors(vec![vec![0.1, 0.2]]);
        let request = EmbeddingRequest::new("mock-embed").input("hello");
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings, vec![vec![0.1, 0.2]]);

        let requests = provider.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].inputs, vec!["hello".to_string()]);
    }

    #[tokio::test]
    async fn mock_embedding_provider_returns_queued_error() {
        let provider = MockEmbeddingProvider::with_error(Error::Auth("bad".into()));
        let err = provider
            .embed(&EmbeddingRequest::new("m").input("x"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Auth(_)));
    }

    #[tokio::test]
    async fn mock_embedding_provider_returns_responses_in_order() {
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![1.0]])),
            Ok(EmbeddingResponse::new(vec![vec![2.0]])),
        ]);
        let first = provider
            .embed(&EmbeddingRequest::new("m").input("a"))
            .await
            .unwrap();
        let second = provider
            .embed(&EmbeddingRequest::new("m").input("b"))
            .await
            .unwrap();
        assert_eq!(first.embeddings, vec![vec![1.0]]);
        assert_eq!(second.embeddings, vec![vec![2.0]]);
    }

    #[tokio::test]
    async fn mock_embedding_provider_reports_exhaustion() {
        let provider = MockEmbeddingProvider::empty();
        let err = provider
            .embed(&EmbeddingRequest::new("m").input("x"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse(_)));
    }

    #[test]
    fn mock_embedding_provider_exposes_provider_identity_and_capabilities() {
        let provider = MockEmbeddingProvider::empty()
            .with_provider_name("demo-embed")
            .with_embedding_capability(
                EmbeddingCapability::BatchInput,
                CapabilitySupport::Supported,
            );
        assert_eq!(provider.provider_name(), "demo-embed");
        assert_eq!(
            provider.embedding_capability("m", EmbeddingCapability::BatchInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.embedding_capability("m", EmbeddingCapability::OutputDimensions),
            CapabilitySupport::Unknown
        );
    }
}
