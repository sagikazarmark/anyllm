#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ChatRequestOptions {
    pub top_k: Option<u32>,
    /// Currently only `1` is supported by `anyllm` because the normalized
    /// response model returns a single candidate.
    pub candidate_count: Option<u32>,
    pub response_mime_type: Option<String>,
    pub cached_content: Option<String>,
    /// Provider-specific thinking budget override (in tokens). When set, takes
    /// precedence over `ChatRequest.reasoning.budget_tokens` and `effort`.
    pub thinking_budget: Option<u32>,
}

/// Per-request options applied to Gemini embedding API calls.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingRequestOptions {
    /// Optional Gemini `taskType` hint (e.g. `RETRIEVAL_QUERY`).
    ///
    /// Forwarded verbatim to the Gemini API. The set of accepted values and
    /// their semantics are provider-specific; `anyllm` does not interpret this
    /// field.
    pub task_type: Option<String>,
    /// Optional `title` associated with the content for retrieval indexing.
    pub title: Option<String>,
}
