use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A model entry from the `models.dev` registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Model identifier (e.g. `"gpt-4o"`, `"claude-sonnet-4-20250514"`).
    pub id: String,

    /// Human-readable model name.
    pub name: String,

    /// Model family (e.g. `"gpt"`, `"claude"`).
    pub family: Option<String>,

    /// Whether the model supports file attachments.
    pub attachment: bool,

    /// Whether the model supports reasoning.
    pub reasoning: bool,

    /// Whether the model supports tool calls.
    pub tool_call: bool,

    /// Whether the model supports temperature configuration.
    pub temperature: Option<bool>,

    /// Whether the model supports structured output.
    pub structured_output: Option<bool>,

    /// Training data knowledge cutoff (e.g. `"2024-10"`).
    pub knowledge: Option<String>,

    /// Release date (e.g. `"2025-08-08"`).
    pub release_date: Option<String>,

    /// Last update date (e.g. `"2025-08-08"`).
    pub last_updated: Option<String>,

    /// Input and output modalities.
    pub modalities: Modalities,

    /// Whether the model weights are openly available.
    pub open_weights: bool,

    /// Pricing information.
    pub cost: Option<Cost>,

    /// Token limit information.
    pub limit: Limit,

    /// Model lifecycle status (e.g. `"beta"`, `"deprecated"`).
    pub status: Option<String>,

    /// Provider-specific override hints (e.g. npm package routing).
    pub provider: Option<Value>,

    /// Interleaved reasoning field configuration (provider-specific).
    pub interleaved: Option<Value>,

    /// Experimental feature flags or mode overrides.
    pub experimental: Option<Value>,
}

/// Input and output modalities for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Modalities {
    /// Supported input modalities (e.g. `["text", "image"]`).
    pub input: Vec<String>,

    /// Supported output modalities (e.g. `["text"]`).
    pub output: Vec<String>,
}

/// Pricing information for a model.
///
/// Costs are in the units defined by `models.dev` (USD per million tokens).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Input token cost.
    pub input: f64,

    /// Output token cost.
    pub output: f64,

    /// Reasoning token cost.
    pub reasoning: Option<f64>,

    /// Cache read cost.
    pub cache_read: Option<f64>,

    /// Cache write cost.
    pub cache_write: Option<f64>,

    /// Audio input cost.
    pub input_audio: Option<f64>,

    /// Audio output cost.
    pub output_audio: Option<f64>,

    /// Input cost for prompts exceeding 200k tokens (provider-specific tier).
    pub context_over_200k: Option<Value>,
}

/// Token limit information for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Limit {
    /// Context window size in tokens.
    pub context: u64,

    /// Maximum output tokens.
    pub output: u64,

    /// Maximum input tokens (when distinct from context).
    pub input: Option<u64>,
}
