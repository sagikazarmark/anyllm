use std::collections::HashMap;

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

    /// Provider-specific overrides for this model (e.g. alternative API
    /// endpoint, SDK package, or request shape).
    pub provider: Option<ModelProvider>,

    /// Interleaved reasoning configuration.
    pub interleaved: Option<Interleaved>,

    /// Experimental feature modes and their configuration.
    pub experimental: Option<Experimental>,
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

    /// Tiered pricing for prompts exceeding 200k tokens.
    pub context_over_200k: Option<OverrideCost>,
}

/// Partial cost override used in tiered pricing and experimental modes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverrideCost {
    /// Input token cost.
    pub input: Option<f64>,

    /// Output token cost.
    pub output: Option<f64>,

    /// Reasoning token cost.
    pub reasoning: Option<f64>,

    /// Cache read cost.
    pub cache_read: Option<f64>,

    /// Cache write cost.
    pub cache_write: Option<f64>,
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

/// Provider-specific overrides for a model entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelProvider {
    /// Alternative API endpoint URL.
    pub api: Option<String>,

    /// Alternative NPM package for the provider SDK.
    pub npm: Option<String>,

    /// Request shape override (e.g. `"completions"`).
    pub shape: Option<String>,
}

/// Interleaved reasoning configuration.
///
/// Either a simple boolean (`true`) or a configuration object specifying
/// which response field carries reasoning content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Interleaved {
    /// Interleaved reasoning is enabled with default configuration.
    Enabled(bool),

    /// Interleaved reasoning with an explicit field name.
    Config {
        /// The response field containing reasoning content
        /// (e.g. `"reasoning_content"`, `"reasoning_details"`).
        field: String,
    },
}

/// Experimental feature modes for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Experimental {
    /// Named experimental modes (e.g. `"fast"`), each with its own cost
    /// and provider configuration.
    pub modes: HashMap<String, ExperimentalMode>,
}

/// Configuration for a single experimental mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentalMode {
    /// Cost overrides when using this mode.
    pub cost: Option<OverrideCost>,

    /// Provider-specific request overrides for this mode.
    pub provider: Option<ExperimentalModeProvider>,
}

/// Provider-specific request overrides for an experimental mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentalModeProvider {
    /// Additional request body fields.
    pub body: Option<HashMap<String, Value>>,

    /// Additional request headers.
    pub headers: Option<HashMap<String, String>>,
}
