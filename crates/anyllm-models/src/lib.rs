//! Typed Rust structs for the [`models.dev`](https://models.dev) model registry.
//!
//! This crate deserializes the `models.dev` JSON registry into typed Rust
//! values. It supports both static JSON input and an optional live fetch
//! path from `https://models.dev/api.json`.
//!
//! ## Advisory data
//!
//! All metadata in this crate is advisory. The source of truth is the
//! upstream `models.dev` data, not provider APIs. Fields may be missing,
//! stale, or incomplete.
//!
//! ## Usage
//!
//! ```no_run
//! let json = std::fs::read_to_string("api.json").unwrap();
//! let registry: anyllm_models::Registry = anyllm_models::from_str(&json).unwrap();
//!
//! if let Some(provider) = registry.get("openai") {
//!     if let Some(model) = provider.models.get("gpt-4o") {
//!         println!("context: {:?}", model.limit.context);
//!     }
//! }
//! ```

use std::collections::HashMap;

mod model;
mod provider;

pub use model::{Cost, Limit, Modalities, Model};
pub use provider::Provider;

#[cfg(feature = "http")]
mod fetch;

#[cfg(feature = "http")]
pub use fetch::{fetch, FetchOptions, FetchResult};

/// The top-level registry: a map from provider id to [`Provider`].
pub type Registry = HashMap<String, Provider>;

/// Deserialize a [`Registry`] from a JSON string.
pub fn from_str(s: &str) -> serde_json::Result<Registry> {
    serde_json::from_str(s)
}

/// Deserialize a [`Registry`] from a byte slice.
pub fn from_slice(s: &[u8]) -> serde_json::Result<Registry> {
    serde_json::from_slice(s)
}

/// Deserialize a [`Registry`] from a reader.
pub fn from_reader<R: std::io::Read>(r: R) -> serde_json::Result<Registry> {
    serde_json::from_reader(r)
}
