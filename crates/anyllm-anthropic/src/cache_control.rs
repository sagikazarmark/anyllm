//! Typed option for Anthropic's `cache_control` per-block cache hint.
//!
//! Attach a [`CacheControl`] to a
//! [`SystemPrompt`](anyllm::SystemPrompt) so the Anthropic adapter emits
//! the corresponding `cache_control` field on that block's wire payload.
//! Adapters for other providers ignore this option.

use serde_json::{Value, json};

/// Anthropic per-block cache control hint.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheControl {
    /// Ephemeral cache — the default 5-minute TTL.
    Ephemeral,
}

impl CacheControl {
    /// Convenience constructor for the default ephemeral cache variant.
    #[must_use]
    pub fn ephemeral() -> Self {
        Self::Ephemeral
    }

    /// Render to the wire JSON shape Anthropic expects.
    pub(crate) fn to_wire(&self) -> Value {
        match self {
            Self::Ephemeral => json!({"type": "ephemeral"}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ephemeral_constructor() {
        let c = CacheControl::ephemeral();
        assert_eq!(c, CacheControl::Ephemeral);
    }

    #[test]
    fn ephemeral_wire_shape() {
        let c = CacheControl::ephemeral();
        assert_eq!(c.to_wire(), json!({"type": "ephemeral"}));
    }

    #[test]
    fn clone_and_equality() {
        let a = CacheControl::ephemeral();
        let b = a.clone();
        assert_eq!(a, b);
    }
}
