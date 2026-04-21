use serde::{Deserialize, Serialize};

/// Support state for a provider/model capability query.
///
/// The enum is intentionally a closed three-state set (`Supported`,
/// `Unsupported`, `Unknown`). It is not `#[non_exhaustive]` because the answer
/// to a capability query is either yes, no, or unknown — there is no fourth
/// state to add later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CapabilitySupport {
    /// The provider explicitly supports this capability for the queried model.
    Supported,
    /// The provider explicitly does not support this capability for the queried model.
    Unsupported,
    /// The provider does not provide a definitive answer for this capability.
    ///
    /// Treat this as "unknown, ask carefully" rather than as support.
    #[default]
    Unknown,
}

impl CapabilitySupport {
    /// Returns `true` only when the provider explicitly reports support.
    ///
    /// [`Unknown`](Self::Unknown) is **not** treated as supported: code that
    /// gates behavior on capability reporting should opt in only when the
    /// provider is certain.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Returns `true` when the provider gave a definitive answer, regardless
    /// of whether that answer was positive or negative.
    ///
    /// Equivalent to `self != CapabilitySupport::Unknown`. Useful when a
    /// caller wants to distinguish "the provider told us" from "we have to
    /// guess."
    #[must_use]
    pub const fn is_known(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_supported_matches_only_supported() {
        assert!(CapabilitySupport::Supported.is_supported());
        assert!(!CapabilitySupport::Unsupported.is_supported());
        assert!(!CapabilitySupport::Unknown.is_supported());
    }

    #[test]
    fn is_known_matches_supported_and_unsupported() {
        assert!(CapabilitySupport::Supported.is_known());
        assert!(CapabilitySupport::Unsupported.is_known());
        assert!(!CapabilitySupport::Unknown.is_known());
    }

    #[test]
    fn default_is_unknown_and_not_known() {
        let value = CapabilitySupport::default();
        assert_eq!(value, CapabilitySupport::Unknown);
        assert!(!value.is_known());
        assert!(!value.is_supported());
    }
}
