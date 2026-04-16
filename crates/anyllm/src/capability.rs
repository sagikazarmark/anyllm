use serde::{Deserialize, Serialize};

/// Support state for a provider/model capability query.
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

impl From<bool> for CapabilitySupport {
    fn from(supported: bool) -> Self {
        if supported {
            Self::Supported
        } else {
            Self::Unsupported
        }
    }
}
