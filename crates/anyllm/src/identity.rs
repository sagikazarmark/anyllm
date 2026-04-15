use std::sync::Arc;

/// Shared provider identity surface for chat and embedding providers.
///
/// Returns the provider identity string used for logs, tracing, fixtures, and
/// diagnostics (for example `"anthropic"` or `"openai"`).
///
/// This is an open string surface rather than a closed enum so wrappers and
/// custom providers can preserve their existing naming. Callers should not
/// treat it as a stable capability-dispatch mechanism.
pub trait ProviderIdentity: Send + Sync {
    /// Returns the provider identity name.
    fn provider_name(&self) -> &'static str {
        "unknown"
    }
}

impl<T> ProviderIdentity for &T
where
    T: ProviderIdentity + ?Sized,
{
    fn provider_name(&self) -> &'static str {
        T::provider_name(*self)
    }
}

impl<T> ProviderIdentity for Box<T>
where
    T: ProviderIdentity + ?Sized,
{
    fn provider_name(&self) -> &'static str {
        T::provider_name(self.as_ref())
    }
}

impl<T> ProviderIdentity for Arc<T>
where
    T: ProviderIdentity + ?Sized,
{
    fn provider_name(&self) -> &'static str {
        T::provider_name(self.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Named;

    impl ProviderIdentity for Named {
        fn provider_name(&self) -> &'static str {
            "named"
        }
    }

    struct Default;

    impl ProviderIdentity for Default {}

    #[test]
    fn default_method_returns_unknown() {
        assert_eq!(Default.provider_name(), "unknown");
    }

    #[test]
    fn concrete_impl_returns_configured_name() {
        assert_eq!(Named.provider_name(), "named");
    }

    #[test]
    fn ref_forwarding_delegates() {
        let named = Named;
        let borrowed: &Named = &named;
        assert_eq!(borrowed.provider_name(), "named");
    }

    #[test]
    fn box_forwarding_delegates() {
        let boxed: Box<dyn ProviderIdentity> = Box::new(Named);
        assert_eq!(boxed.provider_name(), "named");
    }

    #[test]
    fn arc_forwarding_delegates() {
        let arced: Arc<dyn ProviderIdentity> = Arc::new(Named);
        assert_eq!(arced.provider_name(), "named");
    }
}
