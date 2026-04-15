//! Request-level instruction types.
//!
//! `SystemPrompt` carries a single system instruction plus a typed,
//! per-prompt options bag (`SystemOptions`) for provider-specific metadata
//! like Anthropic cache hints. See the design doc at
//! `docs/superpowers/specs/2026-04-15-instruction-carrier-fidelity-design.md`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

type ErasedAny = dyn Any + Send + Sync;

/// Type-erased bag of per-prompt options keyed by Rust type.
///
/// Mirrors [`RequestOptions`](crate::RequestOptions) semantics but scoped to
/// a single [`SystemPrompt`]. Each prompt owns its own bag so distinct
/// prompts can carry distinct provider-specific data (e.g., different
/// Anthropic `cache_control` hints).
///
/// Stored values must be `Clone` at insertion time so cloning a
/// [`SystemPrompt`] preserves its typed options.
#[derive(Clone, Default)]
pub struct SystemOptions {
    inner: HashMap<TypeId, ErasedValue>,
}

impl SystemOptions {
    /// Create an empty typed system-prompt option bag.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a typed option.
    pub fn insert<T>(&mut self, value: T)
    where
        T: Clone + Send + Sync + 'static,
    {
        self.inner
            .insert(TypeId::of::<T>(), ErasedValue::new(value));
    }

    /// Borrow a typed option by type.
    #[must_use]
    pub fn get<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.inner
            .get(&TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<T>()
    }

    /// Mutably borrow a typed option by type.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Send + Sync + 'static,
    {
        self.inner
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<T>()
    }

    /// Remove and return a typed option by type.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Send + Sync + 'static,
    {
        self.inner
            .remove(&TypeId::of::<T>())
            .and_then(|entry| entry.into_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Returns whether a typed option of `T` is present.
    #[must_use]
    pub fn contains<T>(&self) -> bool
    where
        T: Send + Sync + 'static,
    {
        self.inner.contains_key(&TypeId::of::<T>())
    }

    /// Returns whether the bag is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of stored typed options.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl fmt::Debug for SystemOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemOptions")
            .field("len", &self.len())
            .finish()
    }
}

struct ErasedValue {
    value: Box<ErasedAny>,
    clone_fn: fn(&ErasedAny) -> Box<ErasedAny>,
}

impl ErasedValue {
    fn new<T>(value: T) -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        fn clone_value<T>(value: &ErasedAny) -> Box<ErasedAny>
        where
            T: Clone + Send + Sync + 'static,
        {
            let typed = value
                .downcast_ref::<T>()
                .expect("value type did not match clone function");
            Box::new(typed.clone())
        }

        Self {
            value: Box::new(value),
            clone_fn: clone_value::<T>,
        }
    }

    fn as_any(&self) -> &ErasedAny {
        self.value.as_ref()
    }

    fn as_any_mut(&mut self) -> &mut ErasedAny {
        self.value.as_mut()
    }

    fn into_any(self) -> Box<ErasedAny> {
        self.value
    }
}

impl Clone for ErasedValue {
    fn clone(&self) -> Self {
        Self {
            value: (self.clone_fn)(self.value.as_ref()),
            clone_fn: self.clone_fn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Flag(bool);

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Count(u32);

    #[test]
    fn system_options_insert_and_get() {
        let mut opts = SystemOptions::new();
        opts.insert(Flag(true));
        assert_eq!(opts.get::<Flag>(), Some(&Flag(true)));
    }

    #[test]
    fn system_options_stores_multiple_types() {
        let mut opts = SystemOptions::new();
        opts.insert(Flag(true));
        opts.insert(Count(7));
        assert_eq!(opts.len(), 2);
        assert_eq!(opts.get::<Flag>(), Some(&Flag(true)));
        assert_eq!(opts.get::<Count>(), Some(&Count(7)));
    }

    #[test]
    fn system_options_insert_replaces_same_type() {
        let mut opts = SystemOptions::new();
        opts.insert(Flag(true));
        opts.insert(Flag(false));
        assert_eq!(opts.get::<Flag>(), Some(&Flag(false)));
        assert_eq!(opts.len(), 1);
    }

    #[test]
    fn system_options_remove_returns_value() {
        let mut opts = SystemOptions::new();
        opts.insert(Count(42));
        let removed: Option<Count> = opts.remove::<Count>();
        assert_eq!(removed, Some(Count(42)));
        assert!(opts.get::<Count>().is_none());
    }

    #[test]
    fn system_options_contains_and_is_empty() {
        let mut opts = SystemOptions::new();
        assert!(opts.is_empty());
        assert!(!opts.contains::<Flag>());
        opts.insert(Flag(true));
        assert!(!opts.is_empty());
        assert!(opts.contains::<Flag>());
    }

    #[test]
    fn system_options_clone_preserves_values() {
        let mut opts = SystemOptions::new();
        opts.insert(Flag(true));
        opts.insert(Count(3));
        let clone = opts.clone();
        assert_eq!(clone.get::<Flag>(), Some(&Flag(true)));
        assert_eq!(clone.get::<Count>(), Some(&Count(3)));
    }

    #[test]
    fn system_options_get_mut_modifies_in_place() {
        let mut opts = SystemOptions::new();
        opts.insert(Count(1));
        if let Some(c) = opts.get_mut::<Count>() {
            c.0 = 99;
        }
        assert_eq!(opts.get::<Count>(), Some(&Count(99)));
    }
}
