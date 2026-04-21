use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ExtraMap;

type ErasedAny = dyn Any + Send + Sync;

/// Type-erased bag of per-request options keyed by Rust type.
///
/// Providers can define custom option types and retrieve them by type at
/// request time, enabling provider-specific configuration without polluting
/// the shared [`ChatRequest`](crate::ChatRequest) surface.
///
/// Stored values must be `Clone` at insertion time so cloning a
/// [`ChatRequest`](crate::ChatRequest) preserves its typed options.
#[derive(Clone, Default)]
pub struct RequestOptions {
    inner: HashMap<TypeId, ErasedValue>,
}

impl RequestOptions {
    /// Create an empty typed request option bag
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a typed request option
    pub fn insert<T>(&mut self, value: T)
    where
        T: Clone + Send + Sync + 'static,
    {
        self.inner
            .insert(TypeId::of::<T>(), ErasedValue::new(value));
    }

    /// Borrow a typed request option by type
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

    /// Mutably borrow a typed request option by type
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Send + Sync + 'static,
    {
        self.inner
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<T>()
    }

    /// Remove and return a typed request option by type
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Send + Sync + 'static,
    {
        self.inner
            .remove(&TypeId::of::<T>())
            .and_then(|entry| entry.into_any().downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Returns whether a typed option of `T` is present
    #[must_use]
    pub fn contains<T>(&self) -> bool
    where
        T: Send + Sync + 'static,
    {
        self.inner.contains_key(&TypeId::of::<T>())
    }

    /// Borrow a typed option, inserting a default value produced by `f` when
    /// none is present.
    ///
    /// The returned reference is always valid: if no value of type `T` was
    /// stored, `f` is invoked, its result is inserted, and a mutable borrow
    /// of the stored value is returned. Otherwise the existing value is
    /// borrowed in place without invoking `f`.
    ///
    /// Useful for providers or wrappers that maintain an accumulating
    /// request-scoped configuration (for example, appending beta flags to a
    /// provider-specific options struct) without having to pattern-match on
    /// the absence of a prior value.
    pub fn get_or_insert_with<T, F>(&mut self, f: F) -> &mut T
    where
        T: Clone + Send + Sync + 'static,
        F: FnOnce() -> T,
    {
        self.inner
            .entry(TypeId::of::<T>())
            .or_insert_with(|| ErasedValue::new(f()))
            .as_any_mut()
            .downcast_mut::<T>()
            .expect("type id matches inserted value type")
    }

    /// Returns whether the bag contains no typed options
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of stored typed options
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl fmt::Debug for RequestOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestOptions")
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
            // INVARIANT: `clone_fn` is created alongside the value with the same type
            // parameter T, so the downcast always succeeds unless ErasedValue is misused.
            // We use `expect` rather than `unwrap_unchecked` to get a clear panic message
            // if the invariant is ever violated, without introducing unsafe code.
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

/// Type-erased bag of provider-specific metadata attached to a response.
///
/// Providers insert typed metadata values after a chat completion. Consumers
/// retrieve them by type or serialize the bag to JSON for logging.
/// Portable JSON metadata may also be stored directly, but callers should use
/// distinct keys rather than intentionally reusing keys owned by typed
/// metadata. Lossy export preserves portable values on collision, while strict
/// export returns an error.
///
/// Typed entries must be cloneable at insertion time so cloning a
/// [`ChatResponse`](crate::ChatResponse) preserves attached metadata.
#[derive(Clone, Default)]
pub struct ResponseMetadata {
    inner: HashMap<TypeId, ResponseMetadataEntry>,
    portable: ExtraMap,
}

impl ResponseMetadata {
    /// Create an empty response metadata bag
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow typed metadata by type
    #[must_use]
    pub fn get<T>(&self) -> Option<&T>
    where
        T: ResponseMetadataType,
    {
        self.inner
            .get(&TypeId::of::<T>())?
            .inner
            .as_any()
            .downcast_ref::<T>()
    }

    /// Mutably borrow typed metadata by type
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: ResponseMetadataType,
    {
        self.inner
            .get_mut(&TypeId::of::<T>())?
            .inner
            .as_any_mut()
            .downcast_mut::<T>()
    }

    /// Insert or replace typed metadata
    pub fn insert<T>(&mut self, value: T)
    where
        T: ResponseMetadataType + Clone,
    {
        self.inner
            .insert(TypeId::of::<T>(), ResponseMetadataEntry::new(value));
    }

    /// Returns whether typed metadata of `T` is present
    #[must_use]
    pub fn contains<T>(&self) -> bool
    where
        T: ResponseMetadataType,
    {
        self.inner.contains_key(&TypeId::of::<T>())
    }

    /// Insert or replace portable JSON metadata by key
    pub fn insert_portable(
        &mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Option<serde_json::Value> {
        // Portable metadata remains flexible by design; collision handling is
        // deferred to export so callers can choose strict or lossy behavior.
        self.portable.insert(key.into(), value)
    }

    /// Borrow portable JSON metadata by key
    #[must_use]
    pub fn get_portable(&self, key: &str) -> Option<&serde_json::Value> {
        self.portable.get(key)
    }

    /// Borrow the portable JSON metadata map
    #[must_use]
    pub fn portable(&self) -> &ExtraMap {
        &self.portable
    }

    /// Mutably borrow the portable JSON metadata map
    pub fn portable_mut(&mut self) -> &mut ExtraMap {
        &mut self.portable
    }

    /// Extend the portable JSON metadata map with additional entries
    pub fn extend_portable(&mut self, entries: ExtraMap) {
        self.portable.extend(entries);
    }

    /// Create metadata from portable JSON entries only
    #[must_use]
    pub fn from_portable(portable: ExtraMap) -> Self {
        Self {
            inner: HashMap::new(),
            portable,
        }
    }

    /// Converts metadata into a portable JSON map, skipping typed entries that
    /// fail to serialize.
    #[must_use]
    pub fn to_portable_map(&self) -> ExtraMap {
        let mut portable = self.portable.clone();

        for entry in self.inner.values() {
            let key = entry.inner.key();

            if portable.contains_key(key) {
                continue;
            }

            if let Ok(value) = serde_json::to_value(&entry.inner) {
                portable.insert(key.to_owned(), value);
            }
        }

        portable
    }

    /// Converts metadata into a portable JSON map.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`](crate::Error::Serialization) if any
    /// typed metadata entry fails to serialize or if a typed metadata export
    /// key collides with an existing portable metadata key.
    pub fn try_to_portable_map(&self) -> crate::Result<ExtraMap> {
        let mut portable = self.portable.clone();

        for entry in self.inner.values() {
            let key = entry.inner.key();

            if portable.contains_key(key) {
                return Err(crate::Error::serialization(format!(
                    "response metadata key collision for '{key}'"
                )));
            }

            let value = serde_json::to_value(&entry.inner)?;
            portable.insert(key.to_owned(), value);
        }

        Ok(portable)
    }

    /// Returns whether both typed and portable metadata are empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty() && self.portable.is_empty()
    }

    /// Returns the total number of visible metadata keys
    #[must_use]
    pub fn len(&self) -> usize {
        let mut keys: HashSet<&str> = self.portable.keys().map(String::as_str).collect();
        keys.extend(self.inner.values().map(|entry| entry.inner.key()));
        keys.len()
    }
}

impl serde::Serialize for ResponseMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let portable = self
            .try_to_portable_map()
            .map_err(serde::ser::Error::custom)?;
        let mut map = serializer.serialize_map(Some(portable.len()))?;
        for (key, value) in portable {
            map.serialize_entry(&key, &value)?;
        }
        map.end()
    }
}

impl fmt::Debug for ResponseMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ResponseMetadata");

        match self.try_to_portable_map() {
            Ok(portable) => debug.field("json", &portable).finish(),
            Err(error) => debug
                .field("json", &self.to_portable_map())
                .field("export_error", &error.to_string())
                .finish(),
        }
    }
}

/// Marker trait for typed response metadata entries
pub trait ResponseMetadataType: Any + Send + Sync + serde::Serialize {
    /// Stable export key for this metadata type
    const KEY: &'static str;
}

trait ErasedResponseMetadata: erased_serde::Serialize + Send + Sync {
    fn as_any(&self) -> &(dyn Any + Send + Sync);
    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync);
    fn clone_box(&self) -> Box<dyn ErasedResponseMetadata>;
    fn key(&self) -> &'static str;
}

erased_serde::serialize_trait_object!(ErasedResponseMetadata);

impl<T> ErasedResponseMetadata for T
where
    T: ResponseMetadataType + Clone,
{
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        self
    }

    fn clone_box(&self) -> Box<dyn ErasedResponseMetadata> {
        Box::new(self.clone())
    }

    fn key(&self) -> &'static str {
        T::KEY
    }
}

struct ResponseMetadataEntry {
    inner: Box<dyn ErasedResponseMetadata>,
}

impl ResponseMetadataEntry {
    fn new<T>(value: T) -> Self
    where
        T: ResponseMetadataType + Clone,
    {
        Self {
            inner: Box::new(value),
        }
    }
}

impl Clone for ResponseMetadataEntry {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_box(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DemoOption {
        enabled: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
    struct DemoMetadata {
        request_id: String,
    }

    impl ResponseMetadataType for DemoMetadata {
        const KEY: &'static str = "demo";
    }

    #[derive(Debug, Clone)]
    struct BrokenMetadata;

    impl serde::Serialize for BrokenMetadata {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(serde::ser::Error::custom("broken metadata"))
        }
    }

    impl ResponseMetadataType for BrokenMetadata {
        const KEY: &'static str = "broken";
    }

    #[test]
    fn request_options_insert_get_get_mut_remove() {
        let mut options = RequestOptions::new();
        options.insert(DemoOption { enabled: true });

        assert_eq!(
            options.get::<DemoOption>(),
            Some(&DemoOption { enabled: true })
        );
        assert!(options.contains::<DemoOption>());

        options.get_mut::<DemoOption>().unwrap().enabled = false;
        assert_eq!(
            options.get::<DemoOption>(),
            Some(&DemoOption { enabled: false })
        );

        let removed = options.remove::<DemoOption>().unwrap();
        assert_eq!(removed, DemoOption { enabled: false });
        assert!(!options.contains::<DemoOption>());
    }

    #[test]
    fn request_options_len_and_is_empty_track_changes() {
        let mut options = RequestOptions::new();
        assert!(options.is_empty());
        assert_eq!(options.len(), 0);

        options.insert(DemoOption { enabled: true });
        assert!(!options.is_empty());
        assert_eq!(options.len(), 1);

        let _ = options.remove::<DemoOption>();
        assert!(options.is_empty());
        assert_eq!(options.len(), 0);
    }

    #[test]
    fn request_options_get_or_insert_with_inserts_when_missing() {
        let mut options = RequestOptions::new();
        let value: &mut DemoOption = options.get_or_insert_with(|| DemoOption { enabled: true });
        assert_eq!(value, &mut DemoOption { enabled: true });

        value.enabled = false;

        assert_eq!(
            options.get::<DemoOption>(),
            Some(&DemoOption { enabled: false })
        );
        assert_eq!(options.len(), 1);
    }

    #[test]
    fn request_options_get_or_insert_with_does_not_invoke_when_present() {
        let mut options = RequestOptions::new();
        options.insert(DemoOption { enabled: true });

        let mut invoked = false;
        let value: &mut DemoOption = options.get_or_insert_with(|| {
            invoked = true;
            DemoOption { enabled: false }
        });

        assert!(!invoked, "factory should not run when value is present");
        assert_eq!(value, &mut DemoOption { enabled: true });
        assert_eq!(options.len(), 1);
    }

    #[test]
    fn request_options_clone_is_independent() {
        let mut options = RequestOptions::new();
        options.insert(DemoOption { enabled: true });

        let mut cloned = options.clone();
        cloned.get_mut::<DemoOption>().unwrap().enabled = false;

        assert_eq!(
            options.get::<DemoOption>(),
            Some(&DemoOption { enabled: true })
        );
        assert_eq!(
            cloned.get::<DemoOption>(),
            Some(&DemoOption { enabled: false })
        );
    }

    #[test]
    fn response_metadata_clone_is_independent_and_preserves_json() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });

        let mut cloned = metadata.clone();
        cloned.get_mut::<DemoMetadata>().unwrap().request_id = "req_456".into();

        assert_eq!(
            metadata.get::<DemoMetadata>(),
            Some(&DemoMetadata {
                request_id: "req_123".into(),
            })
        );
        assert_eq!(
            serde_json::to_value(&cloned).unwrap(),
            serde_json::json!({"demo": {"request_id": "req_456"}})
        );
    }

    #[test]
    fn response_metadata_portable_entries_round_trip() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert_portable("provider_request_id", serde_json::json!("req_789"));

        assert_eq!(
            metadata.get_portable("provider_request_id"),
            Some(&serde_json::json!("req_789"))
        );
        assert_eq!(
            serde_json::to_value(&metadata).unwrap(),
            serde_json::json!({"provider_request_id": "req_789"})
        );
    }

    #[test]
    fn response_metadata_portable_entries_can_override_typed_serialization() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        metadata.insert_portable("demo", serde_json::json!({"request_id": "portable"}));

        assert_eq!(
            metadata.to_portable_map(),
            serde_json::Map::from_iter([(
                "demo".into(),
                serde_json::json!({"request_id": "portable"}),
            )])
        );
    }

    #[test]
    fn response_metadata_try_to_portable_map_reports_key_collisions() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        metadata.insert_portable("demo", serde_json::json!({"request_id": "portable"}));

        let error = metadata.try_to_portable_map().unwrap_err();
        assert!(matches!(error, crate::Error::Serialization(_)));
        assert!(
            error
                .to_string()
                .contains("response metadata key collision for 'demo'")
        );
    }

    #[test]
    fn response_metadata_try_to_portable_map_reports_serialization_errors() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(BrokenMetadata);

        let error = metadata.try_to_portable_map().unwrap_err();
        assert!(matches!(error, crate::Error::Serialization(_)));
    }

    #[test]
    fn response_metadata_to_portable_map_skips_unserializable_typed_entries() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(BrokenMetadata);
        metadata.insert_portable("portable", serde_json::json!(true));

        assert_eq!(
            metadata.to_portable_map(),
            serde_json::Map::from_iter([("portable".into(), serde_json::json!(true))])
        );
    }

    #[test]
    fn response_metadata_len_counts_unique_keys_without_serializing() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        metadata.insert(BrokenMetadata);
        metadata.insert_portable("demo", serde_json::json!({"request_id": "portable"}));
        metadata.insert_portable("portable", serde_json::json!(true));

        assert_eq!(metadata.len(), 3);
    }

    #[test]
    fn response_metadata_serialize_fails_for_unserializable_typed_entries() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(BrokenMetadata);

        assert!(serde_json::to_value(&metadata).is_err());
    }

    #[test]
    fn response_metadata_serialize_fails_for_portable_typed_key_collisions() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        metadata.insert_portable("demo", serde_json::json!({"request_id": "portable"}));

        assert!(serde_json::to_value(&metadata).is_err());
    }

    #[test]
    fn response_metadata_debug_is_lossy_for_unserializable_typed_entries() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(BrokenMetadata);
        metadata.insert_portable("portable", serde_json::json!(true));

        let debug = format!("{metadata:?}");

        assert!(debug.contains("ResponseMetadata"));
        assert!(debug.contains("portable"));
        assert!(debug.contains("export_error"));
    }

    #[test]
    fn response_metadata_debug_is_lossy_for_portable_typed_key_collisions() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        metadata.insert_portable("demo", serde_json::json!({"request_id": "portable"}));

        let debug = format!("{metadata:?}");

        assert!(debug.contains("ResponseMetadata"));
        assert!(debug.contains("portable"));
        assert!(debug.contains("export_error"));
    }
}
