use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::utils::{concat_segments, write_segments};
use crate::{ContentBlock, Message, ResponseMetadata, ToolCallRef, Usage};

/// A provider-agnostic chat completion response.
///
/// Use [`ChatResponse::new`] and the fluent setters to construct responses.
/// Provider-specific metadata that does not fit the portable core belongs in
/// [`ResponseMetadata`].
///
/// Does not implement `PartialEq` because [`ResponseMetadata`] contains
/// type-erased provider-specific values. Use [`ChatResponseRecord`]
/// for portable, equality-comparable representations (e.g. in tests).
/// Converting through `ChatResponseRecord` is lossy for typed metadata, which is
/// preserved only as portable JSON.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ChatResponse {
    /// Ordered assistant content blocks returned by the provider.
    pub content: Vec<ContentBlock>,
    /// Why the provider stopped generating, when reported.
    pub finish_reason: Option<FinishReason>,
    /// Token-usage information for the request, when available.
    pub usage: Option<Usage>,
    /// Provider-reported model identifier for the completed response.
    pub model: Option<String>,
    /// Provider-assigned response identifier.
    pub id: Option<String>,
    /// Provider-specific typed and portable metadata attached to the response.
    pub metadata: ResponseMetadata,
}

impl ChatResponse {
    /// Create a response from an ordered list of assistant content blocks.
    #[must_use]
    pub fn new(content: Vec<ContentBlock>) -> Self {
        Self {
            content,
            finish_reason: None,
            usage: None,
            model: None,
            id: None,
            metadata: ResponseMetadata::new(),
        }
    }

    /// Set the response finish reason.
    #[must_use]
    pub fn finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// Attach token-usage information.
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Attach the provider-reported model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Attach the provider-assigned response identifier.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Replace the response metadata bag.
    #[must_use]
    pub fn metadata(mut self, metadata: ResponseMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Iterate over all text segments in order.
    pub fn text_segments(&self) -> impl Iterator<Item = &str> {
        self.content.iter().filter_map(|b| b.as_text())
    }

    /// Concatenate all text blocks into one string.
    pub fn text(&self) -> Option<String> {
        concat_segments(self.text_segments())
    }

    /// Return concatenated text, or an empty string when the response has no text.
    pub fn text_or_empty(&self) -> String {
        self.text().unwrap_or_default()
    }

    /// Borrow the first text block, if present.
    pub fn first_text(&self) -> Option<&str> {
        self.text_segments().next()
    }

    /// Borrow the first reasoning block, if present.
    pub fn first_reasoning(&self) -> Option<&str> {
        self.reasoning_segments().next()
    }

    /// Append all text blocks to `output`, returning whether any text was written.
    pub fn write_text_to(&self, output: &mut String) -> bool {
        write_segments(output, self.text_segments())
    }

    /// Iterate over all reasoning text segments in order.
    pub fn reasoning_segments(&self) -> impl Iterator<Item = &str> {
        self.content.iter().filter_map(|b| b.as_reasoning())
    }

    /// Concatenate all reasoning blocks into one string.
    pub fn reasoning_text(&self) -> Option<String> {
        concat_segments(self.reasoning_segments())
    }

    /// Append all reasoning blocks to `output`, returning whether any text was written.
    pub fn write_reasoning_text_to(&self, output: &mut String) -> bool {
        write_segments(output, self.reasoning_segments())
    }

    /// Iterate over all tool-call blocks in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = ToolCallRef<'_>> {
        self.content.iter().filter_map(|b| b.as_tool_call())
    }

    /// Borrow the first tool call, if present.
    pub fn first_tool_call(&self) -> Option<ToolCallRef<'_>> {
        self.tool_calls().next()
    }

    /// Return whether the response contains any tool-call blocks.
    pub fn has_tool_calls(&self) -> bool {
        self.content.iter().any(|b| b.as_tool_call().is_some())
    }

    /// Convert this response into an assistant message by cloning its content blocks.
    pub fn to_assistant_message(&self) -> Message {
        self.into()
    }

    /// Consume this response into an assistant message.
    pub fn into_assistant_message(self) -> Message {
        Message::Assistant {
            content: self.content,
            name: None,
            extensions: None,
        }
    }

    /// Build a portable response record, failing if typed metadata cannot be serialized.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`](crate::Error::Serialization) if typed
    /// metadata cannot be represented as portable JSON.
    pub fn try_to_record(&self) -> crate::Result<ChatResponseRecord> {
        ChatResponseRecord::try_from_response(self)
    }

    /// Build a portable response record, skipping typed metadata that cannot be serialized.
    #[must_use]
    pub fn to_record_lossy(&self) -> ChatResponseRecord {
        self.into()
    }

    /// Consume this response into a portable record, skipping typed metadata that
    /// cannot be serialized.
    #[must_use]
    pub fn into_record_lossy(self) -> ChatResponseRecord {
        let metadata = self.metadata.to_portable_map();

        ChatResponseRecord {
            content: self.content,
            finish_reason: self.finish_reason,
            usage: self.usage,
            model: self.model,
            id: self.id,
            metadata,
        }
    }

    /// Build a structured log value from the exact portable representation,
    /// failing if typed metadata cannot be serialized.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`](crate::Error::Serialization) if typed
    /// metadata cannot be represented as portable JSON.
    pub fn try_to_log_value(&self) -> crate::Result<serde_json::Value> {
        serde_json::to_value(self.try_to_record()?).map_err(crate::Error::from)
    }

    /// Build a portable JSON log value, skipping typed metadata that cannot be serialized.
    #[must_use]
    pub fn to_log_value(&self) -> serde_json::Value {
        serde_json::to_value(self.to_record_lossy())
            .expect("ChatResponseRecord serialization should be infallible")
    }
}

/// Portable, serde-friendly snapshot of a [`ChatResponse`].
///
/// This record is intended for logging, fixtures, and replayable artifacts.
/// It preserves portable response data and serializes metadata as plain JSON.
/// Provider-specific typed [`ResponseMetadata`] entries are flattened into that
/// portable JSON representation.
///
/// Converting a `ChatResponse` into a `ChatResponseRecord` is lossless for the
/// portable representation. Rebuilding a `ChatResponse` from a record is lossy
/// with respect to typed metadata: the rebuilt response contains only portable
/// metadata values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatResponseRecord {
    /// Ordered assistant content blocks.
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Why generation stopped, when available.
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Token-usage information.
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Provider-reported model identifier.
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Provider-assigned response identifier.
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    /// Portable JSON metadata attached to the response.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl From<ChatResponse> for ChatResponseRecord {
    fn from(response: ChatResponse) -> Self {
        response.into_record_lossy()
    }
}

impl From<&ChatResponse> for ChatResponseRecord {
    fn from(response: &ChatResponse) -> Self {
        Self::from_response_lossy(response)
    }
}

impl ChatResponseRecord {
    /// Build the exact portable representation of a response, failing if any
    /// typed metadata entry cannot be serialized.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`](crate::Error::Serialization) if typed
    /// metadata cannot be represented as portable JSON.
    pub fn try_from_response(response: &ChatResponse) -> crate::Result<Self> {
        Ok(Self {
            content: response.content.clone(),
            finish_reason: response.finish_reason.clone(),
            usage: response.usage.clone(),
            model: response.model.clone(),
            id: response.id.clone(),
            metadata: response.metadata.try_to_portable_map()?,
        })
    }

    /// Build the lossy portable representation of a response, skipping typed
    /// metadata entries that fail to serialize.
    #[must_use]
    pub fn from_response_lossy(response: &ChatResponse) -> Self {
        Self {
            content: response.content.clone(),
            finish_reason: response.finish_reason.clone(),
            usage: response.usage.clone(),
            model: response.model.clone(),
            id: response.id.clone(),
            metadata: response.metadata.to_portable_map(),
        }
    }

    /// Rebuild a `ChatResponse` from the portable record.
    ///
    /// This conversion is lossy with respect to typed metadata: the rebuilt
    /// response contains only portable JSON metadata via
    /// [`ResponseMetadata::from_portable`].
    #[must_use]
    pub fn into_chat_response_lossy(self) -> ChatResponse {
        ChatResponse {
            content: self.content,
            finish_reason: self.finish_reason,
            usage: self.usage,
            model: self.model,
            id: self.id,
            metadata: ResponseMetadata::from_portable(self.metadata),
        }
    }
}

impl From<ChatResponse> for Message {
    fn from(response: ChatResponse) -> Self {
        response.into_assistant_message()
    }
}

impl From<&ChatResponse> for Message {
    fn from(response: &ChatResponse) -> Self {
        Message::Assistant {
            content: response.content.clone(),
            name: None,
            extensions: None,
        }
    }
}

/// Why the model stopped generating.
///
/// The five-variant portable set is frozen for 0.1.0. The enum is
/// `#[non_exhaustive]`: callers pattern-matching on it must include a
/// catch-all arm, and new portable variants may be added in a minor version
/// if a cross-provider pattern emerges. Provider-specific reasons that do not
/// fit the portable set surface through [`Other`](Self::Other).
///
/// Serialization emits a fixed string per variant: `"stop"`, `"length"`,
/// `"tool_calls"`, `"content_filter"`. [`Other`](Self::Other) serializes as
/// the raw inner string. Deserialization and [`from_str`](Self::from_str) are
/// infallible: any string that does not match a known variant becomes
/// `Other(s)`, preserving round-trip fidelity through
/// [`ChatResponseRecord`].
///
/// Provider crates are responsible for mapping their wire-level reasons onto
/// this enum; inspect a provider's `wire.rs` for its current translation
/// table.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// The model completed its response naturally: end-of-turn signal,
    /// stop-sequence match, or equivalent provider-specific termination.
    Stop,

    /// Generation was truncated because the model reached a token budget,
    /// either the `max_tokens` on the request or the model's own context
    /// limit.
    Length,

    /// The model emitted one or more tool calls. The calls themselves appear
    /// in [`ChatResponse`]'s `content` field as [`ContentBlock::ToolCall`]
    /// entries.
    ToolCalls,

    /// Generation was interrupted by a provider's content-safety system. The
    /// response body may be partial or empty.
    ContentFilter,

    /// A finish reason the portable enum does not model. The inner string is
    /// the provider's raw wire token, preserved verbatim. Avoid pattern-
    /// matching on specific strings: the value is provider-specific and may
    /// change between provider versions.
    Other(String),
}

impl FinishReason {
    /// Return the stable string form used by serde and portable records.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Other(reason) => reason,
        }
    }
}

impl FromStr for FinishReason {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            other => FinishReason::Other(other.to_owned()),
        })
    }
}

impl Serialize for FinishReason {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FinishReasonVisitor;

        impl Visitor<'_> for FinishReasonVisitor {
            type Value = FinishReason;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string representing a finish reason")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<FinishReason, E> {
                Ok(FinishReason::from_str(value).expect("FinishReason::from_str is infallible"))
            }
        }

        deserializer.deserialize_str(FinishReasonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
    struct DemoMetadata {
        request_id: String,
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

    impl crate::ResponseMetadataType for DemoMetadata {
        const KEY: &'static str = "demo";
    }

    impl crate::ResponseMetadataType for BrokenMetadata {
        const KEY: &'static str = "broken";
    }

    #[test]
    fn finish_reason_known_values_round_trip() {
        let cases = [
            (FinishReason::Stop, "stop"),
            (FinishReason::Length, "length"),
            (FinishReason::ToolCalls, "tool_calls"),
            (FinishReason::ContentFilter, "content_filter"),
            (FinishReason::Other("custom_reason".into()), "custom_reason"),
        ];

        for (finish_reason, expected) in cases {
            let value = serde_json::to_value(&finish_reason).unwrap();
            assert_eq!(value, json!(expected));

            let round_tripped: FinishReason = serde_json::from_value(value).unwrap();
            assert_eq!(round_tripped, finish_reason);
        }
    }

    #[test]
    fn finish_reason_unknown_strings_round_trip_through_helpers() {
        assert_eq!(FinishReason::ToolCalls.as_str(), "tool_calls");
        assert_eq!(
            FinishReason::from_str("custom_reason").unwrap(),
            FinishReason::Other("custom_reason".into())
        );

        let finish_reason: FinishReason =
            serde_json::from_value(json!("some_unknown_reason")).unwrap();
        assert_eq!(
            finish_reason,
            FinishReason::Other("some_unknown_reason".into())
        );
    }

    #[test]
    fn response_record_preserves_metadata_json_for_logging() {
        let mut response = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage {
                input_tokens: Some(1),
                output_tokens: Some(1),
                total_tokens: Some(2),
                ..Default::default()
            }),
            model: Some("gpt-4o".into()),
            id: Some("resp_1".into()),
            metadata: ResponseMetadata::new(),
        };
        response.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });

        let record = ChatResponseRecord::from(&response);
        assert_eq!(record.metadata["demo"]["request_id"], "req_123");

        let rebuilt = record.into_chat_response_lossy();
        assert_eq!(
            rebuilt.metadata.get_portable("demo"),
            Some(&serde_json::json!({"request_id": "req_123"}))
        );
        assert_eq!(rebuilt.id.as_deref(), Some("resp_1"));
    }

    #[test]
    fn response_record_try_from_preserves_metadata_json_for_logging() {
        let mut response = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: Some("gpt-4o".into()),
            id: Some("resp_2".into()),
            metadata: ResponseMetadata::new(),
        };
        response.metadata.insert(DemoMetadata {
            request_id: "req_456".into(),
        });

        let record = ChatResponseRecord::try_from_response(&response).unwrap();
        assert_eq!(record.metadata["demo"]["request_id"], "req_456");
    }

    #[test]
    fn response_record_try_from_response_returns_error_for_unserializable_metadata() {
        let mut response = ChatResponse::default();
        response.metadata.insert(BrokenMetadata);

        let error = ChatResponseRecord::try_from_response(&response).unwrap_err();
        assert!(matches!(error, crate::Error::Serialization(_)));
    }

    #[test]
    fn response_record_from_is_lossy_for_unserializable_metadata() {
        let mut response = ChatResponse::default();
        response.metadata.insert(BrokenMetadata);
        response
            .metadata
            .insert_portable("portable", serde_json::json!(true));

        let record = ChatResponseRecord::from(&response);
        assert_eq!(
            record.metadata,
            serde_json::Map::from_iter([("portable".into(), serde_json::json!(true))])
        );
    }

    #[test]
    fn response_record_try_from_response_returns_error_for_metadata_key_collisions() {
        let mut response = ChatResponse::default();
        response.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        response
            .metadata
            .insert_portable("demo", serde_json::json!({"request_id": "portable"}));

        let error = ChatResponseRecord::try_from_response(&response).unwrap_err();
        assert!(matches!(error, crate::Error::Serialization(_)));
    }

    #[test]
    fn response_record_from_is_lossy_for_metadata_key_collisions() {
        let mut response = ChatResponse::default();
        response.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        response
            .metadata
            .insert_portable("demo", serde_json::json!({"request_id": "portable"}));

        let record = ChatResponseRecord::from(&response);
        assert_eq!(
            record.metadata,
            serde_json::Map::from_iter([(
                "demo".into(),
                serde_json::json!({"request_id": "portable"}),
            )])
        );
    }

    #[test]
    fn chat_response_debug_is_lossy_for_unserializable_metadata() {
        let mut response = ChatResponse::default();
        response.metadata.insert(BrokenMetadata);
        response
            .metadata
            .insert_portable("portable", serde_json::json!(true));

        let debug = format!("{response:?}");

        assert!(debug.contains("ChatResponse"));
        assert!(debug.contains("ResponseMetadata"));
        assert!(debug.contains("export_error"));
    }

    #[test]
    fn chat_response_debug_is_lossy_for_metadata_key_collisions() {
        let mut response = ChatResponse::default();
        response.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        response
            .metadata
            .insert_portable("demo", serde_json::json!({"request_id": "portable"}));

        let debug = format!("{response:?}");

        assert!(debug.contains("ChatResponse"));
        assert!(debug.contains("ResponseMetadata"));
        assert!(debug.contains("export_error"));
    }

    #[test]
    fn response_record_round_trip_preserves_portable_fields_but_not_typed_metadata() {
        let mut response = ChatResponse {
            content: vec![
                ContentBlock::Reasoning {
                    text: "thinking".into(),
                    signature: Some("sig_1".into()),
                },
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"rust"}"#.into(),
                },
                ContentBlock::Other {
                    type_name: "citation".into(),
                    data: serde_json::Map::from_iter([(
                        "url".into(),
                        serde_json::json!("https://example.com"),
                    )]),
                },
            ],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                total_tokens: Some(15),
                ..Default::default()
            }),
            model: Some("gpt-4o".into()),
            id: Some("resp_42".into()),
            metadata: ResponseMetadata::new(),
        };
        response.metadata.insert(DemoMetadata {
            request_id: "req_456".into(),
        });

        let record = ChatResponseRecord::from(&response);
        let rebuilt = record.clone().into_chat_response_lossy();

        assert_eq!(record.content, response.content);
        assert_eq!(record.finish_reason, response.finish_reason);
        assert_eq!(record.usage, response.usage);
        assert_eq!(record.model, response.model);
        assert_eq!(record.id, response.id);
        assert_eq!(record.metadata["demo"]["request_id"], "req_456");

        assert_eq!(rebuilt.content, response.content);
        assert_eq!(rebuilt.finish_reason, response.finish_reason);
        assert_eq!(rebuilt.usage, response.usage);
        assert_eq!(rebuilt.model, response.model);
        assert_eq!(rebuilt.id, response.id);
        assert_eq!(
            rebuilt.metadata.get_portable("demo"),
            Some(&serde_json::json!({"request_id": "req_456"}))
        );
        assert!(!rebuilt.metadata.contains::<DemoMetadata>());
    }

    #[test]
    fn response_record_from_owned_matches_lossy_helper() {
        let mut response = ChatResponse::default();
        response.metadata.insert(DemoMetadata {
            request_id: "req_789".into(),
        });

        let borrowed = ChatResponseRecord::from_response_lossy(&response);
        let owned = ChatResponseRecord::from(response);

        assert_eq!(borrowed, owned);
    }

    #[test]
    fn response_record_serde_skips_absent_optional_fields_and_empty_metadata() {
        let response = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
            finish_reason: None,
            usage: None,
            model: None,
            id: None,
            metadata: ResponseMetadata::new(),
        };

        let value = serde_json::to_value(ChatResponseRecord::from(&response)).unwrap();
        let obj = value.as_object().unwrap();
        assert!(obj.contains_key("content"));
        assert!(!obj.contains_key("finish_reason"));
        assert!(!obj.contains_key("usage"));
        assert!(!obj.contains_key("model"));
        assert!(!obj.contains_key("id"));
        assert!(!obj.contains_key("metadata"));
    }

    #[test]
    fn response_record_deserialize_defaults_missing_metadata_to_empty_map() {
        let record: ChatResponseRecord = serde_json::from_value(serde_json::json!({
            "content": [{ "type": "text", "text": "Hello" }]
        }))
        .unwrap();

        assert!(record.metadata.is_empty());

        let rebuilt = record.into_chat_response_lossy();
        assert!(rebuilt.metadata.to_portable_map().is_empty());
    }

    #[test]
    fn text_concatenates_multiple_text_blocks() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Hello ".into(),
                },
                ContentBlock::Text {
                    text: "world!".into(),
                },
            ],
            ..Default::default()
        };
        assert_eq!(resp.text(), Some("Hello world!".into()));
    }

    #[test]
    fn text_returns_none_when_no_text_blocks() {
        let resp = ChatResponse {
            content: vec![ContentBlock::ToolCall {
                id: "tc-1".into(),
                name: "search".into(),
                arguments: "{}".into(),
            }],
            ..Default::default()
        };
        assert_eq!(resp.text(), None);
    }

    #[test]
    fn text_returns_none_for_empty_content() {
        let resp = ChatResponse::default();
        assert_eq!(resp.text(), None);
    }

    #[test]
    fn text_or_empty_returns_empty_when_no_text() {
        let resp = ChatResponse::default();
        assert_eq!(resp.text_or_empty(), "");
    }

    #[test]
    fn text_or_empty_returns_text_when_present() {
        let resp = ChatResponse {
            content: vec![ContentBlock::Text { text: "hi".into() }],
            ..Default::default()
        };
        assert_eq!(resp.text_or_empty(), "hi");
    }

    #[test]
    fn first_reasoning_returns_first_reasoning_block() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Text {
                    text: "answer".into(),
                },
                ContentBlock::Reasoning {
                    text: "step 1".into(),
                    signature: None,
                },
                ContentBlock::Reasoning {
                    text: "step 2".into(),
                    signature: Some("sig".into()),
                },
            ],
            ..Default::default()
        };

        assert_eq!(resp.first_reasoning(), Some("step 1"));
    }

    #[test]
    fn first_text_returns_first_text_block() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Reasoning {
                    text: "thinking...".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "first".into(),
                },
                ContentBlock::Text {
                    text: "second".into(),
                },
            ],
            ..Default::default()
        };
        assert_eq!(resp.first_text(), Some("first"));
    }

    #[test]
    fn first_text_returns_none_when_no_text() {
        let resp = ChatResponse {
            content: vec![ContentBlock::Reasoning {
                text: "thinking".into(),
                signature: None,
            }],
            ..Default::default()
        };
        assert_eq!(resp.first_text(), None);
    }

    #[test]
    fn write_text_to_appends_all_text_segments() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Text {
                    text: "hello".into(),
                },
                ContentBlock::Reasoning {
                    text: "thinking".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: " world".into(),
                },
            ],
            ..Default::default()
        };

        let mut output = String::from("prefix:");
        assert!(resp.write_text_to(&mut output));
        assert_eq!(output, "prefix:hello world");
        assert_eq!(
            resp.text_segments().collect::<Vec<_>>(),
            vec!["hello", " world"]
        );
    }

    #[test]
    fn reasoning_text_concatenates_reasoning_blocks() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Reasoning {
                    text: "Step 1. ".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "Answer".into(),
                },
                ContentBlock::Reasoning {
                    text: "Step 2.".into(),
                    signature: Some("sig".into()),
                },
            ],
            ..Default::default()
        };
        assert_eq!(resp.reasoning_text(), Some("Step 1. Step 2.".into()));
    }

    #[test]
    fn reasoning_text_returns_none_when_no_reasoning() {
        let resp = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            ..Default::default()
        };
        assert_eq!(resp.reasoning_text(), None);
    }

    #[test]
    fn write_reasoning_text_to_appends_all_reasoning_segments() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Reasoning {
                    text: "step 1".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
                ContentBlock::Reasoning {
                    text: " step 2".into(),
                    signature: Some("sig".into()),
                },
            ],
            ..Default::default()
        };

        let mut output = String::new();
        assert!(resp.write_reasoning_text_to(&mut output));
        assert_eq!(output, "step 1 step 2");
        assert_eq!(
            resp.reasoning_segments().collect::<Vec<_>>(),
            vec!["step 1", " step 2"]
        );
    }

    #[test]
    fn tool_calls_returns_iterator_over_tool_call_blocks() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Let me search".into(),
                },
                ContentBlock::ToolCall {
                    id: "tc-1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"rust"}"#.into(),
                },
                ContentBlock::ToolCall {
                    id: "tc-2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"/tmp"}"#.into(),
                },
            ],
            ..Default::default()
        };

        let calls: Vec<_> = resp.tool_calls().collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "tc-1");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[1].id, "tc-2");
        assert_eq!(calls[1].name, "read");
    }

    #[test]
    fn first_tool_call_returns_first_tool_call_block() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Let me search".into(),
                },
                ContentBlock::ToolCall {
                    id: "tc-1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"rust"}"#.into(),
                },
                ContentBlock::ToolCall {
                    id: "tc-2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"/tmp"}"#.into(),
                },
            ],
            ..Default::default()
        };

        let call = resp.first_tool_call().unwrap();
        assert_eq!(call.id, "tc-1");
        assert_eq!(call.name, "search");
    }

    #[test]
    fn has_tool_calls_true_when_present() {
        let resp = ChatResponse {
            content: vec![ContentBlock::ToolCall {
                id: "tc-1".into(),
                name: "search".into(),
                arguments: "{}".into(),
            }],
            ..Default::default()
        };
        assert!(resp.has_tool_calls());
    }

    #[test]
    fn has_tool_calls_false_when_absent() {
        let resp = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            ..Default::default()
        };
        assert!(!resp.has_tool_calls());
    }

    #[test]
    fn to_assistant_message_preserves_all_blocks() {
        let resp = ChatResponse {
            content: vec![
                ContentBlock::Reasoning {
                    text: "thinking".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "hello".into(),
                },
                ContentBlock::ToolCall {
                    id: "tc-1".into(),
                    name: "search".into(),
                    arguments: "{}".into(),
                },
            ],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(Usage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                ..Default::default()
            }),
            model: Some("gpt-4o".into()),
            id: Some("chatcmpl-123".into()),
            metadata: ResponseMetadata::new(),
        };

        let msg = resp.to_assistant_message();
        match &msg {
            Message::Assistant {
                content,
                name,
                extensions,
            } => {
                assert_eq!(content.len(), 3);
                assert_eq!(content, &resp.content);
                assert!(name.is_none());
                assert!(extensions.is_none());
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn into_assistant_message_consumes_response() {
        let resp = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        };

        let msg = resp.into_assistant_message();
        match &msg {
            Message::Assistant { content, .. } => {
                assert_eq!(content.len(), 1);
                assert_eq!(
                    content[0],
                    ContentBlock::Text {
                        text: "hello".into()
                    }
                );
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn from_chat_response_for_message() {
        let resp = ChatResponse {
            content: vec![ContentBlock::Text { text: "hi".into() }],
            ..Default::default()
        };

        let msg: Message = resp.into();
        assert_eq!(msg.role(), "assistant");
        match &msg {
            Message::Assistant { content, .. } => {
                assert_eq!(content.len(), 1);
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn from_chat_response_ref_for_message() {
        let resp = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            ..Default::default()
        };

        let msg = Message::from(&resp);
        assert_eq!(msg.role(), "assistant");
        match &msg {
            Message::Assistant { content, .. } => {
                assert_eq!(content.len(), 1);
                assert_eq!(content, &resp.content);
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn metadata_serializes_into_log_output() {
        let mut resp = ChatResponse::default();
        resp.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });

        assert_eq!(
            serde_json::to_value(&resp.metadata).unwrap(),
            json!({
                "demo": {"request_id": "req_123"}
            })
        );
        assert_eq!(
            resp.to_log_value()["metadata"]["demo"]["request_id"],
            json!("req_123")
        );
    }

    #[test]
    fn response_record_helpers_preserve_portable_fields() {
        let mut resp = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "hello".into(),
            }],
            id: Some("resp_1".into()),
            metadata: ResponseMetadata::new(),
            ..Default::default()
        };
        resp.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });

        let borrowed = resp.to_record_lossy();
        let owned = resp.clone().into_record_lossy();
        let log = resp.try_to_log_value().unwrap();

        assert_eq!(borrowed, owned);
        assert_eq!(borrowed.id.as_deref(), Some("resp_1"));
        assert_eq!(log["metadata"]["demo"]["request_id"], json!("req_123"));
    }

    #[test]
    fn to_log_value_is_lossy_for_unserializable_typed_metadata() {
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

        impl crate::ResponseMetadataType for BrokenMetadata {
            const KEY: &'static str = "broken";
        }

        let mut resp = ChatResponse::default();
        resp.metadata.insert(BrokenMetadata);
        resp.metadata.insert_portable("portable", json!(true));

        assert!(matches!(
            resp.try_to_log_value(),
            Err(crate::Error::Serialization(_))
        ));
        assert_eq!(resp.to_log_value()["metadata"]["portable"], json!(true));
        assert!(resp.to_log_value()["metadata"].get("broken").is_none());
    }

    #[test]
    fn try_to_log_value_returns_error_for_metadata_key_collisions() {
        let mut resp = ChatResponse::default();
        resp.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        resp.metadata
            .insert_portable("demo", json!({"request_id": "portable"}));

        assert!(matches!(
            resp.try_to_log_value(),
            Err(crate::Error::Serialization(_))
        ));
    }

    #[test]
    fn to_log_value_is_lossy_for_metadata_key_collisions() {
        let mut resp = ChatResponse::default();
        resp.metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        resp.metadata
            .insert_portable("demo", json!({"request_id": "portable"}));

        assert_eq!(
            resp.to_log_value()["metadata"]["demo"],
            json!({"request_id": "portable"})
        );
    }
}
