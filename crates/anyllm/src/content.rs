use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::ser::{self, SerializeMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::utils::{
    insert_if_some, optional_string_field, required_string_field, set_once, truncate_display,
};
use crate::{Error, ExtraMap, ImageSource};

/// A single block of content returned by an LLM provider.
///
/// Uses custom serde to produce a flat `{"type": "...", ...}` JSON shape
/// rather than Rust's default externally-tagged enum encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContentBlock {
    /// Plain text output block
    Text {
        /// Text emitted by the provider.
        text: String,
    },
    /// Image output block
    Image {
        /// Image payload returned by the provider.
        source: ImageSource,
    },
    /// Tool call block emitted by the model
    ToolCall {
        /// Provider-issued tool call identifier.
        id: String,
        /// Tool name selected by the model.
        name: String,
        /// Raw JSON string containing the tool-call arguments.
        arguments: String,
    },
    /// Reasoning block emitted by the model
    Reasoning {
        /// Reasoning text emitted by the provider.
        text: String,
        /// Optional provider-specific signature for the reasoning block.
        signature: Option<String>,
    },
    /// Provider-specific content block not modeled explicitly
    Other {
        /// Provider-specific block type name.
        type_name: String,
        /// Provider-specific block payload fields.
        data: ExtraMap,
    },
}

impl ContentBlock {
    /// Borrow the text content for a text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Borrow the image content for an image block without cloning it.
    pub fn as_image(&self) -> Option<ImageBlockRef<'_>> {
        match self {
            ContentBlock::Image { source } => Some(ImageBlockRef { source }),
            _ => None,
        }
    }

    /// Borrow the tool call payload for a tool-call block without cloning it.
    pub fn as_tool_call(&self) -> Option<ToolCallRef<'_>> {
        match self {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some(ToolCallRef {
                id,
                name,
                arguments,
            }),
            _ => None,
        }
    }

    /// Borrow the reasoning text for a reasoning block.
    pub fn as_reasoning(&self) -> Option<&str> {
        match self {
            ContentBlock::Reasoning { text, .. } => Some(text),
            _ => None,
        }
    }
}

impl std::fmt::Display for ContentBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContentBlock::Text { text } => f.write_str(text),
            ContentBlock::Image { source } => match source {
                ImageSource::Url { url } => {
                    let truncated = truncate_display(url, 50);
                    write!(f, "image:url({truncated})")
                }
                ImageSource::Base64 { media_type, .. } => write!(f, "image:base64({media_type})"),
            },
            ContentBlock::ToolCall {
                name, arguments, ..
            } => {
                let truncated = truncate_display(arguments, 50);
                write!(f, "tool_call:{name}({truncated})")
            }
            ContentBlock::Reasoning { text, .. } => {
                let truncated = truncate_display(text, 50);
                write!(f, "[reasoning] {truncated}")
            }
            ContentBlock::Other { type_name, .. } => write!(f, "other:{type_name}"),
        }
    }
}

impl Serialize for ContentBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            ContentBlock::Text { text } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
                map.end()
            }
            ContentBlock::Image { source } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "image")?;
                map.serialize_entry("source", source)?;
                map.end()
            }
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "tool_call")?;
                map.serialize_entry("id", id)?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("arguments", arguments)?;
                map.end()
            }
            ContentBlock::Reasoning { text, signature } => {
                let count = if signature.is_some() { 3 } else { 2 };
                let mut map = serializer.serialize_map(Some(count))?;
                map.serialize_entry("type", "reasoning")?;
                map.serialize_entry("text", text)?;
                if let Some(sig) = signature {
                    map.serialize_entry("signature", sig)?;
                }
                map.end()
            }
            ContentBlock::Other { type_name, data } => {
                if data.contains_key("type") {
                    return Err(ser::Error::custom(
                        "ContentBlock::Other data must not contain reserved key \"type\"",
                    ));
                }

                let mut map = serializer.serialize_map(Some(1 + data.len()))?;
                map.serialize_entry("type", type_name)?;
                for (k, v) in data {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_map(ContentBlockVisitor)
    }
}

struct ContentBlockVisitor;

impl<'de> Visitor<'de> for ContentBlockVisitor {
    type Value = ContentBlock;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a map with a \"type\" field")
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut fields = ContentBlockFields::default();
        while let Some(key) = map.next_key::<String>()? {
            let (slot, field_name) = match key.as_str() {
                "type" => (&mut fields.type_name, "type"),
                "text" => (&mut fields.text, "text"),
                "source" => (&mut fields.source, "source"),
                "id" => (&mut fields.id, "id"),
                "name" => (&mut fields.name, "name"),
                "arguments" => (&mut fields.arguments, "arguments"),
                "signature" => (&mut fields.signature, "signature"),
                _ => {
                    fields.other.insert(key, map.next_value()?);
                    continue;
                }
            };

            set_once(slot, field_name, map.next_value()?)?;
        }

        let type_name = required_string_field(fields.type_name.take(), "type")?;

        match type_name.as_str() {
            "text" => {
                let text = required_string_field(fields.text.take(), "text")?;
                Ok(ContentBlock::Text { text })
            }
            "image" => {
                let source = fields
                    .source
                    .take()
                    .ok_or_else(|| serde::de::Error::custom("missing \"source\" field"))?;
                let source = serde_json::from_value(source).map_err(serde::de::Error::custom)?;
                Ok(ContentBlock::Image { source })
            }
            "tool_call" => {
                let id = required_string_field(fields.id.take(), "id")?;
                let name = required_string_field(fields.name.take(), "name")?;
                let arguments = required_string_field(fields.arguments.take(), "arguments")?;
                Ok(ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                })
            }
            "reasoning" => {
                let text = required_string_field(fields.text.take(), "text")?;
                let signature = optional_string_field(fields.signature.take(), "signature")?;
                Ok(ContentBlock::Reasoning { text, signature })
            }
            _ => Ok(fields.into_other(type_name)),
        }
    }
}

#[derive(Default)]
struct ContentBlockFields {
    type_name: Option<serde_json::Value>,
    text: Option<serde_json::Value>,
    source: Option<serde_json::Value>,
    id: Option<serde_json::Value>,
    name: Option<serde_json::Value>,
    arguments: Option<serde_json::Value>,
    signature: Option<serde_json::Value>,
    other: ExtraMap,
}

impl ContentBlockFields {
    fn into_other(mut self, type_name: String) -> ContentBlock {
        insert_if_some(&mut self.other, "text", self.text);
        insert_if_some(&mut self.other, "source", self.source);
        insert_if_some(&mut self.other, "id", self.id);
        insert_if_some(&mut self.other, "name", self.name);
        insert_if_some(&mut self.other, "arguments", self.arguments);
        insert_if_some(&mut self.other, "signature", self.signature);

        ContentBlock::Other {
            type_name,
            data: self.other,
        }
    }
}

/// Borrowed view returned by [`ContentBlock::as_tool_call`].
///
/// This is the zero-copy form used by helpers such as
/// [`ChatResponse::tool_calls`](crate::ChatResponse::tool_calls). Call
/// [`ToolCallRef::parse_arguments`] to deserialize the raw JSON arguments into
/// a typed payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolCallRef<'a> {
    /// Provider-issued tool call identifier.
    pub id: &'a str,
    /// Tool name selected by the model.
    pub name: &'a str,
    /// Raw JSON string containing the tool-call arguments.
    pub arguments: &'a str,
}

/// Borrowed view returned by [`ContentBlock::as_image`].
///
/// Use this when inspecting image output blocks without cloning the underlying
/// [`ImageSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageBlockRef<'a> {
    /// Borrowed image payload for this output block.
    pub source: &'a ImageSource,
}

impl ToolCallRef<'_> {
    /// Deserialize the raw JSON arguments into the requested type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`](crate::Error::Serialization) if the
    /// arguments string is not valid JSON or cannot be deserialized into `T`.
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> crate::Result<T> {
        serde_json::from_str(self.arguments).map_err(Error::from)
    }

    #[must_use]
    /// Convert this borrowed tool call view into an owned value.
    pub fn into_owned(self) -> OwnedToolCall {
        self.into()
    }
}

/// Owned representation of a tool call, independent of any `ContentBlock`.
///
/// This is useful when a borrowed [`ToolCallRef`] needs to outlive the source
/// response or be stored for later processing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnedToolCall {
    /// Provider-issued tool call identifier.
    pub id: String,
    /// Tool name selected by the model.
    pub name: String,
    /// Raw JSON string containing the tool-call arguments.
    pub arguments: String,
}

impl OwnedToolCall {
    /// Deserialize the raw JSON arguments into the requested type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`](crate::Error::Serialization) if the
    /// arguments string is not valid JSON or cannot be deserialized into `T`.
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> crate::Result<T> {
        serde_json::from_str(&self.arguments).map_err(Error::from)
    }
}

impl From<ToolCallRef<'_>> for OwnedToolCall {
    fn from(tool_call: ToolCallRef<'_>) -> Self {
        Self {
            id: tool_call.id.to_string(),
            name: tool_call.name.to_string(),
            arguments: tool_call.arguments.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_call_serde_round_trip() {
        let block = ContentBlock::ToolCall {
            id: "call_123".into(),
            name: "search".into(),
            arguments: r#"{"query":"rust"}"#.into(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            json!({
                "type": "tool_call",
                "id": "call_123",
                "name": "search",
                "arguments": r#"{"query":"rust"}"#,
            })
        );
        // arguments stays as a string, not parsed
        assert!(json["arguments"].is_string());
        let deserialized: ContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn image_serde_round_trip() {
        let block = ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.com/cat.png".into(),
            },
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            json!({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/cat.png"
                }
            })
        );
        let deserialized: ContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn image_accessor_exposes_source() {
        let block = ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "iVBORw0KGgoAAAANSUhEUg==".into(),
            },
        };

        let image = block.as_image().unwrap();
        assert!(matches!(
            image.source,
            ImageSource::Base64 { media_type, .. } if media_type == "image/png"
        ));
    }

    #[test]
    fn display_tool_call_short_args() {
        let block = ContentBlock::ToolCall {
            id: "id".into(),
            name: "search".into(),
            arguments: r#"{"q":"hi"}"#.into(),
        };
        assert_eq!(block.to_string(), r#"tool_call:search({"q":"hi"})"#);
    }

    #[test]
    fn display_tool_call_long_args_truncated() {
        let long_args = "a".repeat(100);
        let block = ContentBlock::ToolCall {
            id: "id".into(),
            name: "search".into(),
            arguments: long_args,
        };
        let displayed = block.to_string();
        assert!(displayed.starts_with("tool_call:search("));
        assert!(displayed.ends_with("...)"));
        // 50 chars of 'a' + "..." = 53 chars inside parens
        assert!(displayed.contains(&"a".repeat(50)));
    }

    #[test]
    fn display_reasoning_short() {
        let block = ContentBlock::Reasoning {
            text: "thinking".into(),
            signature: None,
        };
        assert_eq!(block.to_string(), "[reasoning] thinking");
    }

    #[test]
    fn display_reasoning_long_truncated() {
        let long_text = "b".repeat(100);
        let block = ContentBlock::Reasoning {
            text: long_text,
            signature: None,
        };
        let displayed = block.to_string();
        assert!(displayed.starts_with("[reasoning] "));
        assert!(displayed.ends_with("..."));
        assert!(displayed.contains(&"b".repeat(50)));
    }

    #[test]
    fn tool_call_ref_parse_arguments_valid() {
        let tcr = ToolCallRef {
            id: "id",
            name: "test",
            arguments: r#"{"key":"value"}"#,
        };
        let parsed: serde_json::Value = tcr.parse_arguments().unwrap();
        assert_eq!(parsed, json!({"key": "value"}));
    }

    #[test]
    fn tool_call_ref_parse_arguments_invalid() {
        let tcr = ToolCallRef {
            id: "id",
            name: "test",
            arguments: "not json at all",
        };
        let result: crate::Result<serde_json::Value> = tcr.parse_arguments();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::Serialization(_)),
            "expected Serialization error, got {err:?}"
        );
    }

    #[test]
    fn owned_tool_call_parse_arguments() {
        let otc = OwnedToolCall {
            id: "id".into(),
            name: "test".into(),
            arguments: r#"{"count":42}"#.into(),
        };
        let parsed: serde_json::Value = otc.parse_arguments().unwrap();
        assert_eq!(parsed, json!({"count": 42}));
    }

    #[test]
    fn deserialize_rejects_duplicate_known_field() {
        let err = serde_json::from_str::<ContentBlock>(
            r#"{"type":"text","text":"first","text":"second"}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate field `text`"));
    }

    #[test]
    fn deserialize_rejects_duplicate_type_field() {
        let err = serde_json::from_str::<ContentBlock>(r#"{"type":"text","type":"reasoning"}"#)
            .unwrap_err();
        assert!(err.to_string().contains("duplicate field `type`"));
    }

    #[test]
    fn deserialize_rejects_non_string_known_field() {
        let err =
            serde_json::from_str::<ContentBlock>(r#"{"type":"text","text":123}"#).unwrap_err();
        assert!(err.to_string().contains("\"text\" must be a string"));
    }

    #[test]
    fn unknown_block_round_trips_known_and_unknown_fields() {
        let value = json!({
            "type": "vendor_blob",
            "text": "opaque text",
            "signature": "sig",
            "extra": {"nested": true}
        });

        let block: ContentBlock = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            block,
            ContentBlock::Other {
                type_name: "vendor_blob".into(),
                data: serde_json::from_value(json!({
                    "text": "opaque text",
                    "signature": "sig",
                    "extra": {"nested": true}
                }))
                .unwrap(),
            }
        );

        let round_tripped = serde_json::to_value(&block).unwrap();
        assert_eq!(round_tripped, value);
    }

    #[test]
    fn serializing_other_rejects_reserved_type_key_in_data() {
        let block = ContentBlock::Other {
            type_name: "vendor_blob".into(),
            data: serde_json::from_value(json!({
                "type": "nested",
                "extra": true
            }))
            .unwrap(),
        };

        let err = serde_json::to_value(&block).unwrap_err();
        assert!(err.to_string().contains("reserved key \"type\""));
    }

    #[test]
    fn display_truncation_preserves_char_boundaries() {
        let block = ContentBlock::Reasoning {
            text: "é".repeat(30),
            signature: None,
        };

        let displayed = block.to_string();
        assert!(displayed.ends_with("..."));
        assert!(std::str::from_utf8(displayed.as_bytes()).is_ok());
    }
}
