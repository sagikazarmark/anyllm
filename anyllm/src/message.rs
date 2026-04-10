use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;

use crate::utils::{
    insert_if_some, optional_bool_field, optional_string_field, required_string_field,
    required_string_field_in, set_once, write_truncated, write_truncated_joined,
};
use crate::{ContentBlock, ExtraMap};

const DISPLAY_MAX_BYTES: usize = 50;

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
#[non_exhaustive]
pub enum Message {
    /// System instruction message
    System {
        /// System prompt text.
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Provider-specific JSON extensions.
        extensions: Option<ExtraMap>,
    },
    /// User input message
    User {
        /// User text or multimodal content.
        content: UserContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Optional participant name.
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Provider-specific JSON extensions.
        extensions: Option<ExtraMap>,
    },
    /// Assistant output message
    Assistant {
        /// Assistant content blocks.
        content: Vec<ContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Optional participant name.
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Provider-specific JSON extensions.
        extensions: Option<ExtraMap>,
    },
    /// Tool result message
    Tool {
        /// Tool call identifier this result corresponds to.
        tool_call_id: String,
        /// The name of the tool that produced this result. Providers like Gemini
        /// use name-based correlation instead of ID-based, so this avoids
        /// expensive backwards scans through message history.
        name: String,
        /// Tool output payload.
        content: ToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Whether the tool output should be treated as an error result.
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Provider-specific JSON extensions.
        extensions: Option<ExtraMap>,
    },
}

impl Message {
    /// Create a system message from plain text.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: content.into(),
            extensions: None,
        }
    }

    /// Create a user message from plain text or multimodal parts.
    ///
    /// # Examples
    ///
    /// Plain text:
    ///
    /// ```rust
    /// use anyllm::Message;
    ///
    /// let message = Message::user("Explain ownership in one paragraph.");
    /// assert_eq!(message.role(), "user");
    /// ```
    ///
    /// Multimodal parts:
    ///
    /// ```rust
    /// use anyllm::{ContentPart, Message};
    ///
    /// let message = Message::user(vec![
    ///     ContentPart::text("Describe this image."),
    ///     ContentPart::image_url("https://example.com/cat.png"),
    /// ]);
    /// assert_eq!(message.role(), "user");
    /// ```
    #[must_use]
    pub fn user(content: impl Into<UserContent>) -> Self {
        Message::User {
            content: content.into(),
            name: None,
            extensions: None,
        }
    }

    /// Back-compat convenience constructor for multimodal user content.
    #[must_use]
    pub fn user_multimodal(parts: Vec<ContentPart>) -> Self {
        Self::user(parts)
    }

    /// Create an assistant message containing a single text block.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
            name: None,
            extensions: None,
        }
    }

    /// Create a successful tool result message.
    #[must_use]
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<ToolResultContent>,
    ) -> Self {
        Message::Tool {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: content.into(),
            is_error: None,
            extensions: None,
        }
    }

    /// Create a tool result message marked as an error.
    #[must_use]
    pub fn tool_error(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        error: impl Into<ToolResultContent>,
    ) -> Self {
        Message::Tool {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: error.into(),
            is_error: Some(true),
            extensions: None,
        }
    }

    /// Add or replace a provider-specific JSON extension field.
    #[must_use]
    pub fn with_extension(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let extensions = match &mut self {
            Message::System { extensions, .. } => extensions,
            Message::User { extensions, .. } => extensions,
            Message::Assistant { extensions, .. } => extensions,
            Message::Tool { extensions, .. } => extensions,
        };
        extensions
            .get_or_insert_with(ExtraMap::new)
            .insert(key.into(), value);
        self
    }

    /// Return the portable role string for this message.
    #[must_use]
    pub fn role(&self) -> &'static str {
        match self {
            Message::System { .. } => "system",
            Message::User { .. } => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tool { .. } => "tool",
        }
    }

    /// Borrow the system prompt text if this is a system message.
    #[must_use]
    pub fn as_system(&self) -> Option<&str> {
        match self {
            Message::System { content, .. } => Some(content),
            _ => None,
        }
    }

    #[must_use]
    /// Borrow a user message without cloning its content.
    pub fn as_user(&self) -> Option<UserMessageRef<'_>> {
        match self {
            Message::User {
                content,
                name,
                extensions,
            } => Some(UserMessageRef {
                content,
                name: name.as_deref(),
                extensions: extensions.as_ref(),
            }),
            _ => None,
        }
    }

    #[must_use]
    /// Borrow an assistant message without cloning its content blocks.
    pub fn as_assistant(&self) -> Option<AssistantMessageRef<'_>> {
        match self {
            Message::Assistant {
                content,
                name,
                extensions,
            } => Some(AssistantMessageRef {
                content,
                name: name.as_deref(),
                extensions: extensions.as_ref(),
            }),
            _ => None,
        }
    }

    #[must_use]
    /// Borrow a tool-result message without cloning its content.
    pub fn as_tool(&self) -> Option<ToolMessageRef<'_>> {
        match self {
            Message::Tool {
                tool_call_id,
                name,
                content,
                is_error,
                extensions,
            } => Some(ToolMessageRef {
                tool_call_id,
                name,
                content,
                is_error: *is_error,
                extensions: extensions.as_ref(),
            }),
            _ => None,
        }
    }

    /// Borrow the extensions map, if present.
    #[must_use]
    pub fn extensions(&self) -> Option<&ExtraMap> {
        match self {
            Message::System { extensions, .. }
            | Message::User { extensions, .. }
            | Message::Assistant { extensions, .. }
            | Message::Tool { extensions, .. } => extensions.as_ref(),
        }
    }
}

/// Borrowed view returned by [`Message::as_user`].
///
/// Use this when you want to inspect a user message's multimodal content,
/// optional participant name, or extensions without cloning the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserMessageRef<'a> {
    /// Borrowed user content.
    pub content: &'a UserContent,
    /// Optional participant name.
    pub name: Option<&'a str>,
    /// Borrowed provider-specific extensions.
    pub extensions: Option<&'a ExtraMap>,
}

/// Borrowed view returned by [`Message::as_assistant`].
///
/// This exposes the assistant content blocks, optional name, and extensions in
/// a zero-copy form for routing, logging, or replay logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssistantMessageRef<'a> {
    /// Borrowed assistant content blocks.
    pub content: &'a [ContentBlock],
    /// Optional participant name.
    pub name: Option<&'a str>,
    /// Borrowed provider-specific extensions.
    pub extensions: Option<&'a ExtraMap>,
}

/// Borrowed view returned by [`Message::as_tool`].
///
/// Use this to inspect a tool result message, including whether the tool output
/// was marked as an error, without cloning the message payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolMessageRef<'a> {
    /// Borrowed tool call identifier.
    pub tool_call_id: &'a str,
    /// Borrowed tool name.
    pub name: &'a str,
    /// Borrowed tool output payload.
    pub content: &'a ToolResultContent,
    /// Whether the payload was marked as an error.
    pub is_error: Option<bool>,
    /// Borrowed provider-specific extensions.
    pub extensions: Option<&'a ExtraMap>,
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Message::System { content, .. } => {
                f.write_str("system: ")?;
                write_truncated(f, content, DISPLAY_MAX_BYTES)
            }
            Message::User { content, .. } => {
                f.write_str("user: ")?;
                match content {
                    UserContent::Text(s) => write_truncated(f, s, DISPLAY_MAX_BYTES),
                    UserContent::Parts(parts) => write_truncated_joined(
                        f,
                        parts.iter().filter_map(|part| part.as_text()),
                        "[multimodal]",
                        DISPLAY_MAX_BYTES,
                    ),
                }
            }
            Message::Assistant { content, .. } => {
                f.write_str("assistant: ")?;
                write_truncated_joined(
                    f,
                    content.iter().filter_map(ContentBlock::as_text),
                    "[non-text content]",
                    DISPLAY_MAX_BYTES,
                )
            }
            Message::Tool { content, .. } => {
                f.write_str("tool: ")?;
                match content {
                    ToolResultContent::Text(text) => write_truncated(f, text, DISPLAY_MAX_BYTES),
                    ToolResultContent::Parts(parts) => write_truncated_joined(
                        f,
                        parts.iter().filter_map(|part| part.as_text()),
                        "[multimodal]",
                        DISPLAY_MAX_BYTES,
                    ),
                }
            }
        }
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(MessageVisitor)
    }
}

struct MessageVisitor;

impl<'de> Visitor<'de> for MessageVisitor {
    type Value = Message;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a message object with a \"role\" field")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Message, A::Error> {
        let mut fields = MessageFields::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "role" => set_once(&mut fields.role, "role", map.next_value()?)?,
                "content" => set_once(&mut fields.content, "content", map.next_value()?)?,
                "name" => set_once(&mut fields.name, "name", map.next_value()?)?,
                "tool_call_id" => {
                    set_once(&mut fields.tool_call_id, "tool_call_id", map.next_value()?)?
                }
                "is_error" => set_once(&mut fields.is_error, "is_error", map.next_value()?)?,
                "extensions" => set_once(&mut fields.extensions, "extensions", map.next_value()?)?,
                "extra" => set_once(&mut fields.extra, "extra", map.next_value()?)?,
                _ => fields.insert_unknown_extension(key, map.next_value()?),
            }
        }

        let role = MessageRole::from_value::<A::Error>(fields.role.take())?;
        match role {
            MessageRole::System => Ok(Message::System {
                content: required_string_field_in::<A::Error>(
                    fields.content.take().as_deref().map(raw_to_value),
                    "content",
                    "system message",
                )?,
                extensions: fields.take_extensions(),
            }),
            MessageRole::User => Ok(Message::User {
                content: deserialize_user_content(
                    fields
                        .content
                        .take()
                        .ok_or_else(|| {
                            de::Error::custom("missing \"content\" field for user message")
                        })?
                        .as_ref(),
                )?,
                name: optional_string_field(fields.name.take(), "name")?,
                extensions: fields.take_extensions(),
            }),
            MessageRole::Assistant => Ok(Message::Assistant {
                content: deserialize_assistant_content(
                    fields
                        .content
                        .take()
                        .ok_or_else(|| {
                            de::Error::custom("missing \"content\" field for assistant message")
                        })?
                        .as_ref(),
                )?,
                name: optional_string_field(fields.name.take(), "name")?,
                extensions: fields.take_extensions(),
            }),
            MessageRole::Tool => Ok(Message::Tool {
                tool_call_id: required_string_field_in::<A::Error>(
                    fields.tool_call_id.take(),
                    "tool_call_id",
                    "tool message",
                )?,
                name: required_string_field_in::<A::Error>(
                    fields.name.take(),
                    "name",
                    "tool message",
                )?,
                content: deserialize_tool_result_content(
                    fields
                        .content
                        .take()
                        .ok_or_else(|| {
                            de::Error::custom("missing \"content\" field for tool message")
                        })?
                        .as_ref(),
                )?,
                is_error: optional_bool_field(fields.is_error.take(), "is_error")?,
                extensions: fields.take_extensions(),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    fn from_value<E: de::Error>(value: Option<serde_json::Value>) -> Result<Self, E> {
        match required_string_field(value, "role")?.as_str() {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            other => Err(de::Error::custom(format!("unknown role: {other}"))),
        }
    }
}

#[derive(Default)]
struct MessageFields {
    role: Option<serde_json::Value>,
    content: Option<Box<RawValue>>,
    name: Option<serde_json::Value>,
    tool_call_id: Option<serde_json::Value>,
    is_error: Option<serde_json::Value>,
    extensions: Option<serde_json::Value>,
    extra: Option<serde_json::Value>,
    implicit_extensions: Option<ExtraMap>,
}

impl MessageFields {
    fn insert_unknown_extension(&mut self, key: String, value: serde_json::Value) {
        self.implicit_extensions
            .get_or_insert_with(ExtraMap::new)
            .insert(key, value);
    }

    fn take_extensions(&mut self) -> Option<ExtraMap> {
        resolve_extensions(
            self.extensions.take(),
            self.extra.take(),
            self.implicit_extensions.take(),
        )
    }
}

fn raw_to_value(raw: &RawValue) -> serde_json::Value {
    serde_json::from_str(raw.get()).expect("raw JSON value should remain valid")
}

fn deserialize_user_content<E: de::Error>(raw: &RawValue) -> Result<UserContent, E> {
    match first_non_whitespace_byte(raw.get()) {
        Some(b'"') => serde_json::from_str::<String>(raw.get())
            .map(UserContent::Text)
            .map_err(de::Error::custom),
        Some(b'[') => serde_json::from_str::<Vec<ContentPart>>(raw.get())
            .map(UserContent::Parts)
            .map_err(de::Error::custom),
        _ => Err(de::Error::custom(
            "\"content\" must be a string or array for user message",
        )),
    }
}

fn deserialize_assistant_content<E: de::Error>(raw: &RawValue) -> Result<Vec<ContentBlock>, E> {
    match first_non_whitespace_byte(raw.get()) {
        Some(b'[') => serde_json::from_str(raw.get()).map_err(de::Error::custom),
        _ => Err(de::Error::custom(
            "\"content\" must be an array for assistant message",
        )),
    }
}

fn deserialize_tool_result_content<E: de::Error>(raw: &RawValue) -> Result<ToolResultContent, E> {
    match first_non_whitespace_byte(raw.get()) {
        Some(b'"') => serde_json::from_str::<String>(raw.get())
            .map(ToolResultContent::Text)
            .map_err(de::Error::custom),
        Some(b'[') => serde_json::from_str::<Vec<ContentPart>>(raw.get())
            .map(ToolResultContent::Parts)
            .map_err(de::Error::custom),
        _ => Err(de::Error::custom(
            "\"content\" must be a string or array for tool message",
        )),
    }
}

fn first_non_whitespace_byte(s: &str) -> Option<u8> {
    s.as_bytes()
        .iter()
        .copied()
        .find(|b| !b.is_ascii_whitespace())
}

fn resolve_extensions(
    extensions: Option<serde_json::Value>,
    extra: Option<serde_json::Value>,
    implicit: Option<ExtraMap>,
) -> Option<ExtraMap> {
    for value in [extensions, extra] {
        if let Some(serde_json::Value::Object(obj)) = value {
            if !obj.is_empty() {
                return Some(obj);
            }
            return None;
        }
    }

    implicit.filter(|fields| !fields.is_empty())
}

/// Content of a user message: either plain text or multimodal parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// Plain text user input.
    Text(String),
    /// Multimodal user input made of ordered parts.
    Parts(Vec<ContentPart>),
}

impl From<String> for UserContent {
    fn from(s: String) -> Self {
        UserContent::Text(s)
    }
}

impl From<&str> for UserContent {
    fn from(s: &str) -> Self {
        UserContent::Text(s.to_owned())
    }
}

impl From<Vec<ContentPart>> for UserContent {
    fn from(parts: Vec<ContentPart>) -> Self {
        UserContent::Parts(parts)
    }
}

/// Content of a tool result message: either plain text or multimodal parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Plain text tool output.
    Text(String),
    /// Multimodal tool output made of ordered parts.
    Parts(Vec<ContentPart>),
}

impl ToolResultContent {
    #[must_use]
    /// Create a text tool-result payload.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    #[must_use]
    /// Borrow the payload as plain text when it is text-backed.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(_) => None,
        }
    }

    #[must_use]
    /// Borrow the payload as multimodal parts when it is part-backed.
    pub fn as_parts(&self) -> Option<&[ContentPart]> {
        match self {
            Self::Text(_) => None,
            Self::Parts(parts) => Some(parts),
        }
    }
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for ToolResultContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<&String> for ToolResultContent {
    fn from(s: &String) -> Self {
        Self::Text(s.clone())
    }
}

impl From<Vec<ContentPart>> for ToolResultContent {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
}

/// A single part within multimodal user content.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContentPart {
    /// Plain text user content part
    Text {
        /// Text content for this part.
        text: String,
    },
    /// Image user content part
    Image {
        /// Image source for this part.
        source: ImageSource,
        /// Optional provider-specific detail hint such as `low` or `high`.
        detail: Option<String>,
    },
    /// Provider-specific user content part not modeled explicitly
    Other {
        /// Provider-specific part type name.
        type_name: String,
        /// Provider-specific part payload fields.
        data: ExtraMap,
    },
}

impl ContentPart {
    #[must_use]
    /// Create a text content part.
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    #[must_use]
    /// Create an image content part referencing a remote URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::Image {
            source: ImageSource::Url { url: url.into() },
            detail: None,
        }
    }

    #[must_use]
    /// Create an image content part from inline base64-encoded bytes.
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        ContentPart::Image {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
            detail: None,
        }
    }

    #[must_use]
    /// Borrow the text content if this is a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentPart::Text { text } => Some(text),
            _ => None,
        }
    }

    #[must_use]
    /// Borrow the image data for an image part without cloning it.
    pub fn as_image(&self) -> Option<ImagePartRef<'_>> {
        match self {
            ContentPart::Image { source, detail } => Some(ImagePartRef {
                source,
                detail: detail.as_deref(),
            }),
            _ => None,
        }
    }
}

/// Borrowed view returned by [`ContentPart::as_image`].
///
/// This keeps image-source inspection zero-copy while preserving the optional
/// provider hint carried in `detail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagePartRef<'a> {
    /// Borrowed image source.
    pub source: &'a ImageSource,
    /// Optional provider-specific detail hint.
    pub detail: Option<&'a str>,
}

impl Serialize for ContentPart {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            ContentPart::Text { text } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
                map.end()
            }
            ContentPart::Image { source, detail } => {
                let len = 2 + usize::from(detail.is_some());
                let mut map = serializer.serialize_map(Some(len))?;
                map.serialize_entry("type", "image")?;
                map.serialize_entry("source", source)?;
                if let Some(d) = detail {
                    map.serialize_entry("detail", d)?;
                }
                map.end()
            }
            ContentPart::Other { type_name, data } => {
                if data.contains_key("type") {
                    return Err(serde::ser::Error::custom(
                        "ContentPart::Other data must not contain reserved key \"type\"",
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

impl<'de> Deserialize<'de> for ContentPart {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(ContentPartVisitor)
    }
}

struct ContentPartVisitor;

impl<'de> Visitor<'de> for ContentPartVisitor {
    type Value = ContentPart;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an object for ContentPart")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut fields = ContentPartFields::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "type" => set_once(&mut fields.type_name, "type", map.next_value()?)?,
                "text" => set_once(&mut fields.text, "text", map.next_value()?)?,
                "source" => set_once(&mut fields.source, "source", map.next_value()?)?,
                "detail" => set_once(&mut fields.detail, "detail", map.next_value()?)?,
                _ => {
                    fields.other.insert(key, map.next_value()?);
                }
            }
        }

        let type_name = required_string_field(fields.type_name.take(), "type")?;

        match type_name.as_str() {
            "text" => fields.into_text::<A::Error>(),
            "image" => fields.into_image::<A::Error>(),
            _ => Ok(fields.into_other(type_name)),
        }
    }
}

#[derive(Default)]
struct ContentPartFields {
    type_name: Option<serde_json::Value>,
    text: Option<serde_json::Value>,
    source: Option<Box<RawValue>>,
    detail: Option<serde_json::Value>,
    other: ExtraMap,
}

impl ContentPartFields {
    fn into_text<E: de::Error>(mut self) -> Result<ContentPart, E> {
        Ok(ContentPart::Text {
            text: required_string_field_in(self.text.take(), "text", "text part")?,
        })
    }

    fn into_image<E: de::Error>(self) -> Result<ContentPart, E> {
        let source = self
            .source
            .ok_or_else(|| de::Error::custom("missing \"source\" field for image part"))?;
        Ok(ContentPart::Image {
            source: serde_json::from_str(source.get()).map_err(de::Error::custom)?,
            detail: optional_string_field(self.detail, "detail")?,
        })
    }

    fn into_other(mut self, type_name: String) -> ContentPart {
        insert_if_some(&mut self.other, "text", self.text);
        insert_if_some(
            &mut self.other,
            "source",
            self.source.as_deref().map(raw_to_value),
        );
        insert_if_some(&mut self.other, "detail", self.detail);

        ContentPart::Other {
            type_name,
            data: self.other,
        }
    }
}

/// Source for an image in a content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageSource {
    /// Remote image URL
    Url {
        /// Absolute image URL.
        url: String,
    },
    /// Inline base64-encoded image data
    Base64 {
        /// MIME type for the encoded image bytes.
        media_type: String,
        /// Base64-encoded image bytes.
        data: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContentBlock;
    use serde_json::json;

    #[test]
    fn message_accessors_expose_borrowed_views() {
        let system = Message::system("sys").with_extension("trace", json!(1));
        assert_eq!(system.as_system(), Some("sys"));
        assert_eq!(system.extensions().unwrap().get("trace"), Some(&json!(1)));

        let user = Message::User {
            content: UserContent::Parts(vec![ContentPart::text("hello")]),
            name: Some("alice".into()),
            extensions: Some(ExtraMap::from_iter([("trace_id".into(), json!(123))])),
        };
        let user_ref = user.as_user().unwrap();
        assert_eq!(user_ref.name, Some("alice"));
        assert!(matches!(user_ref.content, UserContent::Parts(_)));

        let assistant = Message::Assistant {
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            name: Some("assistant-1".into()),
            extensions: None,
        };
        let assistant_ref = assistant.as_assistant().unwrap();
        assert_eq!(assistant_ref.name, Some("assistant-1"));
        assert_eq!(assistant_ref.content[0].as_text(), Some("done"));

        let tool = Message::tool_error("call_1", "search", "failed");
        let tool_ref = tool.as_tool().unwrap();
        assert_eq!(tool_ref.tool_call_id, "call_1");
        assert_eq!(tool_ref.name, "search");
        assert_eq!(tool_ref.content.as_text(), Some("failed"));
        assert_eq!(tool_ref.is_error, Some(true));
    }

    #[test]
    fn message_with_extension_works_on_all_variants() {
        let cases = [
            Message::system("s").with_extension("k", json!("v")),
            Message::user("u").with_extension("k", json!("v")),
            Message::assistant("a").with_extension("k", json!("v")),
            Message::tool_result("id", "my_tool", "r").with_extension("k", json!("v")),
        ];

        for message in cases {
            assert_eq!(message.extensions().unwrap().get("k"), Some(&json!("v")));
        }
    }

    #[test]
    fn content_part_accessors_expose_borrowed_views() {
        let text = ContentPart::text("hello");
        assert_eq!(text.as_text(), Some("hello"));
        assert!(text.as_image().is_none());

        let image = ContentPart::Image {
            source: ImageSource::Url {
                url: "https://example.com/cat.png".into(),
            },
            detail: Some("high".into()),
        };
        let image_ref = image.as_image().unwrap();
        assert_eq!(image_ref.detail, Some("high"));
        assert!(
            matches!(image_ref.source, ImageSource::Url { url } if url == "https://example.com/cat.png")
        );
    }

    #[test]
    fn content_part_round_trips_known_and_unknown_variants() {
        let cases = [
            (
                json!({"type": "text", "text": "hello"}),
                ContentPart::Text {
                    text: "hello".into(),
                },
            ),
            (
                json!({
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/cat.png"},
                    "detail": "high"
                }),
                ContentPart::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/cat.png".into(),
                    },
                    detail: Some("high".into()),
                },
            ),
            (
                json!({"type": "audio", "voice": "alloy", "duration_ms": 1200}),
                ContentPart::Other {
                    type_name: "audio".into(),
                    data: serde_json::Map::from_iter([
                        ("voice".into(), json!("alloy")),
                        ("duration_ms".into(), json!(1200)),
                    ]),
                },
            ),
        ];

        for (value, expected) in cases {
            let part: ContentPart = serde_json::from_value(value).unwrap();
            assert_eq!(part, expected);
        }
    }

    #[test]
    fn content_part_rejects_invalid_shapes() {
        let cases = [
            (
                serde_json::from_value::<ContentPart>(json!("not-an-object")).unwrap_err(),
                "expected an object for ContentPart",
            ),
            (
                serde_json::from_value::<ContentPart>(json!({"text": "hello"})).unwrap_err(),
                "missing field `type`",
            ),
            (
                serde_json::from_value::<ContentPart>(json!({"type": "image", "detail": "high"}))
                    .unwrap_err(),
                "missing \"source\" field for image part",
            ),
        ];

        for (err, expected) in cases {
            assert!(err.to_string().contains(expected));
        }
    }

    #[test]
    fn content_part_deserialize_rejects_duplicate_known_field() {
        let err = serde_json::from_str::<ContentPart>(
            r#"{"type":"text","text":"first","text":"second"}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate field `text`"));
    }

    #[test]
    fn content_part_other_rejects_reserved_type_key_in_data() {
        let err = serde_json::to_value(ContentPart::Other {
            type_name: "audio".into(),
            data: serde_json::Map::from_iter([("type".into(), json!("nested"))]),
        })
        .unwrap_err();

        assert!(err.to_string().contains("reserved key \"type\""));
    }

    #[test]
    fn message_deserialize_preserves_extensions() {
        let json = json!({
            "role": "system",
            "content": "test",
            "cache_control": {"type": "ephemeral"}
        });
        let msg: Message = serde_json::from_value(json).unwrap();
        match msg {
            Message::System { extensions, .. } => {
                let extensions = extensions.unwrap();
                assert_eq!(
                    extensions.get("cache_control").unwrap(),
                    &json!({"type": "ephemeral"})
                );
            }
            other => panic!("expected System, got {other:?}"),
        }
    }

    #[test]
    fn message_deserialize_accepts_legacy_extra_field() {
        let json = json!({
            "role": "system",
            "content": "test",
            "extra": {"cache_control": {"type": "ephemeral"}}
        });
        let msg: Message = serde_json::from_value(json).unwrap();
        match msg {
            Message::System { extensions, .. } => {
                assert_eq!(
                    extensions.unwrap().get("cache_control"),
                    Some(&json!({"type": "ephemeral"}))
                );
            }
            other => panic!("expected System, got {other:?}"),
        }
    }

    #[test]
    fn message_deserialize_user_multimodal_preserves_name_and_extensions() {
        let msg: Message = serde_json::from_value(json!({
            "role": "user",
            "name": "alice",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
                    "detail": "low"
                },
                {"type": "input_audio", "format": "wav"}
            ],
            "trace_id": "trace-1"
        }))
        .unwrap();

        match msg {
            Message::User {
                content: UserContent::Parts(parts),
                name,
                extensions,
            } => {
                assert_eq!(name.as_deref(), Some("alice"));
                assert_eq!(parts.len(), 3);
                assert_eq!(parts[0].as_text(), Some("look"));
                assert!(matches!(
                    &parts[1],
                    ContentPart::Image {
                        source: ImageSource::Base64 { media_type, data },
                        detail
                    } if media_type == "image/png" && data == "abc123" && detail.as_deref() == Some("low")
                ));
                assert!(matches!(
                    &parts[2],
                    ContentPart::Other { type_name, data }
                        if type_name == "input_audio" && data.get("format") == Some(&json!("wav"))
                ));
                assert_eq!(extensions.unwrap().get("trace_id"), Some(&json!("trace-1")));
            }
            other => panic!("expected multimodal user message, got {other:?}"),
        }
    }

    #[test]
    fn message_deserialize_invalid_cases_report_consistent_errors() {
        let cases = [
            (
                serde_json::from_value::<Message>(json!({
                    "role": "assistant",
                    "content": "hello"
                }))
                .unwrap_err(),
                "\"content\" must be an array for assistant message",
            ),
            (
                serde_json::from_value::<Message>(json!({
                    "role": "user",
                    "content": {"text": "hello"}
                }))
                .unwrap_err(),
                "\"content\" must be a string or array for user message",
            ),
            (
                serde_json::from_value::<Message>(json!({
                    "role": "developer",
                    "content": "hi"
                }))
                .unwrap_err(),
                "unknown role: developer",
            ),
        ];

        for (err, expected) in cases {
            assert!(err.to_string().contains(expected));
        }
    }

    #[test]
    fn message_deserialize_rejects_duplicate_known_field() {
        let err = serde_json::from_str::<Message>(
            r#"{"role":"system","content":"first","content":"second"}"#,
        )
        .unwrap_err();

        assert!(err.to_string().contains("duplicate field `content`"));
    }

    #[test]
    fn message_deserialize_prefers_explicit_extensions_over_implicit_unknown_fields() {
        let msg: Message = serde_json::from_value(json!({
            "role": "system",
            "content": "test",
            "extensions": {"cache_control": {"type": "ephemeral"}},
            "trace_id": "ignored-when-extensions-present"
        }))
        .unwrap();

        match msg {
            Message::System { extensions, .. } => {
                let extensions = extensions.unwrap();
                assert_eq!(extensions.len(), 1);
                assert_eq!(
                    extensions.get("cache_control"),
                    Some(&json!({"type": "ephemeral"}))
                );
                assert_eq!(extensions.get("trace_id"), None);
            }
            other => panic!("expected System, got {other:?}"),
        }
    }

    #[test]
    fn tool_error_sets_is_error_flag() {
        let msg = Message::tool_error("call_1", "my_tool", "boom");
        let tool_ref = msg.as_tool().unwrap();

        assert_eq!(tool_ref.tool_call_id, "call_1");
        assert_eq!(tool_ref.name, "my_tool");
        assert_eq!(tool_ref.content.as_text(), Some("boom"));
        assert_eq!(tool_ref.is_error, Some(true));
    }

    #[test]
    fn tool_message_deserializes_multimodal_content() {
        let msg: Message = serde_json::from_value(json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "read_file",
            "content": [
                {"type": "text", "text": "see attached"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/result.png"}}
            ]
        }))
        .unwrap();

        match msg {
            Message::Tool { content, .. } => assert!(matches!(
                content,
                ToolResultContent::Parts(parts)
                    if parts.len() == 2 && parts[0].as_text() == Some("see attached")
            )),
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    #[test]
    fn display_handles_multimodal_and_non_text_assistant_content() {
        let user = Message::user(vec![ContentPart::image_url("https://example.com/cat.png")]);
        assert_eq!(user.to_string(), "user: [multimodal]");

        let assistant = Message::Assistant {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: "{}".into(),
            }],
            name: None,
            extensions: None,
        };
        assert_eq!(assistant.to_string(), "assistant: [non-text content]");
    }

    #[test]
    fn display_truncates_joined_text_without_allocating_full_output() {
        let message = Message::Assistant {
            content: vec![
                ContentBlock::Text {
                    text: "a".repeat(30),
                },
                ContentBlock::Text {
                    text: "b".repeat(30),
                },
            ],
            name: None,
            extensions: None,
        };

        assert_eq!(
            message.to_string(),
            format!("assistant: {}...", "a".repeat(30) + " " + &"b".repeat(19))
        );
    }

    #[test]
    fn user_accepts_multimodal_content_via_into_user_content() {
        let message = Message::user(vec![ContentPart::text("look")]);

        assert!(matches!(
            message,
            Message::User {
                content: UserContent::Parts(parts),
                ..
            } if parts == vec![ContentPart::text("look")]
        ));
    }
}
