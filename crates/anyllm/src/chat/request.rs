use serde::{Deserialize, Serialize};

use crate::{
    Message, RequestOptions, SystemPrompt, Tool, ToolCallRef, ToolChoice, ToolResultContent,
    UserContent,
};

/// A provider-agnostic chat completion request.
///
/// This type intentionally keeps its portable fields public so applications,
/// tests, and wrappers can inspect and adjust requests in place. Construct new
/// values with [`ChatRequest::new`] and the fluent setters; the type remains
/// non-exhaustive so new portable fields can be added without breaking callers.
/// Provider-specific configuration that does not fit the portable core belongs
/// in [`RequestOptions`].
///
/// Does not implement `PartialEq` because [`RequestOptions`] contains
/// type-erased provider-specific values. Use [`ChatRequestRecord`](crate::ChatRequestRecord)
/// for portable, equality-comparable representations (e.g. in tests).
/// Converting through `ChatRequestRecord` is lossy because typed options are not
/// preserved.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChatRequest {
    /// Provider-specific model identifier to route this request to.
    pub model: String,
    /// Request-level system instructions (empty when no system prompt is set).
    pub system: Vec<SystemPrompt>,
    /// Ordered conversation history sent to the model.
    pub messages: Vec<Message>,
    /// Sampling temperature when the provider exposes it.
    pub temperature: Option<f32>,
    /// Maximum number of output tokens to generate.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling threshold when supported by the provider.
    pub top_p: Option<f32>,
    /// Stop sequences that should terminate generation early.
    pub stop: Option<Vec<String>>,
    /// Frequency penalty forwarded to providers that support repetition penalties.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty forwarded to providers that support novelty penalties.
    pub presence_penalty: Option<f32>,
    /// Tool definitions available for this request.
    pub tools: Option<Vec<Tool>>,
    /// Tool selection policy for this request.
    pub tool_choice: Option<ToolChoice>,
    /// Requested output format such as text or JSON schema.
    pub response_format: Option<ResponseFormat>,
    /// Deterministic sampling seed when supported by the provider.
    pub seed: Option<u64>,
    /// Provider-agnostic reasoning/thinking configuration.
    pub reasoning: Option<ReasoningConfig>,
    /// Whether the model may emit multiple tool calls in one turn.
    pub parallel_tool_calls: Option<bool>,
    /// Provider-specific typed request extensions not preserved by portable records.
    pub options: RequestOptions,
}

impl ChatRequest {
    /// Create an empty request targeting the given provider model identifier.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system: Vec::new(),
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            seed: None,
            reasoning: None,
            parallel_tool_calls: None,
            options: RequestOptions::new(),
        }
    }

    /// Replace the request's full message history.
    #[must_use]
    pub fn messages<I>(mut self, messages: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        self.messages = messages.into_iter().collect();
        self
    }

    /// Append a single message to the request.
    #[must_use]
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Shorthand for `.message(Message::system(content))`.
    #[must_use]
    pub fn system(self, content: impl Into<String>) -> Self {
        self.message(Message::system(content))
    }

    /// Shorthand for `.message(Message::user(content))`.
    #[must_use]
    pub fn user(self, content: impl Into<UserContent>) -> Self {
        self.message(Message::user(content))
    }

    /// Shorthand for `.message(Message::assistant(content))`.
    #[must_use]
    pub fn assistant(self, content: impl Into<String>) -> Self {
        self.message(Message::assistant(content))
    }

    /// Push a single message in place.
    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Push a successful tool result using a borrowed tool-call view.
    pub fn push_tool_result(
        &mut self,
        call: ToolCallRef<'_>,
        content: impl Into<ToolResultContent>,
    ) {
        self.messages
            .push(Message::tool_result(call.id, call.name, content));
    }

    /// Push a failed tool result using a borrowed tool-call view.
    pub fn push_tool_error(&mut self, call: ToolCallRef<'_>, error: impl Into<ToolResultContent>) {
        self.messages
            .push(Message::tool_error(call.id, call.name, error));
    }

    /// Set the sampling temperature.
    #[must_use]
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the maximum number of output tokens.
    #[must_use]
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set the nucleus-sampling threshold.
    #[must_use]
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set stop sequences, normalizing an empty collection to `None`.
    #[must_use]
    pub fn stop<I, S>(mut self, sequences: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop = into_non_empty_vec(sequences.into_iter().map(Into::into));
        self
    }

    /// Set the frequency penalty.
    #[must_use]
    pub fn frequency_penalty(mut self, p: f32) -> Self {
        self.frequency_penalty = Some(p);
        self
    }

    /// Set the presence penalty.
    #[must_use]
    pub fn presence_penalty(mut self, p: f32) -> Self {
        self.presence_penalty = Some(p);
        self
    }

    /// Set the deterministic sampling seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the available tool definitions, normalizing an empty collection to `None`.
    #[must_use]
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
    {
        self.tools = into_non_empty_vec(tools);
        self
    }

    /// Set the tool-selection policy for the request.
    #[must_use]
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Request a specific response format such as text or JSON schema.
    #[must_use]
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set provider-agnostic reasoning configuration.
    #[must_use]
    pub fn reasoning(mut self, config: impl Into<ReasoningConfig>) -> Self {
        self.reasoning = Some(config.into());
        self
    }

    /// Control whether the model may emit multiple tool calls in one turn.
    #[must_use]
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Insert a typed provider-specific request option.
    #[must_use]
    pub fn with_option<T>(mut self, option: T) -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        self.options.insert(option);
        self
    }

    /// Borrow a typed provider-specific request option by type.
    pub fn option<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.options.get::<T>()
    }

    /// Mutably borrow a typed provider-specific request option by type.
    pub fn option_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Send + Sync + 'static,
    {
        self.options.get_mut::<T>()
    }

    /// Removes and returns a typed provider-specific request option.
    pub fn take_option<T>(&mut self) -> Option<T>
    where
        T: Send + Sync + 'static,
    {
        self.options.remove::<T>()
    }

    /// Convert this request into its portable record form, dropping typed options.
    #[must_use]
    pub fn to_record_lossy(&self) -> ChatRequestRecord {
        ChatRequestRecord::from_request_lossy(self)
    }

    /// Consume this request into its portable record form, dropping typed options.
    #[must_use]
    pub fn into_record_lossy(self) -> ChatRequestRecord {
        ChatRequestRecord {
            model: self.model,
            messages: self.messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop: self.stop,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: self.response_format,
            seed: self.seed,
            reasoning: self.reasoning,
            parallel_tool_calls: self.parallel_tool_calls,
        }
    }
}

/// Portable, serde-friendly snapshot of a [`ChatRequest`].
///
/// This record is intended for logging, fixtures, and replayable artifacts.
/// It preserves only the portable request fields and intentionally omits typed
/// [`RequestOptions`] entries, which cannot be represented in a provider-agnostic
/// JSON format.
///
/// Converting a `ChatRequest` into a `ChatRequestRecord` is lossless for this
/// portable representation. Rebuilding a `ChatRequest` from a record is lossy
/// because the rebuilt request always has empty typed options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequestRecord {
    /// Provider-specific model identifier.
    pub model: String,
    /// Ordered conversation history.
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Sampling temperature.
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Maximum number of output tokens.
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Nucleus-sampling threshold.
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Presence penalty.
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool definitions available to the model.
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tool-selection policy.
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Requested response format.
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Deterministic sampling seed.
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Provider-agnostic reasoning configuration.
    pub reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Whether parallel tool calls are allowed.
    pub parallel_tool_calls: Option<bool>,
}

impl From<ChatRequest> for ChatRequestRecord {
    fn from(request: ChatRequest) -> Self {
        request.into_record_lossy()
    }
}

impl From<&ChatRequest> for ChatRequestRecord {
    fn from(request: &ChatRequest) -> Self {
        Self::from_request_lossy(request)
    }
}

impl ChatRequestRecord {
    /// Build the portable record representation of a request.
    ///
    /// This is lossy because typed [`RequestOptions`] entries are omitted.
    #[must_use]
    pub fn from_request_lossy(request: &ChatRequest) -> Self {
        Self {
            model: request.model.clone(),
            messages: request.messages.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stop: request.stop.clone(),
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            response_format: request.response_format.clone(),
            seed: request.seed,
            reasoning: request.reasoning.clone(),
            parallel_tool_calls: request.parallel_tool_calls,
        }
    }

    /// Rebuild a `ChatRequest` from the portable record.
    ///
    /// This conversion is lossy: typed [`RequestOptions`] are not representable
    /// in `ChatRequestRecord`, so the rebuilt request always has an empty
    /// options bag.
    #[must_use]
    pub fn into_chat_request_lossy(self) -> ChatRequest {
        ChatRequest {
            model: self.model,
            system: Vec::new(),
            messages: self.messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop: self.stop,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: self.response_format,
            seed: self.seed,
            reasoning: self.reasoning,
            parallel_tool_calls: self.parallel_tool_calls,
            options: RequestOptions::new(),
        }
    }
}

fn into_non_empty_vec<I, T>(items: I) -> Option<Vec<T>>
where
    I: IntoIterator<Item = T>,
{
    let items: Vec<T> = items.into_iter().collect();
    (!items.is_empty()).then_some(items)
}

/// How much effort the model should spend on reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    /// Minimal reasoning effort.
    Low,
    /// Balanced reasoning effort.
    Medium,
    /// Maximum reasoning effort.
    High,
}

/// Configuration for model reasoning/thinking behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Enables or disables provider-specific reasoning features for the request.
    pub enabled: bool,
    /// Optional token budget for hidden reasoning output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Optional coarse reasoning-effort hint for providers that support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
}

impl From<ReasoningEffort> for ReasoningConfig {
    fn from(effort: ReasoningEffort) -> Self {
        Self {
            enabled: true,
            budget_tokens: None,
            effort: Some(effort),
        }
    }
}

/// Requested response format from the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseFormat {
    /// Plain text output
    Text,
    /// JSON output without an explicit schema
    Json,
    /// JSON output constrained by a JSON Schema
    JsonSchema {
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Optional schema name supplied to providers that support it.
        name: Option<String>,
        /// JSON Schema describing the requested output shape.
        schema: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Whether the provider should enforce the schema strictly when supported.
        strict: Option<bool>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "extract")]
    use crate::ExtractionMode;
    use crate::SystemPrompt;
    use serde_json::json;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DemoOption {
        enabled: bool,
    }

    #[test]
    #[cfg(feature = "extract")]
    fn request_record_round_trip_preserves_portable_fields() {
        let request = ChatRequest::new("gpt-4o")
            .system("Be concise")
            .message(Message::user("Review this"))
            .temperature(0.3)
            .max_tokens(50)
            .response_format(ResponseFormat::Json)
            .with_option(ExtractionMode::ForcedTool);

        let record = ChatRequestRecord::from(&request);
        let rebuilt = record.clone().into_chat_request_lossy();

        assert_eq!(record.model, "gpt-4o");
        assert_eq!(record.temperature, Some(0.3));
        assert_eq!(record.response_format, Some(ResponseFormat::Json));
        assert!(rebuilt.option::<ExtractionMode>().is_none());
        assert_eq!(rebuilt.model, request.model);
        assert_eq!(rebuilt.messages, request.messages);
    }

    #[test]
    fn request_record_round_trip_preserves_all_portable_fields_and_drops_options() {
        let request =
            ChatRequest::new("gpt-4.1")
                .system("Be concise")
                .message(Message::user("Review this"))
                .temperature(0.3)
                .max_tokens(50)
                .top_p(0.9)
                .stop(["END", "STOP"])
                .frequency_penalty(0.2)
                .presence_penalty(0.1)
                .tools(vec![Tool::new(
                "search",
                serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            )
            .description("Search docs")])
                .tool_choice(ToolChoice::Specific {
                    name: "search".into(),
                })
                .response_format(ResponseFormat::JsonSchema {
                    name: Some("answer".into()),
                    schema: serde_json::json!({"type": "object"}),
                    strict: Some(true),
                })
                .seed(42)
                .reasoning(ReasoningConfig {
                    enabled: true,
                    budget_tokens: Some(256),
                    effort: Some(ReasoningEffort::High),
                })
                .parallel_tool_calls(true)
                .with_option(DemoOption { enabled: true });

        let record = ChatRequestRecord::from(&request);
        let rebuilt = record.clone().into_chat_request_lossy();

        assert_eq!(record.model, "gpt-4.1");
        assert_eq!(record.messages, request.messages);
        assert_eq!(record.temperature, Some(0.3));
        assert_eq!(record.max_tokens, Some(50));
        assert_eq!(record.top_p, Some(0.9));
        assert_eq!(record.stop, Some(vec!["END".into(), "STOP".into()]));
        assert_eq!(record.frequency_penalty, Some(0.2));
        assert_eq!(record.presence_penalty, Some(0.1));
        assert_eq!(record.tools, request.tools);
        assert_eq!(
            record.tool_choice,
            Some(ToolChoice::Specific {
                name: "search".into(),
            })
        );
        assert_eq!(record.response_format, request.response_format);
        assert_eq!(record.seed, Some(42));
        assert_eq!(record.reasoning, request.reasoning);
        assert_eq!(record.parallel_tool_calls, Some(true));

        assert_eq!(rebuilt.model, request.model);
        assert_eq!(rebuilt.messages, request.messages);
        assert_eq!(rebuilt.temperature, request.temperature);
        assert_eq!(rebuilt.max_tokens, request.max_tokens);
        assert_eq!(rebuilt.top_p, request.top_p);
        assert_eq!(rebuilt.stop, request.stop);
        assert_eq!(rebuilt.frequency_penalty, request.frequency_penalty);
        assert_eq!(rebuilt.presence_penalty, request.presence_penalty);
        assert_eq!(rebuilt.tools, request.tools);
        assert_eq!(rebuilt.tool_choice, request.tool_choice);
        assert_eq!(rebuilt.response_format, request.response_format);
        assert_eq!(rebuilt.seed, request.seed);
        assert_eq!(rebuilt.reasoning, request.reasoning);
        assert_eq!(rebuilt.parallel_tool_calls, request.parallel_tool_calls);
        assert!(rebuilt.option::<DemoOption>().is_none());
        assert!(rebuilt.options.is_empty());
    }

    #[test]
    fn request_record_from_owned_matches_lossy_helper() {
        let request = ChatRequest::new("gpt-4o")
            .message(Message::user("Hello"))
            .parallel_tool_calls(true)
            .with_option(DemoOption { enabled: true });

        let borrowed = ChatRequestRecord::from_request_lossy(&request);
        let owned = ChatRequestRecord::from(request);

        assert_eq!(borrowed, owned);
    }

    #[test]
    fn request_record_lossy_rebuild_drops_all_typed_options() {
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct OtherOption {
            level: u8,
        }

        let request = ChatRequest::new("gpt-4.1")
            .message(Message::user("Review this"))
            .with_option(DemoOption { enabled: true })
            .with_option(OtherOption { level: 3 });

        let rebuilt = ChatRequestRecord::from(&request).into_chat_request_lossy();

        assert!(rebuilt.option::<DemoOption>().is_none());
        assert!(rebuilt.option::<OtherOption>().is_none());
        assert!(rebuilt.options.is_empty());
    }

    #[test]
    fn request_record_serde_skips_absent_optional_fields() {
        let request = ChatRequest::new("gpt-4o").message(Message::user("Hello"));

        let value = serde_json::to_value(ChatRequestRecord::from(&request)).unwrap();
        let obj = value.as_object().unwrap();
        assert!(obj.contains_key("model"));
        assert!(obj.contains_key("messages"));
        assert!(!obj.contains_key("temperature"));
        assert!(!obj.contains_key("max_tokens"));
        assert!(!obj.contains_key("top_p"));
        assert!(!obj.contains_key("stop"));
        assert!(!obj.contains_key("frequency_penalty"));
        assert!(!obj.contains_key("presence_penalty"));
        assert!(!obj.contains_key("tools"));
        assert!(!obj.contains_key("tool_choice"));
        assert!(!obj.contains_key("response_format"));
        assert!(!obj.contains_key("seed"));
        assert!(!obj.contains_key("reasoning"));
        assert!(!obj.contains_key("parallel_tool_calls"));
    }

    #[test]
    fn request_record_serde_round_trip_preserves_portable_fields() {
        let request = ChatRequest::new("gpt-4.1")
            .system("Be concise")
            .message(Message::user("Review this"))
            .temperature(0.3)
            .max_tokens(50)
            .top_p(0.9)
            .stop(["END"])
            .tool_choice(ToolChoice::Required)
            .response_format(ResponseFormat::Json)
            .parallel_tool_calls(false);

        let serialized = serde_json::to_string(&ChatRequestRecord::from(&request)).unwrap();
        let record: ChatRequestRecord = serde_json::from_str(&serialized).unwrap();

        assert_eq!(record.model, request.model);
        assert_eq!(record.messages, request.messages);
        assert_eq!(record.temperature, request.temperature);
        assert_eq!(record.max_tokens, request.max_tokens);
        assert_eq!(record.top_p, request.top_p);
        assert_eq!(record.stop, request.stop);
        assert_eq!(record.tool_choice, request.tool_choice);
        assert_eq!(record.response_format, request.response_format);
        assert_eq!(record.parallel_tool_calls, request.parallel_tool_calls);
    }

    #[test]
    fn messages_builder_replaces_vec() {
        let req = ChatRequest::new("gpt-4o")
            .message(Message::user("first"))
            .messages(vec![Message::user("replaced")]);

        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role(), "user");
    }

    #[test]
    fn user_and_assistant_shorthands_append_messages() {
        let req = ChatRequest::new("gpt-4o")
            .system("Be concise")
            .user("Hello")
            .assistant("Hi");

        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[1], Message::user("Hello"));
        assert_eq!(req.messages[2], Message::assistant("Hi"));
    }

    #[test]
    fn push_helpers_append_messages_in_place() {
        let mut req = ChatRequest::new("gpt-4o");
        req.push_message(Message::user("What is the weather?"));

        let call_block = crate::ContentBlock::ToolCall {
            id: "call_1".into(),
            name: "lookup_weather".into(),
            arguments: r#"{"city":"San Francisco"}"#.into(),
        };
        let call = call_block.as_tool_call().unwrap();

        req.push_tool_result(call, "foggy");
        req.push_tool_error(call, "service unavailable");

        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0], Message::user("What is the weather?"));
        assert_eq!(
            req.messages[1],
            Message::tool_result("call_1", "lookup_weather", "foggy")
        );
        assert_eq!(
            req.messages[2],
            Message::tool_error("call_1", "lookup_weather", "service unavailable")
        );
    }

    #[test]
    fn full_builder_chain() {
        let req = ChatRequest::new("claude-3-opus")
            .system("You are a helpful assistant")
            .user("Hello")
            .temperature(0.7)
            .max_tokens(4096)
            .top_p(0.95)
            .stop(["###"])
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
            .seed(123)
            .tools(vec![
                Tool::new("search", json!({"type": "object"})).description("Search the web"),
            ])
            .tool_choice(ToolChoice::Auto)
            .response_format(ResponseFormat::Text)
            .reasoning(ReasoningEffort::High)
            .parallel_tool_calls(true)
            .with_option(DemoOption { enabled: true });

        assert_eq!(req.model, "claude-3-opus");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(4096));
        assert_eq!(req.top_p, Some(0.95));
        assert_eq!(req.stop, Some(vec!["###".into()]));
        assert_eq!(req.frequency_penalty, Some(0.1));
        assert_eq!(req.presence_penalty, Some(0.2));
        assert_eq!(req.seed, Some(123));
        assert!(req.tools.is_some());
        assert_eq!(req.tool_choice, Some(ToolChoice::Auto));
        assert_eq!(req.response_format, Some(ResponseFormat::Text));
        assert_eq!(req.reasoning, Some(ReasoningEffort::High.into()));
        assert_eq!(req.parallel_tool_calls, Some(true));
        assert_eq!(
            req.option::<DemoOption>(),
            Some(&DemoOption { enabled: true })
        );
    }

    #[test]
    fn reasoning_effort_converts_to_enabled_reasoning_config() {
        let config = ReasoningConfig::from(ReasoningEffort::Medium);

        assert_eq!(
            config,
            ReasoningConfig {
                enabled: true,
                budget_tokens: None,
                effort: Some(ReasoningEffort::Medium),
            }
        );
    }

    #[test]
    fn stop_normalizes_empty_sequences_to_none() {
        let req = ChatRequest::new("gpt-4o").stop(Vec::<String>::new());

        assert_eq!(req.stop, None);
    }

    #[test]
    fn tools_normalize_empty_collection_to_none() {
        let req = ChatRequest::new("gpt-4o").tools(Vec::<Tool>::new());

        assert_eq!(req.tools, None);
    }

    #[test]
    fn record_helpers_preserve_portable_fields() {
        let req = ChatRequest::new("gpt-4o")
            .system("Be concise")
            .message(Message::user("Hello"))
            .stop(["END"])
            .parallel_tool_calls(true)
            .with_option(DemoOption { enabled: true });

        let borrowed = req.to_record_lossy();
        let owned = req.clone().into_record_lossy();

        assert_eq!(borrowed, owned);
        assert_eq!(borrowed.model, "gpt-4o");
        assert_eq!(borrowed.stop, Some(vec!["END".into()]));
        assert_eq!(borrowed.parallel_tool_calls, Some(true));
    }

    #[test]
    fn chat_request_new_has_empty_system() {
        let req = ChatRequest::new("gpt-4o");
        assert!(req.system.is_empty());
    }

    #[test]
    fn chat_request_system_field_holds_prompts() {
        let mut req = ChatRequest::new("gpt-4o");
        req.system.push(SystemPrompt::new("Be concise"));
        assert_eq!(req.system.len(), 1);
        assert_eq!(req.system[0].content, "Be concise");
    }

    #[test]
    fn chat_request_clone_preserves_system() {
        let mut req = ChatRequest::new("gpt-4o");
        req.system.push(SystemPrompt::new("X"));
        let cloned = req.clone();
        assert_eq!(cloned.system.len(), 1);
        assert_eq!(cloned.system[0].content, "X");
    }
}
