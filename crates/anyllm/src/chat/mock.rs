use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatStreamExt, ContentBlock, Error, ExtraMap, FinishReason, ProviderIdentity, ResponseMetadata,
    Result, SingleResponseStream, StreamEvent, Usage,
};

/// Deterministic chat provider for tests and downstream fixtures.
///
/// Returns canned responses in sequence, records every request, and supports
/// optional delays for timeout and concurrency tests.
#[derive(Debug, Clone)]
pub struct MockProvider {
    state: Arc<Mutex<MockChatState>>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    provider_name: &'static str,
}

impl MockProvider {
    /// Creates an empty mock provider and configures it via a closure.
    ///
    /// ```rust
    /// use anyllm::{ChatResponseBuilder, MockProvider};
    ///
    /// let provider = MockProvider::build(|builder| {
    ///     builder.response(ChatResponseBuilder::new().text("hello").build())
    /// });
    ///
    /// assert_eq!(provider.pending_responses(), 1);
    /// ```
    #[must_use]
    pub fn build(configure: impl FnOnce(MockProviderBuilder) -> MockProviderBuilder) -> Self {
        configure(MockProviderBuilder::new()).build()
    }

    /// Create a mock provider with no queued responses yet.
    ///
    /// Useful when tests want to configure the provider incrementally with
    /// `push_response(...)` without writing typed `Vec::new()` boilerplate.
    ///
    /// ```rust
    /// use anyllm::{MockProvider, MockResponse};
    ///
    /// let provider = MockProvider::empty();
    /// provider.push_response(MockResponse::text("hello"));
    ///
    /// assert_eq!(provider.pending_responses(), 1);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self::new(std::iter::empty::<MockResponse>())
    }

    /// Create a mock provider from an ordered queue of canned responses.
    #[must_use]
    pub fn new<I, R>(responses: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: Into<MockResponse>,
    {
        Self {
            state: Arc::new(Mutex::new(MockChatState {
                responses: responses.into_iter().map(Into::into).collect(),
                requests: Vec::new(),
            })),
            chat_capabilities: HashMap::new(),
            provider_name: "mock",
        }
    }

    /// Convenience for a single canned response.
    #[must_use]
    pub fn with_response(response: ChatResponse) -> Self {
        Self::new([response])
    }

    /// Convenience for a single canned error.
    #[must_use]
    pub fn with_error(error: Error) -> Self {
        Self::new([MockResponse::Error(error)])
    }

    /// Convenience for a single text response.
    #[must_use]
    pub fn with_text(text: impl Into<String>) -> Self {
        Self::new([MockResponse::text(text)])
    }

    /// Convenience for a single tool-call response.
    #[must_use]
    pub fn with_tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::new([MockResponse::tool_call(id, name, arguments)])
    }

    /// Convenience for all-success responses.
    #[must_use]
    pub fn from_responses<I>(responses: I) -> Self
    where
        I: IntoIterator<Item = ChatResponse>,
    {
        Self::new(responses)
    }

    /// Convenience for a single tool-call turn followed by a final text answer.
    ///
    /// ```rust
    /// use anyllm::MockProvider;
    ///
    /// let provider = MockProvider::tool_round_trip(
    ///     "call_1",
    ///     "lookup_weather",
    ///     serde_json::json!({ "city": "San Francisco" }),
    ///     "Cool and foggy.",
    /// );
    ///
    /// assert_eq!(provider.pending_responses(), 2);
    /// ```
    #[must_use]
    pub fn tool_round_trip(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
        final_text: impl Into<String>,
    ) -> Self {
        Self::from_tool_round_trips([MockToolRoundTrip::single_tool_call_text(
            id, name, arguments, final_text,
        )])
    }

    /// Convenience for a sequence of tool-call turns and follow-up responses.
    ///
    /// ```rust
    /// use anyllm::{MockProvider, MockToolRoundTrip};
    ///
    /// let provider = MockProvider::from_tool_round_trips([
    ///     MockToolRoundTrip::single_tool_call_text(
    ///         "call_1",
    ///         "lookup_weather",
    ///         serde_json::json!({ "city": "San Francisco" }),
    ///         "Cool and foggy.",
    ///     ),
    /// ]);
    ///
    /// assert_eq!(provider.pending_responses(), 2);
    /// ```
    #[must_use]
    pub fn from_tool_round_trips<I>(round_trips: I) -> Self
    where
        I: IntoIterator<Item = MockToolRoundTrip>,
    {
        Self::new(
            round_trips
                .into_iter()
                .flat_map(MockToolRoundTrip::into_responses),
        )
    }

    /// Set explicit support information for one chat capability.
    #[must_use]
    pub fn with_chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Extend this mock provider with explicit support information for many capabilities.
    #[must_use]
    pub fn with_chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = (ChatCapability, CapabilitySupport)>,
    {
        self.chat_capabilities.extend(capabilities);
        self
    }

    /// Mark each listed capability as supported.
    #[must_use]
    pub fn with_supported_chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = ChatCapability>,
    {
        for capability in capabilities {
            self.chat_capabilities
                .insert(capability, CapabilitySupport::Supported);
        }
        self
    }

    /// Override the provider name reported by this mock provider.
    #[must_use]
    pub fn with_provider_name(mut self, provider_name: &'static str) -> Self {
        self.provider_name = provider_name;
        self
    }

    /// Push one additional canned response onto the back of the queue.
    pub fn push_response<R>(&self, response: R)
    where
        R: Into<MockResponse>,
    {
        self.state
            .lock()
            .unwrap()
            .responses
            .push_back(response.into());
    }

    /// Return every request this mock provider has received so far.
    #[must_use]
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Return the most recent recorded request, if any.
    #[must_use]
    pub fn last_request(&self) -> Option<ChatRequest> {
        self.state.lock().unwrap().requests.last().cloned()
    }

    /// Return how many requests have been dispatched to this mock provider.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.state.lock().unwrap().requests.len()
    }

    /// Return how many canned responses remain queued.
    #[must_use]
    pub fn pending_responses(&self) -> usize {
        self.state.lock().unwrap().responses.len()
    }
}

impl ProviderIdentity for MockProvider {
    fn provider_name(&self) -> &'static str {
        self.provider_name
    }
}

impl ChatProvider for MockProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let response = {
            let mut state = self.state.lock().unwrap();
            state.take_next_response(request)
        };

        response.into_result().await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        Ok(self.chat(request).await?.into())
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.chat_capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }
}

#[cfg(feature = "extract")]
impl crate::ExtractExt for MockProvider {}

/// A canned response returned by [`MockProvider`].
#[derive(Debug)]
pub enum MockResponse {
    /// Return a successful full response.
    Success(ChatResponse),
    /// Return an error immediately.
    Error(Error),
    /// Wait, then return another mocked response.
    Delayed(Duration, Box<MockResponse>),
}

impl MockResponse {
    /// Convenience for a text-only response.
    ///
    /// ```rust
    /// use anyllm::MockResponse;
    ///
    /// let response = MockResponse::text("hello");
    /// assert!(matches!(response, MockResponse::Success(_)));
    /// ```
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        ChatResponseBuilder::new().text(text).build().into()
    }

    /// Convenience for a reasoning-plus-text response.
    ///
    /// ```rust
    /// use anyllm::MockResponse;
    ///
    /// let response = MockResponse::reasoning_text("thinking", "done");
    /// assert!(matches!(response, MockResponse::Success(_)));
    /// ```
    #[must_use]
    pub fn reasoning_text(reasoning: impl Into<String>, text: impl Into<String>) -> Self {
        ChatResponseBuilder::new()
            .reasoning(reasoning)
            .text(text)
            .build()
            .into()
    }

    /// Convenience for a tool-call response with `FinishReason::ToolCalls`.
    ///
    /// ```rust
    /// use anyllm::MockResponse;
    ///
    /// let response = MockResponse::tool_call(
    ///     "call_1",
    ///     "search",
    ///     serde_json::json!({ "q": "rust" }),
    /// );
    /// assert!(matches!(response, MockResponse::Success(_)));
    /// ```
    #[must_use]
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        build_tool_call_response(id, name, arguments).into()
    }

    /// Wrap a mock response with a delay before it resolves.
    ///
    /// ```rust
    /// use std::time::Duration;
    ///
    /// use anyllm::MockResponse;
    ///
    /// let delayed = MockResponse::delayed(Duration::from_millis(5), MockResponse::text("hi"));
    /// assert!(matches!(delayed, MockResponse::Delayed(_, _)));
    /// ```
    #[must_use]
    pub fn delayed(duration: Duration, response: impl Into<MockResponse>) -> Self {
        Self::Delayed(duration, Box::new(response.into()))
    }

    async fn into_result(self) -> Result<ChatResponse> {
        let mut next = self;
        loop {
            match next {
                Self::Success(response) => return Ok(response),
                Self::Error(error) => return Err(error),
                Self::Delayed(duration, response) => {
                    futures_timer::Delay::new(duration).await;
                    next = *response;
                }
            }
        }
    }
}

impl From<ChatResponse> for MockResponse {
    fn from(response: ChatResponse) -> Self {
        Self::Success(response)
    }
}

impl From<Error> for MockResponse {
    fn from(error: Error) -> Self {
        Self::Error(error)
    }
}

impl From<Result<ChatResponse>> for MockResponse {
    fn from(result: Result<ChatResponse>) -> Self {
        match result {
            Ok(response) => Self::Success(response),
            Err(error) => Self::Error(error),
        }
    }
}

/// Two canned responses representing a single tool-call round trip.
///
/// The first response is typically an assistant tool call and the second is the
/// follow-up assistant answer after the caller has executed the tool.
#[derive(Debug, Clone)]
pub struct MockToolRoundTrip {
    first: ChatResponse,
    followup: ChatResponse,
}

impl MockToolRoundTrip {
    /// Create a custom two-step tool interaction from explicit responses.
    #[must_use]
    pub fn new(first: ChatResponse, followup: ChatResponse) -> Self {
        Self { first, followup }
    }

    /// Create a round trip where the first response is a single tool call and
    /// the second is an explicit follow-up response.
    #[must_use]
    pub fn single_tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
        followup: ChatResponse,
    ) -> Self {
        Self {
            first: build_tool_call_response(id, name, arguments),
            followup,
        }
    }

    /// Create a round trip where the follow-up is a simple text response.
    ///
    /// ```rust
    /// use anyllm::MockToolRoundTrip;
    ///
    /// let round_trip = MockToolRoundTrip::single_tool_call_text(
    ///     "call_1",
    ///     "search",
    ///     serde_json::json!({ "q": "rust" }),
    ///     "Found it.",
    /// );
    ///
    /// let _ = round_trip;
    /// ```
    #[must_use]
    pub fn single_tool_call_text(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
        followup_text: impl Into<String>,
    ) -> Self {
        Self::single_tool_call(
            id,
            name,
            arguments,
            ChatResponseBuilder::new().text(followup_text).build(),
        )
    }

    fn into_responses(self) -> [ChatResponse; 2] {
        [self.first, self.followup]
    }
}

/// Builder used by [`MockProvider::build`] to assemble canned responses and
/// capability metadata.
#[derive(Debug, Default)]
pub struct MockProviderBuilder {
    responses: Vec<MockResponse>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    provider_name: &'static str,
}

impl MockProviderBuilder {
    /// Create an empty mock-provider builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
            chat_capabilities: HashMap::new(),
            provider_name: "mock",
        }
    }

    /// Append one canned response or error to the provider queue.
    #[must_use]
    pub fn response<R>(mut self, response: R) -> Self
    where
        R: Into<MockResponse>,
    {
        self.responses.push(response.into());
        self
    }

    /// Append a single text response.
    #[must_use]
    pub fn text(self, text: impl Into<String>) -> Self {
        self.response(MockResponse::text(text))
    }

    /// Append a single error response.
    #[must_use]
    pub fn error(self, error: Error) -> Self {
        self.response(error)
    }

    /// Append a single tool-call response.
    #[must_use]
    pub fn tool_call(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        self.response(MockResponse::tool_call(id, name, arguments))
    }

    /// Append multiple canned responses or errors in order.
    #[must_use]
    pub fn responses<I, R>(mut self, responses: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: Into<MockResponse>,
    {
        self.responses.extend(responses.into_iter().map(Into::into));
        self
    }

    /// Set explicit support information for one chat capability.
    #[must_use]
    pub fn chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Extend the builder with explicit support information for many capabilities.
    #[must_use]
    pub fn chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = (ChatCapability, CapabilitySupport)>,
    {
        self.chat_capabilities.extend(capabilities);
        self
    }

    /// Mark each listed capability as supported.
    #[must_use]
    pub fn supported_chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = ChatCapability>,
    {
        for capability in capabilities {
            self.chat_capabilities
                .insert(capability, CapabilitySupport::Supported);
        }
        self
    }

    /// Override the provider name reported by the built mock provider.
    #[must_use]
    pub fn provider_name(mut self, provider_name: &'static str) -> Self {
        self.provider_name = provider_name;
        self
    }

    /// Build the configured [`MockProvider`].
    #[must_use]
    pub fn build(self) -> MockProvider {
        MockProvider::new(self.responses)
            .with_chat_capabilities(self.chat_capabilities)
            .with_provider_name(self.provider_name)
    }
}

#[derive(Debug, Default)]
struct MockChatState {
    responses: VecDeque<MockResponse>,
    requests: Vec<ChatRequest>,
}

impl MockChatState {
    fn take_next_response(&mut self, request: &ChatRequest) -> MockResponse {
        self.requests.push(request.clone());
        self.responses
            .pop_front()
            .expect("MockProvider: no more responses configured")
    }
}

/// Deterministic streaming provider for stream reconstruction and error tests.
#[derive(Debug, Clone)]
pub struct MockStreamingProvider {
    state: Arc<Mutex<MockStreamingState>>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    provider_name: &'static str,
}

impl MockStreamingProvider {
    /// Creates an empty streaming mock provider and configures it via a closure.
    #[must_use]
    pub fn build(
        configure: impl FnOnce(MockStreamingProviderBuilder) -> MockStreamingProviderBuilder,
    ) -> Self {
        configure(MockStreamingProviderBuilder::new()).build()
    }

    /// Create a streaming mock provider with no queued streams yet.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(std::iter::empty::<Vec<MockStreamEvent>>())
    }

    /// Create a streaming mock provider from an ordered queue of mocked stream transcripts.
    #[must_use]
    pub fn new<I, S, E>(streams: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        Self {
            state: Arc::new(Mutex::new(MockStreamingState {
                streams: streams
                    .into_iter()
                    .map(|stream| stream.into_iter().map(Into::into).collect())
                    .collect(),
                requests: Vec::new(),
            })),
            chat_capabilities: HashMap::from([(
                ChatCapability::Streaming,
                CapabilitySupport::Supported,
            )]),
            provider_name: "mock_stream",
        }
    }

    /// Convenience for a single mocked stream transcript.
    ///
    /// ```rust
    /// use anyllm::{MockStreamEvent, MockStreamingProvider};
    ///
    /// let provider = MockStreamingProvider::with_stream(
    ///     MockStreamEvent::text_response("hello"),
    /// );
    ///
    /// assert_eq!(provider.pending_streams(), 1);
    /// ```
    #[must_use]
    pub fn with_stream<S, E>(stream: S) -> Self
    where
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        Self::new([stream])
    }

    /// Convenience for a single text transcript.
    #[must_use]
    pub fn with_text(text: impl Into<String>) -> Self {
        Self::with_stream(MockStreamEvent::text_response(text))
    }

    /// Convenience for a single tool-call transcript.
    #[must_use]
    pub fn with_tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::with_stream(MockStreamEvent::tool_call_response(id, name, arguments))
    }

    /// Convenience for streaming a single full response through the normalized
    /// event model.
    ///
    /// ```rust
    /// use anyllm::{ChatResponseBuilder, MockStreamingProvider};
    ///
    /// let provider = MockStreamingProvider::from_response(
    ///     ChatResponseBuilder::new().text("hello").build(),
    /// );
    ///
    /// assert_eq!(provider.pending_streams(), 1);
    /// ```
    #[must_use]
    pub fn from_response(response: ChatResponse) -> Self {
        Self::with_stream(MockStreamEvent::from_response(response))
    }

    /// Convenience for multiple full responses, each converted into one mocked
    /// stream transcript.
    ///
    /// ```rust
    /// use anyllm::{ChatResponseBuilder, MockStreamingProvider};
    ///
    /// let provider = MockStreamingProvider::from_responses([
    ///     ChatResponseBuilder::new().text("first").build(),
    ///     ChatResponseBuilder::new().text("second").build(),
    /// ]);
    ///
    /// assert_eq!(provider.pending_streams(), 2);
    /// ```
    #[must_use]
    pub fn from_responses<I>(responses: I) -> Self
    where
        I: IntoIterator<Item = ChatResponse>,
    {
        Self::new(responses.into_iter().map(MockStreamEvent::from_response))
    }

    /// Set explicit support information for one chat capability.
    #[must_use]
    pub fn with_chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Extend this streaming mock with explicit support information for many capabilities.
    #[must_use]
    pub fn with_chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = (ChatCapability, CapabilitySupport)>,
    {
        self.chat_capabilities.extend(capabilities);
        self
    }

    /// Mark each listed capability as supported.
    #[must_use]
    pub fn with_supported_chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = ChatCapability>,
    {
        for capability in capabilities {
            self.chat_capabilities
                .insert(capability, CapabilitySupport::Supported);
        }
        self
    }

    /// Override the provider name reported by this streaming mock provider.
    #[must_use]
    pub fn with_provider_name(mut self, provider_name: &'static str) -> Self {
        self.provider_name = provider_name;
        self
    }

    /// Push one additional mocked stream transcript onto the back of the queue.
    pub fn push_stream<S, E>(&self, stream: S)
    where
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        self.state
            .lock()
            .unwrap()
            .streams
            .push_back(stream.into_iter().map(Into::into).collect());
    }

    /// Return every request this streaming mock provider has received so far.
    #[must_use]
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Return the most recent recorded request, if any.
    #[must_use]
    pub fn last_request(&self) -> Option<ChatRequest> {
        self.state.lock().unwrap().requests.last().cloned()
    }

    /// Return how many requests have been dispatched to this streaming mock provider.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.state.lock().unwrap().requests.len()
    }

    /// Return how many mocked stream transcripts remain queued.
    #[must_use]
    pub fn pending_streams(&self) -> usize {
        self.state.lock().unwrap().streams.len()
    }
}

impl ProviderIdentity for MockStreamingProvider {
    fn provider_name(&self) -> &'static str {
        self.provider_name
    }
}

impl ChatProvider for MockStreamingProvider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.chat_stream(request).await?.collect_response().await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let stream = {
            let mut state = self.state.lock().unwrap();
            state.take_next_stream(request)
        };

        let stream = futures_util::stream::unfold(stream, |mut pending| async move {
            loop {
                match pending.pop_front() {
                    Some(MockStreamEvent::Delay(duration)) => {
                        futures_timer::Delay::new(duration).await;
                    }
                    Some(MockStreamEvent::Event(event)) => return Some((Ok(event), pending)),
                    Some(MockStreamEvent::Error(error)) => return Some((Err(error), pending)),
                    None => return None,
                }
            }
        });

        Ok(Box::pin(stream) as ChatStream)
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.chat_capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }
}

#[cfg(feature = "extract")]
impl crate::ExtractExt for MockStreamingProvider {}

/// One item in a mocked streaming transcript.
#[derive(Debug)]
pub enum MockStreamEvent {
    /// Emit one normalized stream event.
    Event(StreamEvent),
    /// Emit one terminal stream error.
    Error(Error),
    /// Wait before yielding the next mocked item.
    Delay(Duration),
}

impl MockStreamEvent {
    /// Insert a delay before the next streamed item.
    #[must_use]
    pub fn delayed(duration: Duration) -> Self {
        Self::Delay(duration)
    }

    /// Convert a full response into the normalized event transcript emitted by
    /// [`SingleResponseStream`](crate::SingleResponseStream).
    ///
    /// ```rust
    /// use anyllm::{ChatResponseBuilder, MockStreamEvent};
    ///
    /// let transcript = MockStreamEvent::from_response(
    ///     ChatResponseBuilder::new().text("hello").build(),
    /// );
    ///
    /// assert!(!transcript.is_empty());
    /// ```
    #[must_use]
    pub fn from_response(response: ChatResponse) -> Vec<Self> {
        response_to_mock_stream(response)
    }

    /// Convenience transcript for a simple text response.
    ///
    /// ```rust
    /// use anyllm::MockStreamEvent;
    ///
    /// let transcript = MockStreamEvent::text_response("hello");
    /// assert!(!transcript.is_empty());
    /// ```
    #[must_use]
    pub fn text_response(text: impl Into<String>) -> Vec<Self> {
        Self::from_response(ChatResponseBuilder::new().text(text).build())
    }

    /// Convenience transcript for a single reasoning block followed by text.
    ///
    /// ```rust
    /// use anyllm::MockStreamEvent;
    ///
    /// let transcript = MockStreamEvent::reasoning_response("thinking", "done");
    /// assert!(!transcript.is_empty());
    /// ```
    #[must_use]
    pub fn reasoning_response(reasoning: impl Into<String>, text: impl Into<String>) -> Vec<Self> {
        Self::from_response(
            ChatResponseBuilder::new()
                .reasoning(reasoning)
                .text(text)
                .build(),
        )
    }

    /// Convenience transcript for a single tool call response.
    ///
    /// ```rust
    /// use anyllm::MockStreamEvent;
    ///
    /// let transcript = MockStreamEvent::tool_call_response(
    ///     "call_1",
    ///     "search",
    ///     serde_json::json!({ "q": "rust" }),
    /// );
    /// assert!(!transcript.is_empty());
    /// ```
    #[must_use]
    pub fn tool_call_response(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Vec<Self> {
        Self::from_response(build_tool_call_response(id, name, arguments))
    }
}

impl From<StreamEvent> for MockStreamEvent {
    fn from(event: StreamEvent) -> Self {
        Self::Event(event)
    }
}

impl From<Error> for MockStreamEvent {
    fn from(error: Error) -> Self {
        Self::Error(error)
    }
}

impl From<Result<StreamEvent>> for MockStreamEvent {
    fn from(result: Result<StreamEvent>) -> Self {
        match result {
            Ok(event) => Self::Event(event),
            Err(error) => Self::Error(error),
        }
    }
}

/// Builder used by [`MockStreamingProvider::build`] to assemble mocked stream
/// transcripts and capability metadata.
#[derive(Debug, Default)]
pub struct MockStreamingProviderBuilder {
    streams: Vec<Vec<MockStreamEvent>>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    provider_name: &'static str,
}

impl MockStreamingProviderBuilder {
    /// Create an empty streaming mock-provider builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
            chat_capabilities: HashMap::from([(
                ChatCapability::Streaming,
                CapabilitySupport::Supported,
            )]),
            provider_name: "mock_stream",
        }
    }

    /// Append one mocked stream transcript.
    #[must_use]
    pub fn stream<S, E>(mut self, stream: S) -> Self
    where
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        self.streams
            .push(stream.into_iter().map(Into::into).collect());
        self
    }

    /// Append a simple text transcript.
    #[must_use]
    pub fn text(self, text: impl Into<String>) -> Self {
        self.stream(MockStreamEvent::text_response(text))
    }

    /// Append a single tool-call transcript.
    #[must_use]
    pub fn tool_call(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        self.stream(MockStreamEvent::tool_call_response(id, name, arguments))
    }

    /// Set explicit support information for one chat capability.
    #[must_use]
    pub fn chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Extend the builder with explicit support information for many capabilities.
    #[must_use]
    pub fn chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = (ChatCapability, CapabilitySupport)>,
    {
        self.chat_capabilities.extend(capabilities);
        self
    }

    /// Mark each listed capability as supported.
    #[must_use]
    pub fn supported_chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = ChatCapability>,
    {
        for capability in capabilities {
            self.chat_capabilities
                .insert(capability, CapabilitySupport::Supported);
        }
        self
    }

    /// Override the provider name reported by the built streaming mock provider.
    #[must_use]
    pub fn provider_name(mut self, provider_name: &'static str) -> Self {
        self.provider_name = provider_name;
        self
    }

    /// Build the configured [`MockStreamingProvider`].
    #[must_use]
    pub fn build(self) -> MockStreamingProvider {
        MockStreamingProvider::new(self.streams)
            .with_chat_capabilities(self.chat_capabilities)
            .with_provider_name(self.provider_name)
    }
}

#[derive(Debug, Default)]
struct MockStreamingState {
    streams: VecDeque<VecDeque<MockStreamEvent>>,
    requests: Vec<ChatRequest>,
}

impl MockStreamingState {
    fn take_next_stream(&mut self, request: &ChatRequest) -> VecDeque<MockStreamEvent> {
        self.requests.push(request.clone());
        self.streams
            .pop_front()
            .expect("MockStreamingProvider: no more streams configured")
    }
}

/// Lightweight builder for common [`ChatResponse`] test fixtures.
///
/// This is intentionally small and opinionated: it covers the most common
/// fixture shapes without hiding the underlying [`ChatResponse`] structure.
#[derive(Debug, Clone)]
pub struct ChatResponseBuilder {
    content: Vec<ContentBlock>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    model: Option<String>,
    id: Option<String>,
    metadata: ResponseMetadata,
}

impl ChatResponseBuilder {
    /// Create an empty response builder with `FinishReason::Stop` by default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            content: Vec::new(),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: None,
            id: None,
            metadata: ResponseMetadata::new(),
        }
    }

    /// Append one text block to the response.
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.content.push(ContentBlock::Text { text: text.into() });
        self
    }

    /// Append one tool-call block to the response.
    #[must_use]
    pub fn tool_call(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        self.content.push(ContentBlock::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        });
        self
    }

    /// Append one reasoning block without a signature.
    #[must_use]
    pub fn reasoning(mut self, text: impl Into<String>) -> Self {
        self.content.push(ContentBlock::Reasoning {
            text: text.into(),
            signature: None,
        });
        self
    }

    /// Append one reasoning block with a provider-specific signature.
    #[must_use]
    pub fn reasoning_with_signature(
        mut self,
        text: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        self.content.push(ContentBlock::Reasoning {
            text: text.into(),
            signature: Some(signature.into()),
        });
        self
    }

    /// Append one provider-specific `Other` content block.
    #[must_use]
    pub fn other(mut self, type_name: impl Into<String>, data: ExtraMap) -> Self {
        self.content.push(ContentBlock::Other {
            type_name: type_name.into(),
            data,
        });
        self
    }

    /// Override the response finish reason.
    #[must_use]
    pub fn finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }

    /// Set basic input/output usage counts.
    #[must_use]
    pub fn usage(mut self, input_tokens: u64, output_tokens: u64) -> Self {
        self.usage = Some(Usage {
            input_tokens: Some(input_tokens),
            output_tokens: Some(output_tokens),
            ..Default::default()
        });
        self
    }

    /// Set the full usage value explicitly.
    #[must_use]
    pub fn usage_value(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set the provider-reported model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the provider-assigned response identifier.
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

    /// Build the final [`ChatResponse`].
    #[must_use]
    pub fn build(self) -> ChatResponse {
        ChatResponse {
            content: self.content,
            finish_reason: self.finish_reason,
            usage: self.usage,
            model: self.model,
            id: self.id,
            metadata: self.metadata,
        }
    }
}

impl Default for ChatResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn build_tool_call_response(
    id: impl Into<String>,
    name: impl Into<String>,
    arguments: serde_json::Value,
) -> ChatResponse {
    ChatResponseBuilder::new()
        .tool_call(id, name, arguments)
        .finish_reason(FinishReason::ToolCalls)
        .build()
}

fn response_to_mock_stream(response: ChatResponse) -> Vec<MockStreamEvent> {
    response
        .stream_events()
        .map(MockStreamEvent::Event)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::{ChatRequest, Message, ResponseMetadataType, StreamBlockType};

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
    struct DemoMetadata {
        request_id: String,
    }

    impl ResponseMetadataType for DemoMetadata {
        const KEY: &'static str = "demo";
    }

    fn assert_recorded_request(request: &ChatRequest, model: &str, message: &str) {
        assert_eq!(request.model, model);
        assert_eq!(request.messages, vec![Message::user(message)]);
    }

    #[tokio::test]
    async fn mock_chat_provider_records_requests_and_returns_responses() {
        let provider = MockProvider::new([
            ChatResponseBuilder::new().text("first").build(),
            ChatResponseBuilder::new().text("second").build(),
        ]);

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("first".into())
        );
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("second".into())
        );
        assert_eq!(provider.call_count(), 2);
        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        assert_recorded_request(&requests[0], "mock-model", "hi");
        assert_recorded_request(&requests[1], "mock-model", "hi");
        assert_recorded_request(&provider.last_request().unwrap(), "mock-model", "hi");
        assert_eq!(provider.pending_responses(), 0);
    }

    #[tokio::test]
    async fn mock_chat_provider_empty_can_be_queued_incrementally() {
        let provider = MockProvider::empty();
        provider.push_response(MockResponse::text("first"));
        provider.push_response(ChatResponseBuilder::new().text("second").build());

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        assert_eq!(provider.pending_responses(), 2);
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("first".into())
        );
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("second".into())
        );
        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        assert_recorded_request(&requests[0], "mock-model", "hi");
        assert_recorded_request(&requests[1], "mock-model", "hi");
    }

    #[tokio::test]
    async fn mock_chat_provider_supports_delayed_errors() {
        let provider = MockProvider::new([MockResponse::delayed(
            Duration::from_millis(1),
            Error::Timeout("slow".into()),
        )]);

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));
        let error = provider.chat(&request).await.unwrap_err();
        assert!(matches!(error, Error::Timeout(message) if message == "slow"));
    }

    #[test]
    fn mock_response_tool_call_sets_tool_finish_reason() {
        let response =
            MockResponse::tool_call("call_1", "search", serde_json::json!({ "q": "rust" }));

        match response {
            MockResponse::Success(response) => {
                assert!(response.has_tool_calls());
                assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
            }
            other => panic!("expected success response, got {other:?}"),
        }
    }

    #[test]
    fn mock_response_reasoning_text_builds_reasoning_and_text() {
        let response = MockResponse::reasoning_text("thinking", "done");

        match response {
            MockResponse::Success(response) => {
                assert_eq!(response.reasoning_text(), Some("thinking".into()));
                assert_eq!(response.text(), Some("done".into()));
                assert_eq!(response.finish_reason, Some(FinishReason::Stop));
            }
            other => panic!("expected success response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mock_chat_provider_tool_round_trip_returns_tool_then_text() {
        let provider = MockProvider::tool_round_trip(
            "call_1",
            "lookup_weather",
            serde_json::json!({ "city": "San Francisco" }),
            "Cool and foggy.",
        );

        let request = ChatRequest::new("mock-model").message(Message::user("weather?"));
        let first = provider.chat(&request).await.unwrap();
        let second = provider.chat(&request).await.unwrap();

        assert!(first.has_tool_calls());
        assert_eq!(first.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(second.text(), Some("Cool and foggy.".into()));
    }

    #[tokio::test]
    async fn mock_chat_provider_from_tool_round_trips_flattens_multiple_turns() {
        let provider = MockProvider::from_tool_round_trips([
            MockToolRoundTrip::single_tool_call_text(
                "call_weather_1",
                "lookup_weather",
                serde_json::json!({ "city": "San Francisco" }),
                "Weather answer",
            ),
            MockToolRoundTrip::single_tool_call_text(
                "call_time_1",
                "lookup_time",
                serde_json::json!({ "city": "London" }),
                "Time answer",
            ),
        ]);

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        assert!(provider.chat(&request).await.unwrap().has_tool_calls());
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("Weather answer".into())
        );
        assert!(provider.chat(&request).await.unwrap().has_tool_calls());
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("Time answer".into())
        );
    }

    #[tokio::test]
    async fn mock_streaming_provider_collects_streams() {
        let provider = MockStreamingProvider::new([vec![
            StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock-model".into()),
            }
            .into(),
            StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            }
            .into(),
            MockStreamEvent::delayed(Duration::from_millis(1)),
            StreamEvent::TextDelta {
                index: 0,
                text: "hello".into(),
            }
            .into(),
            StreamEvent::BlockStop { index: 0 }.into(),
            StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    input_tokens: Some(3),
                    output_tokens: Some(1),
                    ..Default::default()
                }),
                usage_mode: crate::UsageMetadataMode::Snapshot,
                id: Some("resp_1".into()),
                model: Some("mock-model".into()),
                metadata: crate::ExtraMap::new(),
            }
            .into(),
            StreamEvent::ResponseStop.into(),
        ]]);

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));
        let response = provider.chat(&request).await.unwrap();

        assert_eq!(response.text(), Some("hello".into()));
        assert_eq!(provider.call_count(), 1);
        let requests = provider.requests();
        assert_eq!(requests.len(), 1);
        assert_recorded_request(&requests[0], "mock-model", "hi");
        assert_recorded_request(&provider.last_request().unwrap(), "mock-model", "hi");
        assert_eq!(provider.pending_streams(), 0);
    }

    #[tokio::test]
    async fn mock_streaming_provider_empty_can_be_queued_incrementally() {
        let provider = MockStreamingProvider::empty();
        provider.push_stream(MockStreamEvent::text_response("first"));
        provider.push_stream(MockStreamEvent::text_response("second"));

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        assert_eq!(provider.pending_streams(), 2);
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("first".into())
        );
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("second".into())
        );
        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        assert_recorded_request(&requests[0], "mock-model", "hi");
        assert_recorded_request(&requests[1], "mock-model", "hi");
    }

    #[tokio::test]
    async fn mock_stream_event_reasoning_response_builds_collectable_transcript() {
        let provider = MockStreamingProvider::with_stream(MockStreamEvent::reasoning_response(
            "thinking",
            "streamed hello",
        ));

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));
        let response = provider.chat(&request).await.unwrap();

        assert_eq!(response.reasoning_text(), Some("thinking".into()));
        assert_eq!(response.text(), Some("streamed hello".into()));
    }

    #[tokio::test]
    async fn mock_stream_event_text_response_builds_collectable_transcript() {
        let provider =
            MockStreamingProvider::with_stream(MockStreamEvent::text_response("streamed hello"));

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));
        let response = provider.chat(&request).await.unwrap();

        assert_eq!(response.text(), Some("streamed hello".into()));
    }

    #[tokio::test]
    async fn mock_streaming_provider_with_tool_call_convenience_sets_tool_finish_reason() {
        let provider = MockStreamingProvider::with_tool_call(
            "call_1",
            "search",
            serde_json::json!({ "q": "rust" }),
        );

        let request = ChatRequest::new("mock-model").message(Message::user("search"));
        let response = provider.chat(&request).await.unwrap();

        assert!(response.has_tool_calls());
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn mock_chat_provider_chat_stream_propagates_chat_errors() {
        let provider = MockProvider::with_error(Error::Timeout("slow".to_string()));
        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        match provider.chat_stream(&request).await {
            Err(Error::Timeout(message)) => assert_eq!(message, "slow"),
            Err(other) => panic!("expected timeout error, got {other:?}"),
            Ok(_) => panic!("expected chat_stream to return an error"),
        }
    }

    #[tokio::test]
    async fn mock_streaming_provider_from_responses_replays_multiple_transcripts() {
        let provider = MockStreamingProvider::from_responses([
            ChatResponseBuilder::new().text("first").build(),
            ChatResponseBuilder::new().text("second").build(),
        ]);

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));

        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("first".into())
        );
        assert_eq!(
            provider.chat(&request).await.unwrap().text(),
            Some("second".into())
        );
    }

    #[tokio::test]
    async fn mock_streaming_provider_from_response_replays_normalized_stream() {
        let provider = MockStreamingProvider::from_response(
            ChatResponseBuilder::new()
                .reasoning("thinking")
                .text("done")
                .usage(7, 2)
                .model("mock-model")
                .id("resp_stream_1")
                .build(),
        );

        let request = ChatRequest::new("mock-model").message(Message::user("hi"));
        let response = provider.chat(&request).await.unwrap();

        assert_eq!(response.reasoning_text(), Some("thinking".into()));
        assert_eq!(response.text(), Some("done".into()));
        assert_eq!(response.id.as_deref(), Some("resp_stream_1"));
    }

    #[test]
    fn response_builder_builds_tool_calls_and_usage() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });

        let mut data = ExtraMap::new();
        data.insert("url".into(), serde_json::json!("https://example.com"));

        let response = ChatResponseBuilder::new()
            .text("Working on it. ")
            .reasoning_with_signature("Thinking", "sig_123")
            .tool_call("call_1", "search", serde_json::json!({ "q": "rust" }))
            .other("citation", data)
            .finish_reason(FinishReason::ToolCalls)
            .usage_value(Usage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                total_tokens: Some(15),
                reasoning_tokens: Some(2),
                ..Default::default()
            })
            .model("mock")
            .id("resp_1")
            .metadata(metadata)
            .build();

        assert_eq!(response.text(), Some("Working on it. ".into()));
        assert_eq!(response.reasoning_text(), Some("Thinking".into()));
        assert!(response.has_tool_calls());
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.usage.unwrap().input_tokens, Some(10));
        assert!(matches!(
            response.content[1],
            ContentBlock::Reasoning { ref signature, .. } if signature.as_deref() == Some("sig_123")
        ));
        assert!(matches!(
            response.content[3],
            ContentBlock::Other { ref type_name, ref data }
                if type_name == "citation"
                    && data.get("url") == Some(&serde_json::json!("https://example.com"))
        ));
        assert_eq!(
            response.metadata.get::<DemoMetadata>(),
            Some(&DemoMetadata {
                request_id: "req_123".into()
            })
        );
    }
}
