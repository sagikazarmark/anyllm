use anyllm::prelude::*;

struct StaticProvider;

impl ChatProvider for StaticProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> anyllm::Result<ChatResponse> {
        Ok(ChatResponse::new(vec![ContentBlock::Text {
            text: format!("hello from {}", self.provider_name()),
        }])
        .finish_reason(FinishReason::Stop)
        .model(request.model.clone()))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> anyllm::Result<Self::Stream> {
        Ok(self.chat(request).await?.into_stream())
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        match capability {
            ChatCapability::Streaming => CapabilitySupport::Supported,
            _ => CapabilitySupport::Unknown,
        }
    }

    fn provider_name(&self) -> &'static str {
        "static-demo"
    }
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = StaticProvider;
    let request = ChatRequest::new("demo-model")
        .system("You are concise.")
        .user("Say hello");

    let response = provider.chat(&request).await?;
    println!("chat text: {}", response.text_or_empty());
    println!(
        "streaming support: {:?}",
        provider.chat_capability(&request.model, ChatCapability::Streaming)
    );

    let streamed = provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await?;
    println!("stream text: {}", streamed.text_or_empty());

    Ok(())
}
