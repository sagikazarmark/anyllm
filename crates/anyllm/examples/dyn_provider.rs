use anyllm::ProviderIdentity;
use anyllm::prelude::*;

struct StaticProvider;

impl ProviderIdentity for StaticProvider {
    fn provider_name(&self) -> &'static str {
        "static-demo"
    }
}

impl ChatProvider for StaticProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> anyllm::Result<ChatResponse> {
        Ok(ChatResponse::new(vec![ContentBlock::Text {
            text: format!("hello from {}", request.model),
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
            ChatCapability::NativeStreaming => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }
}

async fn run(provider: &DynChatProvider, request: &ChatRequest) -> anyllm::Result<()> {
    let response = provider.chat(request).await?;
    println!("dyn chat text: {}", response.text_or_empty());

    let streamed = provider
        .chat_stream(request)
        .await?
        .collect_response()
        .await?;
    println!("dyn stream text: {}", streamed.text_or_empty());

    Ok(())
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let request = ChatRequest::new("demo-model")
        .system("You are concise.")
        .user("Say hello");

    let provider: DynChatProvider = DynChatProvider::new(StaticProvider);

    run(&provider, &request).await
}
