use anyllm::prelude::*;
use tracing_subscriber::FmtSubscriber;

fn init_tracing() {
    let subscriber = FmtSubscriber::builder()
        .with_target(false)
        .without_time()
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
}

fn build_provider() -> MockStreamingProvider {
    MockStreamingProvider::build(|builder| {
        builder
            .text("Traced hello from anyllm.")
            .text("Traced hello from anyllm.")
    })
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model")
        .system("You are concise.")
        .user("Say hello")
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    init_tracing();

    let provider =
        TracingChatProvider::new(build_provider()).with_content_capture(TracingContentConfig {
            capture_input_messages: true,
            capture_output_messages: true,
            max_payload_chars: 128,
        });

    let request = build_request();
    let response = provider.chat(&request).await?;
    println!("traced chat text: {}", response.text_or_empty());

    let streamed = provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await?;
    println!("traced stream text: {}", streamed.text_or_empty());

    Ok(())
}
