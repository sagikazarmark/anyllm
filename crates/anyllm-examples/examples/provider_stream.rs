use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};

#[tokio::main]
async fn main() -> Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Explain how streaming helps agent UIs.".into());

    let target = load_provider_for_example("provider_stream", "[prompt]")?;
    print_provider_banner(&target);

    let request = ChatRequest::new(&target.model).user(prompt);
    let mut stream = target.provider.chat_stream(&request).await?;
    let mut collector = StreamCollector::new();

    while let Some(event) = stream.next().await {
        let event = event?;

        match &event {
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                eprint!("[reasoning:{}]", text.replace('\n', "\\n"));
            }
            StreamEvent::BlockStart {
                block_type: StreamBlockType::ToolCall,
                index,
                name,
                ..
            } => {
                eprintln!(
                    "\n[tool_call_start index={index} name={}]",
                    name.as_deref().unwrap_or("?")
                );
            }
            StreamEvent::ResponseMetadata {
                finish_reason,
                usage,
                ..
            } => {
                eprintln!("\n[metadata finish_reason={finish_reason:?} usage={usage:?}]");
            }
            StreamEvent::ResponseStop => {
                eprintln!("\n[response_stop]");
            }
            _ => {}
        }

        collector.push(event)?;
    }

    let response = collector.finish()?;
    eprintln!("\n[final_text_len={}]", response.text_or_empty().len());
    Ok(())
}
