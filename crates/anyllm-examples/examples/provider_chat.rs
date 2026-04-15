use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};

#[tokio::main]
async fn main() -> Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Explain Rust ownership in one short paragraph.".into());

    let target = load_provider_for_example("provider_chat", "[prompt]")?;
    print_provider_banner(&target);

    let response = target
        .provider
        .chat(&ChatRequest::new(&target.model).user(prompt))
        .await?;

    if let Some(finish_reason) = &response.finish_reason {
        eprintln!("finish_reason={finish_reason:?}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    println!("{}", response.text_or_empty());
    Ok(())
}
