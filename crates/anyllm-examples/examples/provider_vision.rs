//! Vision example: feed an image URL to the model and ask about it.
//!
//! Works against any of the configured providers; the default models
//! (gpt-4o, claude-sonnet-4-20250514, gemini-2.5-pro) all accept image
//! input.

use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};

const DEFAULT_IMAGE_URL: &str = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png";

#[tokio::main]
async fn main() -> Result<()> {
    let image_url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_IMAGE_URL.into());

    let target = load_provider_for_example("provider_vision", "[image-url]")?;
    print_provider_banner(&target);

    let request = ChatRequest::new(&target.model).message(Message::user_multimodal(vec![
        ContentPart::text("Describe what you see in this image in one sentence."),
        ContentPart::image_url(image_url),
    ]));

    let response = target.provider.chat(&request).await?;

    if let Some(finish_reason) = &response.finish_reason {
        eprintln!("finish_reason={finish_reason:?}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    println!("{}", response.text_or_empty());
    Ok(())
}
