use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct RequestSummary {
    summary: String,
    mentions_rust: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "Rust ownership and borrowing help keep agent state updates safe.".into()
    });

    let target = load_provider_for_example("provider_extract", "[prompt]")?;
    print_provider_banner(&target);

    let request = ChatRequest::new(&target.model)
        .system("Extract a concise summary and whether the request mentions Rust.")
        .user(prompt);

    let extracted: Extracted<RequestSummary> = target.provider.extract(&request).await?;

    eprintln!(
        "passes={} repaired={}",
        extracted.metadata.passes, extracted.metadata.repaired
    );
    eprintln!("raw_text={}", extracted.response.text_or_empty());
    println!(
        "{}",
        serde_json::to_string_pretty(&extracted.value).map_err(Error::from)?
    );
    Ok(())
}
