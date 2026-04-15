use anyllm::prelude::*;
use anyllm_examples::{load_embedding_provider_for_example, print_embedding_provider_banner};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let inputs: Vec<String> = if args.is_empty() {
        vec![
            "Rust makes ownership explicit.".into(),
            "Embedding models turn text into vectors.".into(),
        ]
    } else {
        args
    };

    let target = load_embedding_provider_for_example("provider_embedding", "[input...]")?;
    print_embedding_provider_banner(&target);

    let request = EmbeddingRequest::new(&target.model).inputs(inputs.iter().cloned());
    let response = target.provider.embed(&request).await?;

    for (index, (input, vector)) in inputs.iter().zip(response.embeddings.iter()).enumerate() {
        let preview: Vec<f32> = vector.iter().take(4).copied().collect();
        let suffix = if vector.len() > preview.len() {
            ", ..."
        } else {
            ""
        };
        println!(
            "[{index}] dims={:>4} input={:?} head={:.4?}{}",
            vector.len(),
            input,
            preview,
            suffix,
        );
    }

    if let Some(model) = &response.model {
        eprintln!("response model={model}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    Ok(())
}
