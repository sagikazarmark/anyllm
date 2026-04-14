use anyllm_conformance::e2e;
use anyllm_openai_compat::Provider;
use anyllm_openai_compat::providers::Cloudflare;

fn make_provider() -> Provider {
    Cloudflare::from_env().expect("failed to create Cloudflare provider")
}

fn model() -> String {
    std::env::var("CLOUDFLARE_MODEL").unwrap_or_else(|_| "@cf/meta/llama-3.1-8b-instruct".into())
}

#[tokio::test]
#[ignore]
async fn basic_chat() {
    e2e::basic_chat(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn streaming() {
    e2e::streaming(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn system_prompt() {
    e2e::system_prompt(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn multi_turn() {
    e2e::multi_turn(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_calling() {
    e2e::tool_calling(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn structured_output() {
    e2e::structured_output(&make_provider(), &model()).await;
}

#[cfg(feature = "extract")]
#[tokio::test]
#[ignore]
async fn extract() {
    e2e::extract(&make_provider(), &model()).await;
}
