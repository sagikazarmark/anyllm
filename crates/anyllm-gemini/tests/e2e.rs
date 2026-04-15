use anyllm_conformance::e2e;
use anyllm_gemini::Provider;

fn make_provider() -> Provider {
    let api_key =
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set to run e2e tests");
    Provider::new(api_key).expect("failed to create Gemini provider")
}

fn model() -> String {
    std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".into())
}

fn embedding_model() -> String {
    std::env::var("GEMINI_EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-004".into())
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

#[cfg(feature = "extract")]
#[tokio::test]
#[ignore]
async fn extract() {
    e2e::extract(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn basic_embed() {
    e2e::basic_embed(&make_provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn batch_embed() {
    e2e::batch_embed(&make_provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn dimensions() {
    e2e::dimensions(&make_provider(), &embedding_model(), 32).await;
}
