use anyllm_conformance::e2e;
use anyllm_openai_compat::Provider;

fn make_provider() -> Provider {
    let base_url = std::env::var("OPENAI_COMPAT_BASE_URL")
        .expect("OPENAI_COMPAT_BASE_URL must be set to run e2e tests");
    let api_key = std::env::var("OPENAI_COMPAT_API_KEY")
        .expect("OPENAI_COMPAT_API_KEY must be set to run e2e tests");

    Provider::builder()
        .base_url(base_url)
        .bearer_token(api_key)
        .provider_name("openai_compat")
        .build()
        .expect("failed to create OpenAI-compatible provider")
}

fn model() -> String {
    std::env::var("OPENAI_COMPAT_MODEL").expect("OPENAI_COMPAT_MODEL must be set to run e2e tests")
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
