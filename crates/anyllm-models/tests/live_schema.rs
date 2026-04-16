/// Test that the structs can deserialize the full real models.dev API response.
///
/// This test requires a snapshot of the real API data. It is ignored by default
/// and intended to be run manually during development or CI with a fresh
/// download.
///
/// To run: download the API data first, then run with the env var set:
/// ```sh
/// curl -s https://models.dev/api.json -o /tmp/models-dev-api.json
/// MODELS_DEV_SNAPSHOT=/tmp/models-dev-api.json cargo test -p anyllm-models --test live_schema -- --ignored
/// ```
#[test]
#[ignore]
fn deserializes_real_api_snapshot() {
    let path = std::env::var("MODELS_DEV_SNAPSHOT")
        .expect("set MODELS_DEV_SNAPSHOT to the path of a models.dev/api.json download");

    let json = std::fs::read_to_string(&path).unwrap();
    let registry: anyllm_models::Registry = anyllm_models::from_str(&json).unwrap();

    // Sanity checks — the real registry has many providers
    assert!(
        registry.len() > 10,
        "expected many providers, got {}",
        registry.len()
    );

    // Spot-check a known provider
    let openai = registry
        .get("openai")
        .expect("openai should be in the registry");
    assert_eq!(openai.id, "openai");
    assert!(!openai.models.is_empty(), "openai should have models");
}
