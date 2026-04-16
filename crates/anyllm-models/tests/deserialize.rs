use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("api.json")
}

#[test]
fn deserializes_fixture() {
    let json = std::fs::read_to_string(fixture_path()).unwrap();
    let registry: anyllm_models::Registry = anyllm_models::from_str(&json).unwrap();

    assert_eq!(registry.len(), 2);
    assert!(registry.contains_key("test-provider-full"));
    assert!(registry.contains_key("test-provider-minimal"));
}

#[test]
fn provider_fields() {
    let json = std::fs::read_to_string(fixture_path()).unwrap();
    let registry: anyllm_models::Registry = anyllm_models::from_str(&json).unwrap();

    let provider = &registry["test-provider-full"];
    assert_eq!(provider.id, "test-provider-full");
    assert_eq!(provider.name, "Test Provider Full");
    assert_eq!(provider.env, vec!["TEST_API_KEY"]);
    assert_eq!(provider.npm, "@test/sdk");
    assert_eq!(provider.api.as_deref(), Some("https://api.test.example/v1"));
    assert_eq!(provider.doc, "https://docs.test.example");
    assert_eq!(provider.models.len(), 2);
}

#[test]
fn model_with_all_optional_fields() {
    let json = std::fs::read_to_string(fixture_path()).unwrap();
    let registry: anyllm_models::Registry = anyllm_models::from_str(&json).unwrap();

    let model = &registry["test-provider-full"].models["model-complete"];
    assert_eq!(model.id, "model-complete");
    assert_eq!(model.name, "Model Complete");
    assert_eq!(model.family.as_deref(), Some("test-family"));
    assert!(model.attachment);
    assert!(model.reasoning);
    assert!(model.tool_call);
    assert_eq!(model.temperature, Some(true));
    assert_eq!(model.structured_output, Some(true));
    assert_eq!(model.knowledge.as_deref(), Some("2025-01"));
    assert_eq!(model.release_date.as_deref(), Some("2025-03-15"));
    assert_eq!(model.last_updated.as_deref(), Some("2025-04-01"));
    assert!(!model.open_weights);

    // Modalities
    assert_eq!(model.modalities.input, vec!["text", "image", "audio"]);
    assert_eq!(model.modalities.output, vec!["text", "audio"]);

    // Costs
    let cost = model.cost.as_ref().expect("model-complete should have cost");
    assert_eq!(cost.input, 2.5);
    assert_eq!(cost.output, 10.0);
    assert_eq!(cost.reasoning, Some(15.0));
    assert_eq!(cost.cache_read, Some(1.25));
    assert_eq!(cost.cache_write, Some(3.75));
    assert_eq!(cost.input_audio, Some(40.0));
    assert_eq!(cost.output_audio, Some(80.0));

    // Limits
    assert_eq!(model.limit.context, 200000);
    assert_eq!(model.limit.output, 32768);
    assert_eq!(model.limit.input, Some(180000));
}

#[test]
fn model_with_missing_optional_fields() {
    let json = std::fs::read_to_string(fixture_path()).unwrap();
    let registry: anyllm_models::Registry = anyllm_models::from_str(&json).unwrap();

    let model = &registry["test-provider-full"].models["model-sparse"];
    assert_eq!(model.id, "model-sparse");
    assert_eq!(model.family, None);
    assert!(!model.attachment);
    assert!(!model.reasoning);
    assert!(!model.tool_call);
    assert_eq!(model.structured_output, None);
    assert_eq!(model.knowledge, None);
    assert_eq!(model.release_date, None);
    assert_eq!(model.last_updated, None);
    assert!(model.open_weights);

    // Cost optional fields absent
    let cost = model.cost.as_ref().expect("model-sparse should have cost");
    assert_eq!(cost.reasoning, None);
    assert_eq!(cost.cache_read, None);
    assert_eq!(cost.cache_write, None);
    assert_eq!(cost.input_audio, None);
    assert_eq!(cost.output_audio, None);

    // Limit optional field absent
    assert_eq!(model.limit.input, None);
}

#[test]
fn from_slice_works() {
    let bytes = std::fs::read(fixture_path()).unwrap();
    let registry: anyllm_models::Registry = anyllm_models::from_slice(&bytes).unwrap();
    assert_eq!(registry.len(), 2);
}

#[test]
fn from_reader_works() {
    let file = std::fs::File::open(fixture_path()).unwrap();
    let registry: anyllm_models::Registry = anyllm_models::from_reader(file).unwrap();
    assert_eq!(registry.len(), 2);
}
