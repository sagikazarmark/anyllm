use anyllm::{ExtraMap, Tool, ToolChoice};
use serde_json::json;

#[test]
fn tool_serialization_uses_description_and_extensions_fields() {
    let tool = Tool::new(
        "search",
        json!({
            "type": "object",
            "properties": {
                "q": {"type": "string"}
            },
            "required": ["q"]
        }),
    )
    .description("Search docs")
    .with_extension("cache_control", json!({"type": "ephemeral"}));

    let expected_extensions = Some(ExtraMap::from_iter([(
        "cache_control".into(),
        json!({"type": "ephemeral"}),
    )]));
    assert_eq!(tool.extensions, expected_extensions);

    let value = serde_json::to_value(&tool).unwrap();
    assert_eq!(
        value,
        json!({
            "name": "search",
            "description": "Search docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"}
                },
                "required": ["q"]
            },
            "extensions": {
                "cache_control": {"type": "ephemeral"}
            }
        })
    );
}

#[test]
fn tool_choice_serialization_is_stable() {
    assert_eq!(
        serde_json::to_value(ToolChoice::Auto).unwrap(),
        json!("auto")
    );
    assert_eq!(
        serde_json::to_value(ToolChoice::Disabled).unwrap(),
        json!("disabled")
    );
    assert_eq!(
        serde_json::to_value(ToolChoice::Required).unwrap(),
        json!("required")
    );
    assert_eq!(
        serde_json::to_value(ToolChoice::Specific {
            name: "search".into(),
        })
        .unwrap(),
        json!({"specific": {"name": "search"}})
    );
}
