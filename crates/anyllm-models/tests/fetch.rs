/// Integration test that fetches the real models.dev API.
///
/// Ignored by default — run manually:
/// ```sh
/// cargo test -p anyllm-models --features http --test fetch -- --ignored
/// ```
#[tokio::test]
#[ignore]
async fn fetch_live_registry() {
    let client = reqwest::Client::new();
    let options = anyllm_models::FetchOptions::default();

    let result = anyllm_models::fetch(&client, &options).await.unwrap();

    match result {
        anyllm_models::FetchResult::Modified {
            registry,
            etag,
            last_modified,
        } => {
            assert!(
                registry.len() > 10,
                "expected many providers, got {}",
                registry.len()
            );
            println!("fetched {} providers", registry.len());
            println!("etag: {:?}", etag);
            println!("last_modified: {:?}", last_modified);
        }
        anyllm_models::FetchResult::NotModified => {
            panic!("first fetch should never return NotModified");
        }
    }
}

/// Test conditional fetch with a previously obtained ETag.
///
/// Ignored by default — run manually:
/// ```sh
/// cargo test -p anyllm-models --features http --test fetch -- --ignored
/// ```
#[tokio::test]
#[ignore]
async fn conditional_fetch_with_etag() {
    let client = reqwest::Client::new();

    // First fetch to get validators
    let result = anyllm_models::fetch(&client, &anyllm_models::FetchOptions::default())
        .await
        .unwrap();

    let (etag, last_modified) = match result {
        anyllm_models::FetchResult::Modified {
            etag,
            last_modified,
            ..
        } => (etag, last_modified),
        _ => panic!("first fetch should return Modified"),
    };

    // Second fetch with validators — should get NotModified
    let options = anyllm_models::FetchOptions {
        etag,
        last_modified,
    };
    let result = anyllm_models::fetch(&client, &options).await.unwrap();

    // The server may or may not support conditional requests,
    // so we accept both outcomes
    match result {
        anyllm_models::FetchResult::NotModified => {
            println!("got NotModified as expected");
        }
        anyllm_models::FetchResult::Modified { registry, .. } => {
            println!(
                "server returned full response anyway ({} providers)",
                registry.len()
            );
        }
    }
}
