use std::time::Duration;

use anyllm::Error;
use serde_json::json;

#[test]
fn error_log_contract_is_stable_for_structured_variants() {
    let rate_limited = Error::RateLimited {
        message: "too many requests".into(),
        retry_after: Some(Duration::from_secs(5)),
        request_id: Some("req_1".into()),
    };
    assert_eq!(
        serde_json::to_value(rate_limited.as_log()).unwrap(),
        json!({
            "type": "rate_limited",
            "message": "too many requests",
            "retry_after_secs": 5.0,
            "request_id": "req_1"
        })
    );

    let overloaded = Error::Overloaded {
        message: "degraded".into(),
        retry_after: Some(Duration::from_millis(2500)),
        request_id: Some("req_2".into()),
    };
    assert_eq!(
        serde_json::to_value(overloaded.as_log()).unwrap(),
        json!({
            "type": "overloaded",
            "message": "degraded",
            "retry_after_secs": 2.5,
            "request_id": "req_2"
        })
    );

    let provider = Error::Provider {
        status: Some(503),
        message: "service unavailable".into(),
        body: Some("upstream".into()),
        request_id: Some("req_3".into()),
    };
    assert_eq!(
        serde_json::to_value(provider.as_log()).unwrap(),
        json!({
            "type": "provider",
            "status": 503,
            "message": "service unavailable",
            "body_present": true,
            "body_len": 8,
            "request_id": "req_3"
        })
    );

    let context = Error::ContextLengthExceeded {
        message: "too long".into(),
        max_tokens: Some(128000),
    };
    assert_eq!(
        serde_json::to_value(context.as_log()).unwrap(),
        json!({
            "type": "context_length_exceeded",
            "message": "too long",
            "max_tokens": 128000
        })
    );
}

#[test]
fn error_serde_round_trips_public_variants() {
    let cases = vec![
        Error::Provider {
            status: None,
            message: "connection refused".into(),
            body: None,
            request_id: None,
        },
        Error::Timeout("30s elapsed".into()),
        Error::Auth("bad key".into()),
        Error::RateLimited {
            message: "too many requests".into(),
            retry_after: Some(Duration::from_secs(5)),
            request_id: Some("req_1".into()),
        },
        Error::Overloaded {
            message: "degraded".into(),
            retry_after: Some(Duration::from_millis(2500)),
            request_id: Some("req_2".into()),
        },
        Error::Provider {
            status: Some(503),
            message: "service unavailable".into(),
            body: Some("upstream payload".into()),
            request_id: Some("req_3".into()),
        },
        Error::serialization("bad json"),
        Error::Unsupported("vision unavailable".into()),
        Error::InvalidRequest("missing model".into()),
        Error::UnexpectedResponse("missing text block".into()),
        Error::ContextLengthExceeded {
            message: "too long".into(),
            max_tokens: Some(128000),
        },
        Error::ContentFiltered("blocked".into()),
        Error::Stream("unexpected EOF".into()),
    ];

    for error in cases {
        let json = serde_json::to_value(&error).unwrap();
        let round_tripped: Error = serde_json::from_value(json).unwrap();
        assert_eq!(error.to_string(), round_tripped.to_string());
        assert_eq!(
            serde_json::to_value(error.as_log()).unwrap(),
            serde_json::to_value(round_tripped.as_log()).unwrap()
        );
    }
}

#[test]
fn error_serde_uses_flat_human_friendly_shape() {
    let rate_limited = Error::RateLimited {
        message: "too many requests".into(),
        retry_after: Some(Duration::from_millis(2500)),
        request_id: Some("req_1".into()),
    };
    assert_eq!(
        serde_json::to_value(&rate_limited).unwrap(),
        json!({
            "type": "rate_limited",
            "message": "too many requests",
            "retry_after_secs": 2.5,
            "request_id": "req_1"
        })
    );

    let provider = Error::Provider {
        status: Some(503),
        message: "service unavailable".into(),
        body: Some("upstream payload".into()),
        request_id: Some("req_3".into()),
    };
    assert_eq!(
        serde_json::to_value(&provider).unwrap(),
        json!({
            "type": "provider",
            "status": 503,
            "message": "service unavailable",
            "body": "upstream payload",
            "request_id": "req_3"
        })
    );
}

#[test]
fn error_serde_preserves_provider_body() {
    let error = Error::Provider {
        status: Some(500),
        message: "internal error".into(),
        body: Some("sensitive but explicit".into()),
        request_id: Some("req_9".into()),
    };

    let value = serde_json::to_value(&error).unwrap();
    assert_eq!(value["body"], "sensitive but explicit");

    let round_tripped: Error = serde_json::from_value(value).unwrap();
    match round_tripped {
        Error::Provider { body, .. } => {
            assert_eq!(body.as_deref(), Some("sensitive but explicit"));
        }
        other => panic!("expected provider error, got {other:?}"),
    }
}
