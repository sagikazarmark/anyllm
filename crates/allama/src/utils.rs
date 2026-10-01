use std::fmt;

use serde::de;

use crate::ExtraMap;

pub(crate) fn truncate_display(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }

    let end = s.floor_char_boundary(max_bytes);
    format!("{}...", &s[..end])
}

pub(crate) fn write_truncated(
    f: &mut fmt::Formatter<'_>,
    s: &str,
    max_bytes: usize,
) -> fmt::Result {
    if s.len() <= max_bytes {
        return f.write_str(s);
    }

    let end = s.floor_char_boundary(max_bytes);
    write!(f, "{}...", &s[..end])
}

pub(crate) fn write_truncated_joined<'a>(
    f: &mut fmt::Formatter<'_>,
    mut texts: impl Iterator<Item = &'a str>,
    empty: &str,
    max_bytes: usize,
) -> fmt::Result {
    let Some(first) = texts.next() else {
        return f.write_str(empty);
    };

    let mut remaining = max_bytes;
    let mut truncated = write_limited_fragment(f, first, &mut remaining)?;

    if !truncated {
        for text in texts {
            if write_limited_fragment(f, " ", &mut remaining)?
                || write_limited_fragment(f, text, &mut remaining)?
            {
                truncated = true;
                break;
            }
        }
    }

    if truncated {
        f.write_str("...")?;
    }

    Ok(())
}

pub(crate) fn concat_segments<'a>(mut segments: impl Iterator<Item = &'a str>) -> Option<String> {
    let first = segments.next()?;
    let mut joined = String::from(first);
    for segment in segments {
        joined.push_str(segment);
    }
    Some(joined)
}

pub(crate) fn write_segments<'a>(
    output: &mut String,
    mut segments: impl Iterator<Item = &'a str>,
) -> bool {
    let Some(first) = segments.next() else {
        return false;
    };

    output.push_str(first);
    for segment in segments {
        output.push_str(segment);
    }
    true
}

pub(crate) fn insert_if_some(map: &mut ExtraMap, key: &str, value: Option<serde_json::Value>) {
    if let Some(value) = value {
        map.insert(key.to_owned(), value);
    }
}

pub(crate) fn set_once<E: de::Error, T>(
    slot: &mut Option<T>,
    field: &'static str,
    value: T,
) -> Result<(), E> {
    if slot.is_some() {
        return Err(de::Error::duplicate_field(field));
    }

    *slot = Some(value);
    Ok(())
}

pub(crate) fn required_string_field<E: de::Error>(
    value: Option<serde_json::Value>,
    field: &'static str,
) -> Result<String, E> {
    match value {
        Some(serde_json::Value::String(s)) => Ok(s),
        Some(_) => Err(de::Error::custom(format!("\"{field}\" must be a string"))),
        None => Err(de::Error::missing_field(field)),
    }
}

pub(crate) fn required_string_field_in<E: de::Error>(
    value: Option<serde_json::Value>,
    field: &'static str,
    context: &'static str,
) -> Result<String, E> {
    match value {
        Some(serde_json::Value::String(s)) => Ok(s),
        Some(_) => Err(de::Error::custom(format!("\"{field}\" must be a string"))),
        None => Err(de::Error::custom(format!(
            "missing \"{field}\" field for {context}"
        ))),
    }
}

pub(crate) fn optional_string_field<E: de::Error>(
    value: Option<serde_json::Value>,
    field: &'static str,
) -> Result<Option<String>, E> {
    match value {
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(de::Error::custom(format!("\"{field}\" must be a string"))),
        None => Ok(None),
    }
}

pub(crate) fn optional_bool_field<E: de::Error>(
    value: Option<serde_json::Value>,
    field: &'static str,
) -> Result<Option<bool>, E> {
    match value {
        Some(serde_json::Value::Bool(value)) => Ok(Some(value)),
        Some(_) => Err(de::Error::custom(format!("\"{field}\" must be a boolean"))),
        None => Ok(None),
    }
}

fn write_limited_fragment(
    f: &mut fmt::Formatter<'_>,
    fragment: &str,
    remaining: &mut usize,
) -> Result<bool, fmt::Error> {
    if *remaining == 0 {
        return Ok(true);
    }

    if fragment.len() <= *remaining {
        f.write_str(fragment)?;
        *remaining -= fragment.len();
        return Ok(false);
    }

    let end = fragment.floor_char_boundary(*remaining);
    f.write_str(&fragment[..end])?;
    *remaining = 0;
    Ok(true)
}
