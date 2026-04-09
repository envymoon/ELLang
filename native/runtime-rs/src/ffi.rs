use crate::bytecode::FfiBinding;
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn describe_bindings(bindings: &[FfiBinding]) -> Value {
    Value::Array(
        bindings
            .iter()
            .map(|binding| {
                json!({
                    "name": binding.name,
                    "library": binding.library,
                    "abi": binding.abi,
                    "required_capabilities": binding.required_capabilities
                })
            })
            .collect(),
    )
}

type FfiHandler = fn(&Value) -> Result<Value, String>;

pub fn invoke_binding(binding: &FfiBinding, input: &Value) -> Result<Value, String> {
    if !matches!(binding.abi.as_str(), "system" | "prototype" | "registry") {
        return Err(format!("unsupported ffi abi: {}", binding.abi));
    }
    if !binding.required_capabilities.iter().any(|cap| cap == "ffi.call") {
        return Err(format!("ffi binding {} is missing ffi.call capability", binding.name));
    }
    let registry = ffi_registry();
    let Some(handler) = registry.get(binding.name.as_str()) else {
        return Err(format!("unknown ffi binding: {}", binding.name));
    };
    handler(input)
}

fn ffi_registry() -> HashMap<&'static str, FfiHandler> {
    HashMap::from([
        ("ellang_native.identity", ffi_identity as FfiHandler),
        ("ellang_native.keys", ffi_keys as FfiHandler),
        ("ellang_native.len", ffi_len as FfiHandler),
        ("ellang_native.uppercase", ffi_uppercase as FfiHandler),
    ])
}

fn ffi_identity(input: &Value) -> Result<Value, String> {
    Ok(input.clone())
}

fn ffi_keys(input: &Value) -> Result<Value, String> {
    match input {
        Value::Object(map) => Ok(Value::Array(map.keys().cloned().map(Value::String).collect())),
        _ => Err("ellang_native.keys expects a record".to_string()),
    }
}

fn ffi_len(input: &Value) -> Result<Value, String> {
    let size = match input {
        Value::Array(items) => items.len() as i64,
        Value::Object(map) => map.len() as i64,
        Value::String(text) => text.len() as i64,
        _ => return Err("ellang_native.len expects a list, record, or string".to_string()),
    };
    Ok(Value::Number(size.into()))
}

fn ffi_uppercase(input: &Value) -> Result<Value, String> {
    let Some(text) = input.as_str() else {
        return Err("ellang_native.uppercase expects a string".to_string());
    };
    Ok(Value::String(text.to_uppercase()))
}
