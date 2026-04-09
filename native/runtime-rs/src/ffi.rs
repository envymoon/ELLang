use crate::bytecode::FfiBinding;
use serde_json::{json, Value};

pub fn describe_bindings(bindings: &[FfiBinding]) -> Value {
    Value::Array(
        bindings
            .iter()
            .map(|binding| {
                json!({
                    "name": binding.name,
                    "library": binding.library,
                    "abi": binding.abi
                })
            })
            .collect(),
    )
}
