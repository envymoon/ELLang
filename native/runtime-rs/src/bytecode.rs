use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: String,
    pub operand: serde_json::Value,
    #[serde(default)]
    pub result_type: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeMetadata {
    #[serde(default = "default_target")]
    pub target: String,
    #[serde(default = "default_jit_tier")]
    pub jit_tier: String,
    #[serde(default = "default_true")]
    pub aot_enabled: bool,
    #[serde(default = "default_hot_threshold")]
    pub hot_threshold: u64,
    #[serde(default)]
    pub cross_platform_targets: Vec<String>,
    #[serde(default)]
    pub budget: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FfiBinding {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub library: String,
    #[serde(default = "default_abi")]
    pub abi: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub program_name: String,
    pub instructions: Vec<Instruction>,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub runtime: RuntimeMetadata,
    #[serde(default)]
    pub ffi_bindings: Vec<FfiBinding>,
    #[serde(default)]
    pub exported_types: serde_json::Value,
}

pub type Bindings = HashMap<String, serde_json::Value>;

fn default_target() -> String {
    "native".to_string()
}

fn default_jit_tier() -> String {
    "baseline".to_string()
}

fn default_true() -> bool {
    true
}

fn default_hot_threshold() -> u64 {
    8
}

fn default_abi() -> String {
    "system".to_string()
}
