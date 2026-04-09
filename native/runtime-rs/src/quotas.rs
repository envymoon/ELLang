use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeQuota {
    #[serde(default)]
    pub max_tokens: u64,
    #[serde(default)]
    pub max_vram_mb: u64,
    #[serde(default)]
    pub max_cpu_ms: u64,
    #[serde(default)]
    pub max_wall_ms: u64,
    #[serde(default)]
    pub max_writes: u64,
    #[serde(default)]
    pub max_network_calls: u64,
    #[serde(default)]
    pub max_ffi_calls: u64,
}
