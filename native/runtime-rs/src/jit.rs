use crate::bytecode::Program;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HotPathPlan {
    pub hot_nodes: Vec<String>,
    pub tier: String,
    pub aot_enabled: bool,
}

pub struct JitCompiler;

impl JitCompiler {
    pub fn compile_hot_path(program: &Program) -> Result<HotPathPlan, String> {
        let threshold = program.runtime.hot_threshold.max(1) as usize;
        let hot_nodes = program
            .instructions
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx < threshold)
            .filter_map(|(_, inst)| inst.operand.get("node_id").and_then(|item| item.as_str()).map(|item| item.to_string()))
            .collect::<Vec<_>>();
        Ok(HotPathPlan {
            hot_nodes,
            tier: program.runtime.jit_tier.clone(),
            aot_enabled: program.runtime.aot_enabled,
        })
    }
}
