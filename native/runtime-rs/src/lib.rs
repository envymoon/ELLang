pub mod bytecode;
pub mod capabilities;
pub mod ffi;
pub mod jit;
pub mod quotas;
pub mod vm;

pub fn runtime_banner() -> &'static str {
    "ELLang native runtime with typed bytecode, FFI, and AOT/JIT metadata"
}
