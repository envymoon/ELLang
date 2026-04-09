use ellang_runtime::bytecode::{Bindings, Program};
use ellang_runtime::vm::Vm;
use std::env;
use std::fs;

fn main() {
    if let Err(err) = run() {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 4 {
        return Err("usage: ellang-runtime <bytecode.json> <bindings.json> <workspace_root>".to_string());
    }
    let program: Program = serde_json::from_str(&fs::read_to_string(&args[1]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let bindings: Bindings = serde_json::from_str(&fs::read_to_string(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let result = Vm::execute(&program, &bindings, &args[3])?;
    println!("{}", serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?);
    Ok(())
}
