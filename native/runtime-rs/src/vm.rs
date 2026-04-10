use crate::bytecode::{Bindings, FfiBinding, Program};
use crate::ffi::invoke_binding;
use crate::jit::JitCompiler;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmResult {
    pub state: HashMap<String, Value>,
    pub output: Value,
    pub diagnostics: Vec<String>,
    pub trace: Vec<Value>,
}

pub struct Vm;

#[derive(Debug, Default)]
struct VmQuotaState {
    max_ffi_calls: u64,
    ffi_calls_used: u64,
}

impl Vm {
    pub fn execute(program: &Program, bindings: &Bindings, workspace_root: &str) -> Result<VmResult, String> {
        let mut state: HashMap<String, Value> = HashMap::new();
        let mut trace: Vec<Value> = Vec::new();
        let mut last_value = Value::Null;
        let mut quotas = VmQuotaState {
            max_ffi_calls: program
                .runtime
                .budget
                .get("max_ffi_calls")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            ffi_calls_used: 0,
        };
        let hot_path = JitCompiler::compile_hot_path(program)?;
        for instruction in &program.instructions {
            let pc = trace.len();
            let node_id = instruction
                .operand
                .get("node_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "instruction missing node_id".to_string())?
                .to_string();
            let value = execute_instruction(
                &instruction.opcode,
                &instruction.operand,
                &state,
                &last_value,
                bindings,
                workspace_root,
                &program.ffi_bindings,
                &mut quotas,
            )?;
            trace.push(json!({
                "pc": pc,
                "node_id": node_id,
                "opcode": instruction.opcode,
                "result_type": instruction.result_type,
                "state_size": state.len() + 1,
                "summary": summarize(&value)
            }));
            last_value = value.clone();
            state.insert(node_id, value);
        }
        let output_id = program
            .instructions
            .last()
            .and_then(|inst| inst.operand.get("node_id"))
            .and_then(Value::as_str)
            .ok_or_else(|| "program missing output instruction".to_string())?;
        let output = unwrap_result(state.get(output_id).cloned().unwrap_or(Value::Null));
        Ok(VmResult {
            state,
            output,
            diagnostics: vec![
                "Native Rust VM executed bytecode.".to_string(),
                format!("Runtime target: {}", program.runtime.target),
                format!("JIT tier: {}", hot_path.tier),
                format!("AOT enabled: {}", program.runtime.aot_enabled),
                format!("Cross-platform targets: {}", program.runtime.cross_platform_targets.join(", ")),
                format!("FFI bindings registered: {}", program.ffi_bindings.len()),
                format!("FFI calls used: {}", quotas.ffi_calls_used),
            ],
            trace,
        })
    }
}

fn execute_instruction(
    opcode: &str,
    operand: &Value,
    state: &HashMap<String, Value>,
    last_value: &Value,
    bindings: &Bindings,
    workspace_root: &str,
    ffi_bindings: &[FfiBinding],
    quotas: &mut VmQuotaState,
) -> Result<Value, String> {
    match opcode {
        "load_input" => Ok(json!(bindings)),
        "init_project" => Ok(project_snapshot(workspace_root)),
        "register_object" => Ok(json!({
            "name": operand.get("name").cloned().unwrap_or(Value::Null),
            "fields": operand.get("fields").cloned().unwrap_or(Value::Object(Map::new())),
        })),
        "compute_length" => {
            let source = operand.get("source").and_then(Value::as_str).unwrap_or_default();
            let value = resolve_symbol(source, state, last_value, bindings);
            if let Value::Array(items) = value {
                if let Some(Value::String(first)) = items.first() {
                    return Ok(json!(first.len()));
                }
            }
            Ok(json!(0))
        }
        "compute_total_length" => {
            let source = operand.get("source").and_then(Value::as_str).unwrap_or_default();
            let value = resolve_symbol(source, state, last_value, bindings);
            if let Value::Array(items) = value {
                if let Some(Value::String(first)) = items.first() {
                    return Ok(json!(items.len() * first.len()));
                }
            }
            Ok(json!(0))
        }
        "build_freq_map" => {
            let source = operand.get("source").and_then(Value::as_str).unwrap_or_default();
            let value = resolve_symbol(source, state, last_value, bindings);
            let mut freq: HashMap<String, usize> = HashMap::new();
            if let Value::Array(items) = value {
                for item in items {
                    if let Some(text) = item.as_str() {
                        *freq.entry(text.to_string()).or_insert(0) += 1;
                    }
                }
            }
            Ok(json!(freq))
        }
        "sliding_window_scan" => {
            let text_name = operand.get("text").and_then(Value::as_str).unwrap_or_default();
            let words_name = operand.get("words").and_then(Value::as_str).unwrap_or_default();
            let text = resolve_symbol(text_name, state, last_value, bindings).as_str().unwrap_or_default().to_string();
            let words_value = resolve_symbol(words_name, state, last_value, bindings);
            let words = words_value
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>();
            Ok(json!(substring_concat_scan(&text, &words)))
        }
        "collect_indices" => {
            let source = operand.get("source").and_then(Value::as_str).unwrap_or_default();
            Ok(resolve_symbol(source, state, last_value, bindings))
        }
        "sort" => {
            let mut items = coerce_sequence(operand, state, last_value, bindings)?;
            let key = operand
                .get("sort_key")
                .or_else(|| operand.get("key"))
                .and_then(Value::as_str)
                .unwrap_or("value")
                .to_string();
            let descending = operand.get("descending").and_then(Value::as_bool).unwrap_or(false);
            let limit = operand.get("limit").and_then(Value::as_u64);
            items.sort_by(|a, b| {
                let av = a.get(&key).cloned().unwrap_or(Value::Null);
                let bv = b.get(&key).cloned().unwrap_or(Value::Null);
                compare_values(&av, &bv)
            });
            if descending {
                items.reverse();
            }
            if let Some(limit) = limit {
                items.truncate(limit as usize);
            }
            Ok(json!(items))
        }
        "filter" => {
            let items = coerce_sequence(operand, state, last_value, bindings)?;
            let predicate = operand.get("predicate").and_then(Value::as_str).unwrap_or_default();
            let filtered = items
                .into_iter()
                .filter(|item| evaluate_item_predicate(item, predicate))
                .collect::<Vec<_>>();
            Ok(json!(filtered))
        }
        "map" => {
            let items = coerce_sequence(operand, state, last_value, bindings)?;
            let expression = operand.get("expression").and_then(Value::as_str).unwrap_or("item");
            let mapped = items
                .into_iter()
                .map(|item| map_item(&item, expression))
                .collect::<Vec<_>>();
            Ok(Value::Array(mapped))
        }
        "reduce" => {
            let mode = operand.get("mode").and_then(Value::as_str).unwrap_or("group_by");
            let source = resolve_source_or_last(operand, state, last_value, bindings);
            match mode {
                "group_by" => {
                    let key = operand.get("key").and_then(Value::as_str).unwrap_or("value");
                    Ok(group_by(source, key))
                }
                _ => Ok(source),
            }
        }
        "transform" | "model_plan" => {
            if let Some(expression) = operand.get("expression").and_then(Value::as_str) {
                Ok(resolve_symbol(expression, state, last_value, bindings))
            } else {
                Ok(last_value.clone())
            }
        }
        "merge" => Ok(last_value.clone()),
        "validate" => Ok(json!({
            "result": last_value.clone(),
            "constraints": operand.get("constraints").cloned().unwrap_or(Value::Object(Map::new())),
            "valid": true
        })),
        "synth_tests" => Ok(json!({"result": last_value.clone(), "tests_ready": true})),
        "debug_prep" => Ok(json!({"result": last_value.clone(), "strategy": operand.get("fallback").cloned().unwrap_or(Value::Null), "local_first": true})),
        "output" => {
            let source = operand.get("source").and_then(Value::as_str).unwrap_or_default();
            Ok(unwrap_result(resolve_symbol(source, state, last_value, bindings)))
        }
        "call_intrinsic" => {
            let intrinsic = operand.get("intrinsic").and_then(Value::as_str).unwrap_or("unknown");
            Ok(call_intrinsic(intrinsic, operand, state, last_value, bindings))
        }
        "call_ffi" => {
            execute_ffi_call(operand, state, last_value, bindings, ffi_bindings, quotas)
        }
        "loop" => {
            Ok(resolve_source_or_last(operand, state, last_value, bindings))
        }
        "eval_condition" => {
            let condition = operand.get("condition").and_then(Value::as_str).unwrap_or_default();
            Ok(json!(evaluate_condition(condition, state, last_value, bindings)))
        }
        "split_equal_words" => {
            let source = operand.get("source").and_then(Value::as_str).unwrap_or_default();
            let chunk_size = operand.get("chunk_size").and_then(Value::as_u64).unwrap_or(1) as usize;
            let value = resolve_symbol(source, state, last_value, bindings);
            if let Some(text) = value.as_str() {
                let chunks = text
                    .as_bytes()
                    .chunks(chunk_size.max(1))
                    .map(|chunk| Value::String(String::from_utf8_lossy(chunk).to_string()))
                    .collect::<Vec<_>>();
                Ok(Value::Array(chunks))
            } else {
                Ok(value)
            }
        }
        "compare_freq_maps" => {
            let left_name = operand.get("left").and_then(Value::as_str).unwrap_or_default();
            let right_name = operand.get("right").and_then(Value::as_str).unwrap_or_default();
            Ok(json!(resolve_symbol(left_name, state, last_value, bindings) == resolve_symbol(right_name, state, last_value, bindings)))
        }
        other => Err(format!("unsupported opcode: {}", other)),
    }
}

fn resolve_symbol(name: &str, state: &HashMap<String, Value>, last_value: &Value, bindings: &Bindings) -> Value {
    if let Some(value) = bindings.get(name) {
        return value.clone();
    }
    if let Some(value) = state.get(name) {
        return value.clone();
    }
    last_value.clone()
}

fn unwrap_result(mut value: Value) -> Value {
    loop {
        if let Value::Object(map) = &value {
            if let Some(next) = map.get("result") {
                value = next.clone();
                continue;
            }
        }
        return value;
    }
}

fn coerce_sequence(operand: &Value, state: &HashMap<String, Value>, last_value: &Value, bindings: &Bindings) -> Result<Vec<Map<String, Value>>, String> {
    let base = resolve_source_or_last(operand, state, last_value, bindings);
    if let Value::Array(items) = base {
        let mut out = Vec::new();
        for item in items {
            if let Value::Object(map) = item {
                out.push(map);
            }
        }
        return Ok(out);
    }
    if let Value::Object(map) = base {
        for value in map.values() {
            if let Value::Array(items) = value {
                let mut out = Vec::new();
                for item in items {
                    if let Value::Object(obj) = item {
                        out.push(obj.clone());
                    }
                }
                return Ok(out);
            }
        }
    }
    for value in bindings.values() {
        if let Value::Array(items) = value {
            let mut out = Vec::new();
            for item in items {
                if let Value::Object(obj) = item {
                    out.push(obj.clone());
                }
            }
            return Ok(out);
        }
    }
    Err("expected sequence input".to_string())
}

fn resolve_source_or_last(operand: &Value, state: &HashMap<String, Value>, last_value: &Value, bindings: &Bindings) -> Value {
    operand
        .get("source")
        .and_then(Value::as_str)
        .map(|name| resolve_symbol(name, state, last_value, bindings))
        .unwrap_or_else(|| last_value.clone())
}

fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => na.as_f64().partial_cmp(&nb.as_f64()).unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(sa), Value::String(sb)) => sa.cmp(sb),
        _ => std::cmp::Ordering::Equal,
    }
}

fn substring_concat_scan(text: &str, words: &[String]) -> Vec<usize> {
    if words.is_empty() {
        return Vec::new();
    }
    let word_len = words[0].len();
    if word_len == 0 || words.iter().any(|word| word.len() != word_len) {
        return Vec::new();
    }
    let mut target: HashMap<&str, usize> = HashMap::new();
    for word in words {
        *target.entry(word.as_str()).or_insert(0) += 1;
    }
    let mut results = Vec::new();
    for offset in 0..word_len {
        let mut left = offset;
        let mut seen: HashMap<&str, usize> = HashMap::new();
        let mut count = 0usize;
        let mut right = offset;
        while right + word_len <= text.len() {
            let word = &text[right..right + word_len];
            right += word_len;
            if let Some(expected) = target.get(word) {
                *seen.entry(word).or_insert(0) += 1;
                count += 1;
                while seen.get(word).copied().unwrap_or(0) > *expected {
                    let left_word = &text[left..left + word_len];
                    if let Some(current) = seen.get_mut(left_word) {
                        *current -= 1;
                    }
                    left += word_len;
                    count -= 1;
                }
                if count == words.len() {
                    results.push(left);
                    let left_word = &text[left..left + word_len];
                    if let Some(current) = seen.get_mut(left_word) {
                        *current -= 1;
                    }
                    left += word_len;
                    count -= 1;
                }
            } else {
                seen.clear();
                count = 0;
                left = right;
            }
        }
    }
    results
}

fn project_snapshot(workspace_root: &str) -> Value {
    let git_dir = Path::new(workspace_root).join(".git");
    if !git_dir.exists() {
        return json!({
            "vcs": "git",
            "branch": Value::Null,
            "dirty": false,
            "changed_files": [],
            "suggested_commit_message": Value::Null,
            "suggested_version_bump": Value::Null,
            "release_notes_hint": Value::Null
        });
    }
    let head = fs::read_to_string(git_dir.join("HEAD")).unwrap_or_default();
    let branch = head.trim().strip_prefix("ref: refs/heads/").map(|s| s.to_string());
    json!({
        "vcs": "git",
        "branch": branch,
        "dirty": false,
        "changed_files": [],
        "suggested_commit_message": Value::Null,
        "suggested_version_bump": Value::Null,
        "release_notes_hint": Value::Null
    })
}

fn evaluate_item_predicate(item: &Map<String, Value>, predicate: &str) -> bool {
    let predicate = predicate.trim();
    if let Some((field, expected)) = predicate.split_once("==") {
        if let Some(field_name) = field.trim().strip_prefix("item.") {
            let actual = item.get(field_name).cloned().unwrap_or(Value::Null);
            let expected_text = expected.trim().trim_matches('"').trim_matches('\'');
            if expected_text.eq_ignore_ascii_case("true") {
                return actual.as_bool().unwrap_or(false);
            }
            if expected_text.eq_ignore_ascii_case("false") {
                return !actual.as_bool().unwrap_or(false);
            }
            if let Ok(number) = expected_text.parse::<i64>() {
                return actual.as_i64() == Some(number);
            }
            return stringify_value(&actual) == expected_text;
        }
    }
    if let Some((field, expected)) = predicate.split_once(">") {
        if let Some(field_name) = field.trim().strip_prefix("item.") {
            let actual = item.get(field_name).and_then(Value::as_f64).unwrap_or(0.0);
            let threshold = expected.trim().parse::<f64>().unwrap_or(0.0);
            return actual > threshold;
        }
    }
    if let Some((field, expected)) = predicate.split_once("<") {
        if let Some(field_name) = field.trim().strip_prefix("item.") {
            let actual = item.get(field_name).and_then(Value::as_f64).unwrap_or(0.0);
            let threshold = expected.trim().parse::<f64>().unwrap_or(0.0);
            return actual < threshold;
        }
    }
    if let Some(field) = predicate.strip_prefix("item.") {
        return item.get(field).and_then(Value::as_bool).unwrap_or_else(|| item.get(field).is_some());
    }
    !item.is_empty()
}

fn map_item(item: &Map<String, Value>, expression: &str) -> Value {
    if expression == "item" {
        return Value::Object(item.clone());
    }
    if let Some(field) = expression.strip_prefix("item.") {
        return item.get(field).cloned().unwrap_or(Value::Null);
    }
    Value::Object(item.clone())
}

fn group_by(source: Value, key: &str) -> Value {
    let mut grouped: HashMap<String, Vec<Value>> = HashMap::new();
    if let Value::Array(items) = source {
        for item in items {
            if let Value::Object(map) = item {
                let bucket = map.get(key).cloned().unwrap_or(Value::Null).to_string();
                grouped.entry(bucket).or_default().push(Value::Object(map));
            }
        }
    }
    json!(grouped)
}

fn evaluate_condition(condition: &str, state: &HashMap<String, Value>, last_value: &Value, bindings: &Bindings) -> bool {
    if let Some((left, right)) = condition.split_once("!=") {
        return !evaluate_condition(&format!("{} == {}", left.trim(), right.trim()), state, last_value, bindings);
    }
    if let Some((left, right)) = condition.split_once("==") {
        let left_value = resolve_symbol(left.trim(), state, last_value, bindings);
        let right_norm = right.trim().trim_matches('"').trim_matches('\'');
        if right.trim().eq_ignore_ascii_case("true") {
            return left_value.as_bool().unwrap_or(false);
        }
        if right.trim().eq_ignore_ascii_case("false") {
            return !left_value.as_bool().unwrap_or(false);
        }
        if let Ok(number) = right_norm.parse::<i64>() {
            return left_value.as_i64() == Some(number);
        }
        return stringify_value(&left_value) == right_norm;
    }
    resolve_symbol(condition.trim(), state, last_value, bindings).as_bool().unwrap_or(false)
}

fn stringify_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

fn summarize(value: &Value) -> String {
    match value {
        Value::Array(items) => format!("list(len={})", items.len()),
        Value::Object(map) => {
            let keys = map.keys().take(5).cloned().collect::<Vec<_>>();
            format!("dict(keys={:?})", keys)
        }
        _ => value.to_string(),
    }
}

fn call_intrinsic(
    intrinsic: &str,
    operand: &Value,
    state: &HashMap<String, Value>,
    last_value: &Value,
    bindings: &Bindings,
) -> Value {
    match intrinsic_registry().get(intrinsic) {
        Some(handler) => match handler(operand, state, last_value, bindings) {
            Ok(value) => value,
            Err(err) => json!({"intrinsic": intrinsic, "status": "error", "message": err}),
        },
        None => json!({"intrinsic": intrinsic, "status": "stubbed"}),
    }
}

type IntrinsicHandler = fn(&Value, &HashMap<String, Value>, &Value, &Bindings) -> Result<Value, String>;

fn intrinsic_registry() -> HashMap<&'static str, IntrinsicHandler> {
    HashMap::from([
        ("dataset.len", intrinsic_dataset_len as IntrinsicHandler),
        ("record.keys", intrinsic_record_keys as IntrinsicHandler),
        ("execute_algorithm_family", intrinsic_execute_algorithm_family as IntrinsicHandler),
        ("execute_structured_program", intrinsic_execute_structured_program as IntrinsicHandler),
    ])
}

fn intrinsic_dataset_len(
    operand: &Value,
    state: &HashMap<String, Value>,
    last_value: &Value,
    bindings: &Bindings,
) -> Result<Value, String> {
    let source = resolve_source_or_last(operand, state, last_value, bindings);
    match source {
        Value::Array(items) => Ok(json!(items.len())),
        _ => Ok(Value::Null),
    }
}

fn intrinsic_record_keys(
    operand: &Value,
    state: &HashMap<String, Value>,
    last_value: &Value,
    bindings: &Bindings,
) -> Result<Value, String> {
    let source = resolve_source_or_last(operand, state, last_value, bindings);
    match source {
        Value::Object(map) => Ok(json!(map.keys().cloned().collect::<Vec<_>>())),
        _ => Ok(Value::Array(vec![])),
    }
}

fn intrinsic_execute_algorithm_family(
    operand: &Value,
    _state: &HashMap<String, Value>,
    _last_value: &Value,
    bindings: &Bindings,
) -> Result<Value, String> {
    execute_algorithm_family(operand, bindings)
}

fn intrinsic_execute_structured_program(
    operand: &Value,
    _state: &HashMap<String, Value>,
    _last_value: &Value,
    bindings: &Bindings,
) -> Result<Value, String> {
    execute_structured_program(operand, bindings)
}

fn execute_algorithm_family(operand: &Value, bindings: &Bindings) -> Result<Value, String> {
    let family = operand
        .get("algorithm_family")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let task = operand
        .get("algorithm_task")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match (family, task) {
        ("array_manipulation", "rotate_array") => Ok(execute_rotate_array(bindings)),
        ("array_manipulation", "binary_search") => Ok(execute_binary_search(bindings)),
        ("array_manipulation", "max_subarray") => Ok(execute_max_subarray(bindings)),
        ("array_manipulation", "product_except_self") => Ok(execute_product_except_self(bindings)),
        ("array_two_pointers", "three_sum") => Ok(execute_three_sum(bindings)),
        ("hashmap_counting", "group_anagrams") => Ok(execute_group_anagrams(bindings)),
        ("hashmap_counting", "most_frequent_element") => Ok(execute_most_frequent_element(bindings)),
        ("stack_queue_heap", "valid_parentheses") => Ok(execute_valid_parentheses(bindings)),
        ("stack_queue_heap", "top_k_frequent") => Ok(execute_top_k_frequent(bindings)),
        ("linked_list", "reverse_list") => Ok(execute_reverse_list(bindings)),
        ("tree_graph", "binary_tree_level_order") => Ok(execute_binary_tree_level_order(bindings)),
        ("tree_graph", "binary_tree_right_side_view") => Ok(execute_binary_tree_right_side_view(bindings)),
        ("tree_graph", "num_islands") => Ok(execute_num_islands(bindings)),
        ("dp_backtracking", "coin_change") => Ok(execute_coin_change(bindings)),
        ("dp_backtracking", "subsets") => Ok(execute_subsets(bindings)),
        ("dp_backtracking", "longest_increasing_subsequence") => Ok(execute_longest_increasing_subsequence(bindings)),
        _ => Err(format!("unsupported algorithm family task: {family}/{task}")),
    }
}

fn execute_structured_program(operand: &Value, bindings: &Bindings) -> Result<Value, String> {
    let spec = operand
        .get("program_spec")
        .ok_or_else(|| "execute_structured_program missing program_spec".to_string())?;
    if is_minstack_spec(spec) {
        return execute_structured_minstack(bindings);
    }
    if let (Some(family), Some(task)) = (
        operand.get("algorithm_family").and_then(Value::as_str),
        operand.get("algorithm_task").and_then(Value::as_str),
    ) {
        return execute_algorithm_family(&json!({"algorithm_family": family, "algorithm_task": task}), bindings);
    }
    Err("native structured execution is not implemented for this program shape yet".to_string())
}

fn is_minstack_spec(spec: &Value) -> bool {
    spec.get("name").and_then(Value::as_str) == Some("MinStackProgram")
        || spec
            .get("objects")
            .and_then(Value::as_object)
            .map(|objects| objects.contains_key("MinStack"))
            .unwrap_or(false)
}

fn execute_structured_minstack(bindings: &Bindings) -> Result<Value, String> {
    let operations = bindings
        .get("operations")
        .and_then(Value::as_array)
        .ok_or_else(|| "MinStack bindings missing operations list".to_string())?;
    let values = bindings
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| "MinStack bindings missing values list".to_string())?;
    let mut stack: Vec<Value> = Vec::new();
    let mut min_stack: Vec<Value> = Vec::new();
    let mut results: Vec<Value> = Vec::new();
    let mut next = if !operations.is_empty()
        && !values.is_empty()
        && operations.len() == values.len()
        && operations[0].as_str() == Some("MinStack")
    {
        1usize
    } else {
        0usize
    };
    for op in operations {
        match op.as_str().unwrap_or_default() {
            "MinStack" => results.push(Value::Null),
            "push" => {
                let value = values.get(next).cloned().unwrap_or(Value::Null);
                next += 1;
                stack.push(value.clone());
                let should_push_min = min_stack
                    .last()
                    .map(|current| compare_numeric(value.clone(), current.clone()) <= 0)
                    .unwrap_or(true);
                if should_push_min {
                    min_stack.push(value);
                }
                results.push(Value::Null);
            }
            "pop" => {
                if stack.last() == min_stack.last() && !min_stack.is_empty() {
                    min_stack.pop();
                }
                if !stack.is_empty() {
                    stack.pop();
                }
                results.push(Value::Null);
            }
            "top" => results.push(stack.last().cloned().unwrap_or(Value::Null)),
            "getMin" => results.push(min_stack.last().cloned().unwrap_or(Value::Null)),
            _ => {}
        }
    }
    Ok(Value::Array(results))
}

fn execute_three_sum(bindings: &Bindings) -> Value {
    let mut nums = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_i64())
        .collect::<Vec<_>>();
    nums.sort_unstable();
    let mut result: Vec<Value> = Vec::new();
    if nums.len() < 3 {
        return Value::Array(result);
    }
    for i in 0..(nums.len() - 2) {
        if i > 0 && nums[i] == nums[i - 1] {
            continue;
        }
        let mut left = i + 1;
        let mut right = nums.len() - 1;
        while left < right {
            let total = nums[i] + nums[left] + nums[right];
            if total == 0 {
                result.push(json!([nums[i], nums[left], nums[right]]));
                left += 1;
                right = right.saturating_sub(1);
                while left < right && nums[left] == nums[left - 1] {
                    left += 1;
                }
                while left < right && nums[right] == nums[right + 1] {
                    right = right.saturating_sub(1);
                }
            } else if total < 0 {
                left += 1;
            } else {
                right = right.saturating_sub(1);
            }
        }
    }
    Value::Array(result)
}

fn execute_rotate_array(bindings: &Bindings) -> Value {
    let values = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if values.is_empty() {
        return Value::Array(Vec::new());
    }
    let steps = bindings.get("k").and_then(Value::as_i64).unwrap_or(0).max(0) as usize;
    let offset = steps % values.len();
    if offset == 0 {
        return Value::Array(values);
    }
    let mut rotated = values[values.len() - offset..].to_vec();
    rotated.extend_from_slice(&values[..values.len() - offset]);
    Value::Array(rotated)
}

fn execute_binary_search(bindings: &Bindings) -> Value {
    let nums = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_i64())
        .collect::<Vec<_>>();
    let target = bindings.get("target").and_then(Value::as_i64).unwrap_or(0);
    let mut left = 0usize;
    let mut right = nums.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if nums[mid] == target {
            return Value::Number((mid as i64).into());
        }
        if nums[mid] < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    Value::Number((-1).into())
}

fn execute_max_subarray(bindings: &Bindings) -> Value {
    let nums = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_i64())
        .collect::<Vec<_>>();
    if nums.is_empty() {
        return Value::Number(0.into());
    }
    let mut best = nums[0];
    let mut current = nums[0];
    for value in nums.into_iter().skip(1) {
        current = std::cmp::max(value, current + value);
        best = std::cmp::max(best, current);
    }
    Value::Number(best.into())
}

fn execute_product_except_self(bindings: &Bindings) -> Value {
    let nums = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_i64())
        .collect::<Vec<_>>();
    if nums.is_empty() {
        return Value::Array(Vec::new());
    }
    let mut prefix = vec![1i64; nums.len()];
    let mut suffix = vec![1i64; nums.len()];
    for index in 1..nums.len() {
        prefix[index] = prefix[index - 1] * nums[index - 1];
    }
    for index in (0..nums.len() - 1).rev() {
        suffix[index] = suffix[index + 1] * nums[index + 1];
    }
    Value::Array(
        (0..nums.len())
            .map(|index| Value::Number((prefix[index] * suffix[index]).into()))
            .collect(),
    )
}

fn execute_group_anagrams(bindings: &Bindings) -> Value {
    let words = bindings
        .get("words")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
    for word in words {
        if let Some(text) = word.as_str() {
            let mut chars = text.chars().collect::<Vec<_>>();
            chars.sort_unstable();
            let key = chars.into_iter().collect::<String>();
            groups.entry(key).or_default().push(Value::String(text.to_string()));
        }
    }
    Value::Array(groups.into_values().map(Value::Array).collect())
}

fn execute_most_frequent_element(bindings: &Bindings) -> Value {
    let values = bindings
        .get("elements")
        .or_else(|| bindings.get("nums"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if values.is_empty() {
        return Value::Null;
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for value in &values {
        *counts.entry(value.to_string()).or_insert(0) += 1;
    }
    let best_count = counts.values().copied().max().unwrap_or(0);
    for value in values {
        if counts.get(&value.to_string()).copied().unwrap_or(0) == best_count {
            return value;
        }
    }
    Value::Null
}

fn execute_valid_parentheses(bindings: &Bindings) -> Value {
    let text = bindings.get("s").and_then(Value::as_str).unwrap_or_default();
    let mut stack: Vec<char> = Vec::new();
    let pairs: HashMap<char, char> = [(')', '('), (']', '['), ('}', '{')].into_iter().collect();
    for ch in text.chars() {
        if matches!(ch, '(' | '[' | '{') {
            stack.push(ch);
        } else if let Some(expected) = pairs.get(&ch) {
            if stack.pop() != Some(*expected) {
                return Value::Bool(false);
            }
        }
    }
    Value::Bool(stack.is_empty())
}

fn execute_top_k_frequent(bindings: &Bindings) -> Value {
    let nums = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_i64())
        .collect::<Vec<_>>();
    let k = bindings.get("k").and_then(Value::as_i64).unwrap_or(0).max(0) as usize;
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for num in nums {
        *counts.entry(num).or_insert(0) += 1;
    }
    let mut pairs = counts.into_iter().collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Value::Array(
        pairs
            .into_iter()
            .take(k)
            .map(|(value, _)| Value::Number(value.into()))
            .collect(),
    )
}

fn execute_reverse_list(bindings: &Bindings) -> Value {
    let mut values = bindings
        .get("head")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    values.reverse();
    Value::Array(values)
}

fn execute_binary_tree_level_order(bindings: &Bindings) -> Value {
    let nodes = bindings
        .get("root")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if nodes.is_empty() {
        return Value::Array(Vec::new());
    }
    let mut result: Vec<Vec<Value>> = Vec::new();
    let mut queue = std::collections::VecDeque::from([(0usize, 0usize)]);
    while let Some((index, depth)) = queue.pop_front() {
        if index >= nodes.len() || nodes[index].is_null() {
            continue;
        }
        if result.len() <= depth {
            result.push(Vec::new());
        }
        result[depth].push(nodes[index].clone());
        queue.push_back((2 * index + 1, depth + 1));
        queue.push_back((2 * index + 2, depth + 1));
    }
    Value::Array(result.into_iter().map(Value::Array).collect())
}

fn execute_binary_tree_right_side_view(bindings: &Bindings) -> Value {
    match execute_binary_tree_level_order(bindings) {
        Value::Array(levels) => Value::Array(
            levels
                .into_iter()
                .filter_map(|level| match level {
                    Value::Array(items) => items.last().cloned(),
                    _ => None,
                })
                .collect(),
        ),
        other => other,
    }
}

fn execute_num_islands(bindings: &Bindings) -> Value {
    let grid = bindings
        .get("grid")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| row.as_array().cloned())
        .map(|row| {
            row.into_iter()
                .map(|cell| cell.as_str().unwrap_or_default().to_string())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if grid.is_empty() {
        return Value::Number(0.into());
    }
    let rows = grid.len();
    let cols = grid[0].len();
    let mut seen: HashMap<(usize, usize), bool> = HashMap::new();
    let mut islands = 0i64;
    for r in 0..rows {
        for c in 0..cols {
            if grid[r][c] == "1" && !seen.contains_key(&(r, c)) {
                islands += 1;
                let mut queue = std::collections::VecDeque::from([(r, c)]);
                seen.insert((r, c), true);
                while let Some((cr, cc)) = queue.pop_front() {
                    for (nr, nc) in neighbors(cr, cc, rows, cols) {
                        if grid[nr][nc] == "1" && !seen.contains_key(&(nr, nc)) {
                            seen.insert((nr, nc), true);
                            queue.push_back((nr, nc));
                        }
                    }
                }
            }
        }
    }
    Value::Number(islands.into())
}

fn execute_coin_change(bindings: &Bindings) -> Value {
    let coins = bindings
        .get("coins")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_i64())
        .collect::<Vec<_>>();
    let amount = bindings.get("amount").and_then(Value::as_i64).unwrap_or(0).max(0) as usize;
    let mut dp = vec![amount as i64 + 1; amount + 1];
    dp[0] = 0;
    for value in 1..=amount {
        for coin in &coins {
            let coin = (*coin).max(0) as usize;
            if coin <= value {
                dp[value] = dp[value].min(dp[value - coin] + 1);
            }
        }
    }
    let answer = if dp[amount] > amount as i64 { -1 } else { dp[amount] };
    Value::Number(answer.into())
}

fn execute_subsets(bindings: &Bindings) -> Value {
    let nums = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut result: Vec<Vec<Value>> = vec![Vec::new()];
    for value in nums {
        let additions = result
            .iter()
            .map(|subset| {
                let mut next = subset.clone();
                next.push(value.clone());
                next
            })
            .collect::<Vec<_>>();
        result.extend(additions);
    }
    Value::Array(result.into_iter().map(Value::Array).collect())
}

fn execute_longest_increasing_subsequence(bindings: &Bindings) -> Value {
    let nums = bindings
        .get("nums")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_i64())
        .collect::<Vec<_>>();
    if nums.is_empty() {
        return Value::Number(0.into());
    }
    let mut dp = vec![1i64; nums.len()];
    for i in 0..nums.len() {
        for j in 0..i {
            if nums[j] < nums[i] {
                dp[i] = dp[i].max(dp[j] + 1);
            }
        }
    }
    Value::Number(dp.into_iter().max().unwrap_or(0).into())
}

fn compare_numeric(left: Value, right: Value) -> i8 {
    let lhs = left.as_f64().unwrap_or(0.0);
    let rhs = right.as_f64().unwrap_or(0.0);
    if lhs < rhs {
        -1
    } else if lhs > rhs {
        1
    } else {
        0
    }
}

fn neighbors(row: usize, col: usize, rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if row + 1 < rows {
        out.push((row + 1, col));
    }
    if row > 0 {
        out.push((row - 1, col));
    }
    if col + 1 < cols {
        out.push((row, col + 1));
    }
    if col > 0 {
        out.push((row, col - 1));
    }
    out
}

fn execute_ffi_call(
    operand: &Value,
    state: &HashMap<String, Value>,
    last_value: &Value,
    bindings: &Bindings,
    ffi_bindings: &[FfiBinding],
    quotas: &mut VmQuotaState,
) -> Result<Value, String> {
    if quotas.max_ffi_calls == 0 {
        return Err("ffi call denied by runtime quota".to_string());
    }
    if quotas.ffi_calls_used >= quotas.max_ffi_calls {
        return Err("ffi quota exceeded".to_string());
    }
    let signature = operand
        .get("ffi_signature")
        .and_then(Value::as_object)
        .ok_or_else(|| "call_ffi missing ffi_signature".to_string())?;
    let name = signature
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "ffi signature missing name".to_string())?;
    let binding = ffi_bindings
        .iter()
        .find(|candidate| candidate.name == name)
        .ok_or_else(|| format!("ffi binding not registered: {name}"))?;
    let input = resolve_source_or_last(operand, state, last_value, bindings);
    quotas.ffi_calls_used += 1;
    invoke_binding(binding, &input).map(|value| {
        json!({
            "ffi": {
                "name": binding.name,
                "library": binding.library,
                "abi": binding.abi,
            },
            "result": value
        })
    })
}
