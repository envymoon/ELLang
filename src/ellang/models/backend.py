from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from ..cache import TypedIRCache, canonicalize_spec
from ..syntax import ActionStep, EmitStep, IfStep, LoopStep, ModuleCallStep, ProgramSpec
from ..typed_ir import Capability, GenericOperator, ResourceBudget, ValueType

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


@dataclass(slots=True)
class IntentPlan:
    task_family: str
    parameters: dict[str, Any] = field(default_factory=dict)
    operators: list[GenericOperator] = field(default_factory=list)
    deterministic: bool = False
    confidence: float = 0.0
    rationale: str = ""
    template_name: str = ""


@dataclass(slots=True)
class OperatorTemplate:
    name: str
    task_family: str
    match_any: tuple[str, ...]
    operators: tuple[GenericOperator, ...]
    deterministic: bool = True
    confidence: float = 0.0
    defaults: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(slots=True)
class BackendResult:
    typed_program_payload: dict[str, Any]
    diagnostics: list[str] = field(default_factory=list)


class LocalModelBackend(Protocol):
    def plan_typed_program(self, spec: ProgramSpec) -> BackendResult:
        ...


@dataclass(slots=True)
class QwenBackendConfig:
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4"
    quantization: str = "int4"
    max_context_tokens: int = 4096
    max_new_tokens: int = 384
    compile_cache_size: int = 128
    prefer_local_files_only: bool = True
    trust_remote_code: bool = False
    temperature: float = 0.0
    top_p: float = 0.9
    profile: str = "consumer"
    planner_version: str = "lazy-typed-ir-v1"
    classification_confidence_threshold: float = 0.55


MODEL_PROFILES: dict[str, list[tuple[str, str]]] = {
    "consumer": [
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4", "int4"),
        ("Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4", "int4"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "none"),
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "int8"),
    ],
    "midrange": [
        ("Qwen/Qwen2.5-Coder-3B-Instruct", "int8"),
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4", "int4"),
    ],
    "enhanced": [
        ("Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4", "int4"),
        ("Qwen/Qwen2.5-Coder-7B-Instruct", "int8"),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "int8"),
    ],
}


OPERATOR_TEMPLATES: tuple[OperatorTemplate, ...] = (
    OperatorTemplate(
        name="sort_topk",
        task_family="sorting",
        match_any=("sort", "rank", "top", "highest", "descending"),
        operators=(GenericOperator.SORT,),
        confidence=0.95,
        defaults={"sort_key": "value", "descending": True},
        rationale="stable sort/top-k template",
    ),
    OperatorTemplate(
        name="filter_map",
        task_family="filter_transform",
        match_any=("filter", "select", "keep", "transform", "project"),
        operators=(GenericOperator.FILTER, GenericOperator.MAP),
        confidence=0.86,
        defaults={"expression": "item"},
        rationale="stable dataset filter/map pipeline",
    ),
    OperatorTemplate(
        name="group_aggregate",
        task_family="aggregation",
        match_any=("group", "aggregate", "count", "frequency", "bucket"),
        operators=(GenericOperator.REDUCE,),
        confidence=0.84,
        defaults={"mode": "group_by", "key": "value"},
        rationale="stable aggregation template",
    ),
    OperatorTemplate(
        name="search_loop",
        task_family="search",
        match_any=("scan", "search", "find", "match", "window"),
        operators=(GenericOperator.LOOP, GenericOperator.FILTER, GenericOperator.MAP),
        confidence=0.78,
        defaults={"expression": "item"},
        rationale="stable search/scan template",
    ),
)


class QwenLocalBackend:
    def __init__(self, config: QwenBackendConfig | None = None, cache: TypedIRCache | None = None) -> None:
        self.config = config or QwenBackendConfig()
        self.cache = cache or TypedIRCache()
        self._load_error: str | None = None
        self._resolved_model_ref: str | None = None
        self._resolved_quantization: str | None = None

    def plan_typed_program(self, spec: ProgramSpec) -> BackendResult:
        cache_key = self.cache.make_key(
            spec,
            planner_profile=self.config.profile,
            planner_version=self.config.planner_version,
        )
        canonical_spec = canonicalize_spec(spec)
        hit = self.cache.load(cache_key)
        if hit is not None:
            return BackendResult(
                typed_program_payload=hit.payload,
                diagnostics=[f"Typed IR cache hit: {cache_key[:12]}", f"Cache stats: {self.cache.stats()}"],
            )

        template_plan = _structured_template_plan(spec)
        if template_plan is not None:
            payload = _typed_program_from_intent_plan(spec, template_plan)
            self.cache.store(cache_key, payload, canonical=canonical_spec)
            return BackendResult(
                typed_program_payload=payload,
                diagnostics=[
                    "Structured program routed through operator-template planner.",
                    f"Template: {template_plan.template_name}",
                    f"Planner confidence: {template_plan.confidence:.2f}",
                ],
            )

        structured = bool(spec.flow or spec.modules or spec.project or spec.objects)
        if structured or os.getenv("ELLANG_DISABLE_MODEL_PLANNER") == "1":
            payload = _heuristic_typed_program(spec)
            self.cache.store(cache_key, payload, canonical=canonical_spec)
            return BackendResult(
                typed_program_payload=payload,
                diagnostics=["Structured program routed through deterministic rule-based typed planner."],
            )

        rule_plan = _rule_based_intent_plan(spec)
        if rule_plan is not None:
            payload = _typed_program_from_intent_plan(spec, rule_plan)
            self.cache.store(cache_key, payload, canonical=canonical_spec)
            return BackendResult(
                typed_program_payload=payload,
                diagnostics=[
                    "Rule-based typed planner matched the prompt.",
                    f"Task family: {rule_plan.task_family}",
                    f"Planner confidence: {rule_plan.confidence:.2f}",
                ],
            )

        template_plan = _match_operator_template(spec)
        if template_plan is not None:
            payload = _typed_program_from_intent_plan(spec, template_plan)
            self.cache.store(cache_key, payload, canonical=canonical_spec)
            return BackendResult(
                typed_program_payload=payload,
                diagnostics=[
                    "Lazy planner matched operator-template library.",
                    f"Template: {template_plan.template_name}",
                    f"Task family: {template_plan.task_family}",
                    f"Planner confidence: {template_plan.confidence:.2f}",
                ],
            )

        classification = self._classify_intent(spec)
        if classification is not None and classification.confidence >= self.config.classification_confidence_threshold:
            payload = _typed_program_from_intent_plan(spec, classification)
            self.cache.store(cache_key, payload, canonical=canonical_spec)
            return BackendResult(
                typed_program_payload=payload,
                diagnostics=[
                    "Lazy AI typed planner used one-shot classification/extraction/operator-selection.",
                    f"Task family: {classification.task_family}",
                    f"Planner confidence: {classification.confidence:.2f}",
                    f"Resolved model source: {self._resolved_model_ref or self.config.model_id}",
                    f"Quantization mode: {self._resolved_quantization or self.config.quantization}",
                ],
            )

        payload = _heuristic_typed_program(spec)
        self.cache.store(cache_key, payload, canonical=canonical_spec)
        diagnostics = [
            "Planner fell back to deterministic typed-IR builder after low-confidence lazy AI classification.",
            f"Target deployment model: {self.config.model_id}",
            f"Preferred hardware profile: {self.config.profile}",
        ]
        if classification is not None:
            diagnostics.append(f"Low-confidence classification: {classification.task_family} ({classification.confidence:.2f})")
        if self._load_error:
            diagnostics.append(f"Local backend detail: {self._load_error}")
        return BackendResult(typed_program_payload=payload, diagnostics=diagnostics)

    def _classify_intent(self, spec: ProgramSpec) -> IntentPlan | None:
        prompt = self._classification_prompt(spec)
        parsed = self._infer_json(prompt)
        if parsed is None:
            return None
        try:
            operators = [GenericOperator(item) for item in parsed.get("operators", [])]
        except Exception:
            return None
        return IntentPlan(
            task_family=str(parsed.get("task_family", "generic_algorithm")),
            parameters=_normalize_parameters(spec, dict(parsed.get("parameters", {}))),
            operators=operators or list(_generic_operator_sequence(spec)),
            deterministic=bool(parsed.get("deterministic", False)),
            confidence=float(parsed.get("confidence", 0.0)),
            rationale=str(parsed.get("rationale", "")),
        )

    def _classification_prompt(self, spec: ProgramSpec) -> str:
        return (
            "You are a lightweight planner for an executable AI programming language.\n"
            "Do NOT output code. Do NOT output final answers.\n"
            "Your only job is intent classification, parameter extraction, and generic operator selection.\n"
            "Return strict JSON with keys:\n"
            "{"
            "\"task_family\":\"...\","
            "\"parameters\":{...},"
            "\"operators\":[...],"
            "\"deterministic\":true,"
            "\"confidence\":0.0,"
            "\"rationale\":\"...\""
            "}\n"
            "Allowed operators:\n"
            + ", ".join(item.value for item in GenericOperator)
            + "\n"
            "Allowed task families include sorting, filtering, substring_concat, aggregation, transformation, search, generic_algorithm.\n"
            "Prefer operator sequences that can be executed by a deterministic runtime.\n"
            f"Intent: {spec.intent}\n"
            f"Inputs: {json.dumps(spec.inputs, ensure_ascii=False)}\n"
            f"Outputs: {json.dumps(spec.outputs, ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(spec.constraints, ensure_ascii=False)}\n"
        )

    def _infer_json(self, prompt: str) -> dict[str, Any] | None:
        bundle = self._get_model_bundle()
        if bundle is None:
            return None
        tokenizer, model = bundle
        messages = [
            {"role": "system", "content": "You emit strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = inputs.to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            prompt_len = inputs["input_ids"].shape[-1]
            text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
            return _extract_json_object(text)
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            return None

    @lru_cache(maxsize=1)
    def _get_model_bundle(self) -> tuple[Any, Any] | None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            self._load_error = "transformers, torch, or quantization dependencies are not installed."
            return None
        resolved_model_id, resolved_quantization = self._resolve_model_choice()
        self._resolved_model_ref = resolved_model_id
        self._resolved_quantization = resolved_quantization
        kwargs: dict[str, Any] = {
            "local_files_only": self.config.prefer_local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }
        model_kwargs: dict[str, Any] = dict(kwargs)
        if resolved_quantization == "int4" and BitsAndBytesConfig is not None:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, "float16", None),
            )
            model_kwargs["device_map"] = "auto"
        elif resolved_quantization == "int8" and BitsAndBytesConfig is not None:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = "auto"
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(resolved_model_id, **model_kwargs)
            return tokenizer, model
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            return None

    def _resolve_model_choice(self) -> tuple[str, str]:
        explicit_path = os.getenv("ELLANG_MODEL_PATH")
        explicit_id = os.getenv("ELLANG_MODEL_ID")
        explicit_profile = os.getenv("ELLANG_MODEL_PROFILE", self.config.profile)
        if explicit_path:
            return explicit_path, os.getenv("ELLANG_MODEL_QUANTIZATION", self.config.quantization)
        if explicit_id:
            return explicit_id, os.getenv("ELLANG_MODEL_QUANTIZATION", self.config.quantization)
        discovered = discover_local_model_paths(profile=explicit_profile)
        if discovered:
            return discovered[0]
        return MODEL_PROFILES.get(explicit_profile, MODEL_PROFILES["consumer"])[0]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def describe_model_profile(profile: str = "consumer") -> dict[str, object]:
    discovered = discover_local_model_paths(profile=profile)
    return {
        "profile": profile,
        "candidates": [{"model_id": model_id, "quantization": quantization} for model_id, quantization in MODEL_PROFILES.get(profile, [])],
        "discovered_local_paths": [{"path": path, "quantization": quantization} for path, quantization in discovered],
    }


def discover_local_model_paths(profile: str = "consumer") -> list[tuple[str, str]]:
    roots = _candidate_roots()
    matches: list[tuple[str, str]] = []
    for model_id, quantization in MODEL_PROFILES.get(profile, MODEL_PROFILES["consumer"]):
        expected = _expected_cache_dir_name(model_id)
        for root in roots:
            candidate = root / expected
            if candidate.exists():
                snapshot = _snapshot_dir(candidate)
                if snapshot is not None:
                    matches.append((str(snapshot), quantization))
                    break
    return matches


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    if os.getenv("ELLANG_MODEL_DIR"):
        roots.append(Path(os.environ["ELLANG_MODEL_DIR"]))
    if os.getenv("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".huggingface" / "hub")
    return [root for root in roots if root.exists()]


def _expected_cache_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _snapshot_dir(cache_dir: Path) -> Path | None:
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        return cache_dir if any(cache_dir.iterdir()) else None
    children = [child for child in snapshots.iterdir() if child.is_dir()]
    if not children:
        return None
    return sorted(children)[-1]


def _program_flow_summary(spec: ProgramSpec) -> list[dict[str, object]]:
    return [_serialize_step(step) for step in spec.flow]


def _serialize_step(step: object) -> dict[str, object]:
    if isinstance(step, ActionStep):
        return {"kind": "ActionStep", "action": step.action}
    if isinstance(step, ModuleCallStep):
        return {"kind": "ModuleCallStep", "module_name": step.module_name}
    if isinstance(step, EmitStep):
        return {"kind": "EmitStep", "expression": step.expression}
    if isinstance(step, IfStep):
        return {
            "kind": "IfStep",
            "condition": step.condition,
            "then_steps": [_serialize_step(item) for item in step.then_steps],
            "else_steps": [_serialize_step(item) for item in step.else_steps],
        }
    if isinstance(step, LoopStep):
        return {
            "kind": "LoopStep",
            "iterator": step.iterator,
            "source": step.source,
            "body": [_serialize_step(item) for item in step.body],
        }
    return {"kind": type(step).__name__, "repr": str(step)}


def _rule_based_intent_plan(spec: ProgramSpec) -> IntentPlan | None:
    intent = spec.intent.lower()
    if "substring" in intent and "word" in intent:
        return IntentPlan(
            task_family="substring_concat",
            parameters={"text": "s", "words": "words"},
            operators=[
                GenericOperator.COMPUTE_LENGTH,
                GenericOperator.COMPUTE_TOTAL_LENGTH,
                GenericOperator.BUILD_FREQ_MAP,
                GenericOperator.SLIDING_WINDOW_SCAN,
                GenericOperator.COLLECT_INDICES,
            ],
            deterministic=True,
            confidence=0.99,
            rationale="matched substring concatenation pattern",
            template_name="substring_concat",
        )
    if "sort" in intent or "top" in intent:
        return IntentPlan(
            task_family="sorting",
            parameters={
                "source": _pick_primary_input(spec),
                "sort_key": "score" if "score" in intent else "value",
                "descending": any(token in intent for token in ("top", "desc", "highest")),
                "limit": _extract_trailing_number(intent, default=None) if "top" in intent else None,
            },
            operators=[GenericOperator.SORT],
            deterministic=True,
            confidence=0.95,
            rationale="matched sorting/top-k pattern",
            template_name="sort_topk",
        )
    if "filter" in intent:
        return IntentPlan(
            task_family="filtering",
            parameters={"source": _pick_primary_input(spec)},
            operators=[GenericOperator.FILTER],
            deterministic=True,
            confidence=0.90,
            rationale="matched filtering pattern",
            template_name="filter_basic",
        )
    return None


def _match_operator_template(spec: ProgramSpec) -> IntentPlan | None:
    intent = spec.intent.lower()
    flow_text = json.dumps(_program_flow_summary(spec), ensure_ascii=False).lower()
    combined = f"{intent} {flow_text}"
    for template in OPERATOR_TEMPLATES:
        if any(token in combined for token in template.match_any):
            parameters = {**template.defaults, **_infer_parameters_from_spec(spec, template)}
            return IntentPlan(
                task_family=template.task_family,
                parameters=parameters,
                operators=list(template.operators),
                deterministic=template.deterministic,
                confidence=template.confidence,
                rationale=template.rationale,
                template_name=template.name,
            )
    return None


def _structured_template_plan(spec: ProgramSpec) -> IntentPlan | None:
    flow_summary = _program_flow_summary(spec)
    flow_text = json.dumps(flow_summary, ensure_ascii=False).lower()
    if _flow_contains_kind(flow_summary, "LoopStep") and "emit" in flow_text:
        return IntentPlan(
            task_family="loop_emit_pipeline",
            parameters={"source": _pick_primary_input(spec), "expression": "item"},
            operators=[GenericOperator.LOOP, GenericOperator.MAP],
            deterministic=True,
            confidence=0.92,
            rationale="structured flow contains loop + emit",
            template_name="loop_emit",
        )
    if "sort" in flow_text or "top" in flow_text:
        return IntentPlan(
            task_family="structured_sort",
            parameters={
                "source": _pick_primary_input(spec),
                "sort_key": "score" if "score" in spec.intent.lower() or "score" in flow_text else "value",
                "descending": True,
                "limit": _extract_trailing_number(spec.intent.lower(), default=None) if "top" in spec.intent.lower() else None,
            },
            operators=[GenericOperator.SORT],
            deterministic=True,
            confidence=0.93,
            rationale="structured flow references sort/top pattern",
            template_name="structured_sort",
        )
    return None


def _typed_program_from_intent_plan(spec: ProgramSpec, plan: IntentPlan) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    flow_summary = _program_flow_summary(spec)

    def add_node(
        node_id: str,
        operator: GenericOperator,
        input_types: list[ValueType],
        output_type: ValueType,
        *,
        capabilities: list[Capability] | None = None,
        deterministic: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        nodes.append(
            {
                "node_id": node_id,
                "operator": operator.value,
                "input_types": [item.value for item in input_types],
                "output_type": output_type.value,
                "capabilities": [item.value for item in (capabilities or [])],
                "deterministic": deterministic,
                "config": config or {},
            }
        )

    add_node("input.main", GenericOperator.LOAD_INPUT, [], ValueType.RECORD, deterministic=True, config={"bindings": spec.inputs})
    previous = "input.main"

    if spec.project:
        add_node("project.main", GenericOperator.INIT_PROJECT, [ValueType.RECORD], ValueType.PROJECT, capabilities=[Capability.GIT_READ], deterministic=True, config=spec.project)
        edges.append({"source": previous, "target": "project.main", "label": "project"})
        previous = "project.main"

    for name, object_spec in spec.objects.items():
        object_id = f"object.{name}"
        add_node(object_id, GenericOperator.REGISTER_OBJECT, [ValueType.RECORD], ValueType.OBJECT, deterministic=True, config={"name": name, "fields": object_spec.fields})
        edges.append({"source": previous, "target": object_id, "label": "object"})
        previous = object_id

    if plan.task_family == "substring_concat":
        add_node("op.word_len", GenericOperator.COMPUTE_LENGTH, [ValueType.DATASET], ValueType.INT, deterministic=True, config={"source": plan.parameters.get("words", "words")})
        add_node("op.total_len", GenericOperator.COMPUTE_TOTAL_LENGTH, [ValueType.DATASET], ValueType.INT, deterministic=True, config={"source": plan.parameters.get("words", "words")})
        add_node("op.target_freq", GenericOperator.BUILD_FREQ_MAP, [ValueType.DATASET], ValueType.RECORD, deterministic=True, config={"source": plan.parameters.get("words", "words")})
        add_node(
            "op.scan",
            GenericOperator.SLIDING_WINDOW_SCAN,
            [ValueType.STRING, ValueType.DATASET],
            ValueType.INDEX_LIST,
            deterministic=True,
            config={"text": plan.parameters.get("text", "s"), "words": plan.parameters.get("words", "words")},
        )
        add_node("op.collect", GenericOperator.COLLECT_INDICES, [ValueType.INDEX_LIST], ValueType.INDEX_LIST, deterministic=True, config={"source": "op.scan"})
        edges.extend(
            [
                {"source": previous, "target": "op.word_len", "label": "word_len"},
                {"source": "op.word_len", "target": "op.total_len", "label": "total_len"},
                {"source": "op.total_len", "target": "op.target_freq", "label": "freq"},
                {"source": "op.target_freq", "target": "op.scan", "label": "scan"},
                {"source": "op.scan", "target": "op.collect", "label": "collect"},
            ]
        )
        previous = "op.collect"
    else:
        op_index = 0
        for operator in plan.operators:
            node_id = f"op.{operator.value}.{op_index}"
            op_index += 1
            output_type = {
                GenericOperator.SORT: ValueType.DATASET,
                GenericOperator.FILTER: ValueType.DATASET,
                GenericOperator.MAP: ValueType.DATASET,
                GenericOperator.REDUCE: ValueType.RECORD,
                GenericOperator.MODEL_PLAN: ValueType.RECORD,
            }.get(operator, ValueType.ANY)
            capabilities = [Capability.MODEL_INFER] if operator == GenericOperator.MODEL_PLAN else []
            add_node(
                node_id,
                operator,
                [ValueType.ANY],
                output_type,
                deterministic=plan.deterministic and operator != GenericOperator.MODEL_PLAN,
                capabilities=capabilities,
                config={**_normalize_parameters(spec, plan.parameters), "task_family": plan.task_family, "confidence": plan.confidence},
            )
            edges.append({"source": previous, "target": node_id, "label": operator.value})
            previous = node_id

        if not plan.operators:
            add_node(
                "op.plan.0",
                GenericOperator.MODEL_PLAN,
                [ValueType.RECORD],
                ValueType.RECORD,
                capabilities=[Capability.MODEL_INFER],
                deterministic=False,
                config={"intent": spec.intent, "task_family": plan.task_family, "confidence": plan.confidence},
            )
            edges.append({"source": previous, "target": "op.plan.0", "label": "plan"})
            previous = "op.plan.0"

    if _flow_contains_kind(flow_summary, "LoopStep"):
        add_node(
            "flow.loop",
            GenericOperator.LOOP,
            [ValueType.DATASET],
            ValueType.DATASET,
            deterministic=True,
            config={"source": _pick_primary_input(spec)},
        )
        edges.append({"source": previous, "target": "flow.loop", "label": "loop"})
        previous = "flow.loop"

    add_node("validate.output", GenericOperator.VALIDATE, [ValueType.ANY], ValueType.RECORD, deterministic=True, config={"constraints": spec.constraints})
    add_node("test.output", GenericOperator.SYNTH_TESTS, [ValueType.RECORD], ValueType.TEST_REPORT, deterministic=True, config={"intent": spec.intent})
    add_node("debug.output", GenericOperator.DEBUG_PREP, [ValueType.TEST_REPORT], ValueType.DEBUG_REPORT, capabilities=[Capability.DEBUG_ESCALATE], deterministic=False, config={"fallback": "external_api_if_local_debug_fails"})
    add_node("output.main", GenericOperator.OUTPUT, [ValueType.ANY], _output_type_from_spec(spec), deterministic=True, config={"bindings": spec.outputs, "source": previous})
    edges.extend(
        [
            {"source": previous, "target": "validate.output", "label": "validate"},
            {"source": "validate.output", "target": "test.output", "label": "test"},
            {"source": "test.output", "target": "debug.output", "label": "debug"},
            {"source": "debug.output", "target": "output.main", "label": "output"},
        ]
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "diagnostics": [
            f"Intent plan family: {plan.task_family}",
            f"Planner confidence: {plan.confidence:.2f}",
            f"Planner rationale: {plan.rationale}",
        ],
    }


def _heuristic_typed_program(spec: ProgramSpec) -> dict[str, Any]:
    plan = _rule_based_intent_plan(spec)
    if plan is None:
        template = _match_operator_template(spec)
        if template is not None:
            payload = _typed_program_from_intent_plan(spec, template)
            payload.setdefault("diagnostics", []).append("Heuristic planner stabilized output with operator-template library.")
            return payload
        plan = IntentPlan(
            task_family="generic_algorithm",
            parameters={"intent": spec.intent, "source": _pick_primary_input(spec), **_infer_parameters_from_spec(spec, None)},
            operators=list(_generic_operator_sequence(spec)),
            deterministic=False,
            confidence=0.35,
            rationale="generic fallback builder",
        )
    payload = _typed_program_from_intent_plan(spec, plan)
    payload.setdefault("diagnostics", []).append("Heuristic typed planner built a constrained operator graph.")
    return payload


def _pick_primary_input(spec: ProgramSpec) -> str:
    return next(iter(spec.inputs.keys()), "input")


def _output_type_from_spec(spec: ProgramSpec) -> ValueType:
    if not spec.outputs:
        return ValueType.ANY
    first = next(iter(spec.outputs.values()))
    return {
        "dataset": ValueType.DATASET,
        "string": ValueType.STRING,
        "bool": ValueType.BOOL,
        "record": ValueType.RECORD,
    }.get(first, ValueType.ANY)


def _extract_trailing_number(text: str, default: int | None) -> int | None:
    tokens = text.replace(",", " ").split()
    for token in reversed(tokens):
        if token.isdigit():
            return int(token)
    return default


def _flow_contains_kind(flow_summary: list[dict[str, object]], kind: str) -> bool:
    for step in flow_summary:
        if step.get("kind") == kind:
            return True
        for key in ("then_steps", "else_steps", "body"):
            nested = step.get(key)
            if isinstance(nested, list) and _flow_contains_kind([item for item in nested if isinstance(item, dict)], kind):
                return True
    return False


def _generic_operator_sequence(spec: ProgramSpec) -> tuple[GenericOperator, ...]:
    flow_summary = _program_flow_summary(spec)
    if _flow_contains_kind(flow_summary, "LoopStep"):
        return (GenericOperator.LOOP, GenericOperator.MAP)
    if _flow_contains_kind(flow_summary, "IfStep"):
        return (GenericOperator.EVAL_CONDITION, GenericOperator.MERGE)
    return (GenericOperator.MODEL_PLAN, GenericOperator.MAP)


def _infer_parameters_from_spec(spec: ProgramSpec, template: OperatorTemplate | None) -> dict[str, Any]:
    primary = _pick_primary_input(spec)
    intent = spec.intent.lower()
    params: dict[str, Any] = {"source": primary}
    if "score" in intent:
        params["sort_key"] = "score"
        params.setdefault("key", "score")
    if "name" in intent:
        params.setdefault("expression", "item.name")
    if "count" in intent or "group" in intent:
        params.setdefault("mode", "group_by")
        params.setdefault("key", "value")
    if "top" in intent:
        params["descending"] = True
        params["limit"] = _extract_trailing_number(intent, default=None)
    if template is not None and template.name == "filter_map":
        params.setdefault("predicate", "item")
    return params


def _normalize_parameters(spec: ProgramSpec, parameters: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parameters)
    normalized.setdefault("source", _pick_primary_input(spec))
    if normalized.get("expression") is None:
        normalized["expression"] = "item"
    return normalized
