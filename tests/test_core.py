import unittest
from pathlib import Path

from ellang.bytecode import BytecodeProgram, Instruction, OpCode
from ellang.cache import TypedIRCache
from ellang.compiler import Compiler
from ellang.ideation import IdeationEngine
from ellang.models import QwenBackendConfig, QwenLocalBackend
from ellang.runtime import ExecutionEngine
from ellang.syntax import parse_program, parse_program as parse_rendered_program
from ellang.typed_ir import Capability, ResourceBudget, RuntimeTarget, ValueType
from ellang.verifier import typed_program_from_dict, lower_to_bytecode
from ellang.vm import ReferenceVM
from ellang.security import RuntimeQuota

REPO_ROOT = Path(__file__).resolve().parents[1]


class CoreRuntimeTests(unittest.TestCase):
    def test_parse_compile_execute_sort(self) -> None:
        spec = parse_program(
            """
            program "sort_students"
            intent "Sort students by score descending and return the top 2"
            input students: dataset
            input top_only: bool
            output result: dataset
            constraint deterministic = true

            object Student:
              field name: string
              field score: number

            module rank_students:
              intent "Rank students"
              use "sort students by score descending"

            flow:
              use module rank_students
              if top_only == true:
                use "sort students by score descending and return the top 2"
              else:
                emit students
            """
        )

        plan = Compiler().compile(spec)
        result = ExecutionEngine().execute(
            plan,
            {
                "students": [
                    {"name": "Ada", "score": 91},
                    {"name": "Linus", "score": 88},
                    {"name": "Grace", "score": 99},
                ],
                "top_only": True,
            },
        )

        self.assertEqual(result.value[0]["name"], "Grace")
        self.assertEqual(len(result.value), 2)
        self.assertTrue(plan.suggested_tests)
        self.assertIsNotNone(plan.typed_program)
        self.assertIsNotNone(plan.bytecode)
        self.assertIsNotNone(result.replay)

    def test_plan_contains_language_nodes(self) -> None:
        spec = parse_program(
            """
            program "identity"
            intent "Transform records"
            input students: dataset
            output result: dataset

            project:
              vcs: git

            flow:
              loop student in students:
                emit student
            """
        )
        backend = QwenLocalBackend(
            config=QwenBackendConfig(profile="consumer", planner_version="test-loop-v1"),
            cache=TypedIRCache(REPO_ROOT / ".ellang-cache-loop-test"),
        )
        plan = Compiler(backend=backend).compile(spec)
        node_kinds = {node.kind.value for node in plan.nodes}
        self.assertIn("project", node_kinds)
        self.assertIn("loop", node_kinds)
        self.assertEqual(plan.entrypoint, "input.main")
        self.assertGreater(len(plan.bytecode.instructions), 0)

    def test_rule_based_substring_planner_hits_vm_path(self) -> None:
        spec = parse_program(
            """
            program "substring_concat"
            intent "Return all starting indices of substrings in s that are a concatenation of all words in any order"
            input s: string
            input words: dataset
            output result: dataset
            constraint deterministic = true
            """
        )
        plan = Compiler().compile(spec)
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(
            plan,
            {"s": "barfoothefoobarman", "words": ["foo", "bar"]},
        )
        self.assertEqual(result.value, [0, 9])

    def test_typed_ir_cache_reuses_payload(self) -> None:
        cache_root = REPO_ROOT / ".ellang-cache-test"
        backend = QwenLocalBackend(
            config=QwenBackendConfig(profile="consumer"),
            cache=TypedIRCache(cache_root),
        )
        spec = parse_program(
            """
            program "sort_students"
            intent "Sort students by score descending and return the top 2"
            input students: dataset
            output result: dataset
            constraint deterministic = true
            """
        )
        first = backend.plan_typed_program(spec)
        second = backend.plan_typed_program(spec)
        self.assertEqual(first.typed_program_payload, second.typed_program_payload)
        self.assertTrue(any("cache hit" in item.lower() for item in second.diagnostics))
        self.assertGreaterEqual(int(backend.cache.stats().get("hits", 0)), 1)

    def test_operator_template_and_reference_vm_semantics(self) -> None:
        spec = parse_program(
            """
            program "filter_and_group"
            intent "Filter active students and group them"
            input students: dataset
            output result: record
            """
        )
        backend = QwenLocalBackend(
            config=QwenBackendConfig(profile="consumer", planner_version="test-template-v1"),
            cache=TypedIRCache(REPO_ROOT / ".ellang-cache-template-test"),
        )
        planned = backend.plan_typed_program(spec)
        diagnostics = " ".join(planned.diagnostics).lower()
        self.assertTrue(any(token in diagnostics for token in ("template", "rule-based", "cache hit")))

        program = BytecodeProgram(
            program_name="vm_ops",
            instructions=[
                Instruction(OpCode.LOAD_INPUT, {"node_id": "input.main"}, ValueType.RECORD),
                Instruction(OpCode.FILTER, {"node_id": "op.filter", "source": "students", "predicate": "item.active == true"}, ValueType.DATASET),
                Instruction(OpCode.MAP, {"node_id": "op.map", "source": "op.filter", "expression": "item.name"}, ValueType.DATASET),
                Instruction(OpCode.OUTPUT, {"node_id": "output.main", "source": "op.map"}, ValueType.DATASET),
            ],
        )
        result = ReferenceVM().execute(
            program,
            {"students": [{"name": "Ada", "active": True}, {"name": "Linus", "active": False}]},
            RuntimeQuota(ResourceBudget()),
        )
        self.assertEqual(result.output, ["Ada"])
        self.assertTrue(all("pc" in entry for entry in result.trace))

    def test_typed_program_parses_runtime_and_ffi_metadata(self) -> None:
        payload = {
            "nodes": [
                {
                    "node_id": "input.main",
                    "operator": "load_input",
                    "input_types": [],
                    "output_type": "record",
                    "deterministic": True,
                    "config": {},
                },
                {
                    "node_id": "ffi.call",
                    "operator": "call_ffi",
                    "input_types": ["record"],
                    "output_type": "record",
                    "capabilities": ["ffi.call"],
                    "deterministic": False,
                    "config": {"source": "input.main"},
                    "ffi_signature": {
                        "name": "native_transform",
                        "library": "ellang_native",
                        "abi": "system",
                        "args": [{"kind": "record"}],
                        "returns": {"kind": "record"},
                    },
                },
            ],
            "edges": [{"source": "input.main", "target": "ffi.call", "label": "ffi"}],
            "runtime": {
                "target": "native",
                "jit_tier": "hotspot",
                "aot_enabled": True,
                "hot_threshold": 4,
                "cross_platform_targets": ["native", "wasm", "jvm"],
            },
            "exported_types": {
                "Student": {
                    "kind": "object",
                    "name": "Student",
                    "fields": {"name": {"kind": "string"}, "score": {"kind": "number"}},
                }
            },
            "ffi_bindings": [
                {
                    "name": "native_transform",
                    "library": "ellang_native",
                    "abi": "system",
                    "args": [{"kind": "record"}],
                    "returns": {"kind": "record"},
                    "required_capabilities": ["ffi.call"],
                }
            ],
        }
        program = typed_program_from_dict(payload, program_name="ffi_program", budget=ResourceBudget())
        self.assertEqual(program.runtime.target, RuntimeTarget.NATIVE)
        self.assertIn(Capability.FFI_CALL, program.required_capabilities)
        self.assertIn("Student", program.exported_types)
        _, bytecode, diagnostics = lower_to_bytecode(program)
        self.assertIn("runtime", bytecode.to_serializable())
        self.assertTrue(any("FFI bindings" in item for item in diagnostics))

    def test_ideation_generates_ell_and_executes(self) -> None:
        engine = IdeationEngine()
        drafted = engine.ideate(
            "select the top 2 students by score",
            {
                "students": [
                    {"name": "Ada", "score": 91},
                    {"name": "Linus", "score": 88},
                    {"name": "Grace", "score": 99},
                ],
                "top_only": True,
            },
        )
        reparsed = parse_rendered_program(drafted.source)
        plan = Compiler().compile(reparsed)
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(
            plan,
            {
                "students": [
                    {"name": "Ada", "score": 91},
                    {"name": "Linus", "score": 88},
                    {"name": "Grace", "score": 99},
                ],
                "top_only": True,
            },
        )
        self.assertIn('program "', drafted.source)
        self.assertEqual(result.value[0]["name"], "Grace")

    def test_structured_minstack_executes_correctly(self) -> None:
        spec = parse_program(
            """
            program "MinStackProgram"
            intent "Implement a stack that supports push, pop, top, and getMin in O(1) time"

            input operations: list
            input values: list
            output result: list

            constraint deterministic = true

            object MinStack:
              field stack: list
              field min_stack: list

            module push(value):
              append value to stack
              if min_stack is empty OR value <= current minimum:
                append value to min_stack

            module pop:
              if top of stack == top of min_stack:
                remove last(min_stack)
              remove last(stack)

            module top:
              return last(stack)

            module getMin:
              return last(min_stack)

            flow:
              initialize MinStack
              loop op in operations:
                if op == "MinStack":
                  append null to results
                else:
                  if op == "push":
                    call push(values[next])
                    append null to results
                  else:
                    if op == "pop":
                      call pop()
                      append null to results
                    else:
                      if op == "top":
                        append call top() to results
                      else:
                        if op == "getMin":
                          append call getMin() to results
              emit results
            """
        )
        plan = Compiler().compile(spec)
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(
            plan,
            {
                "operations": ["MinStack", "push", "push", "push", "getMin", "pop", "top", "getMin"],
                "values": [None, -2, 0, -3, None, None, None, None],
            },
        )
        self.assertEqual(result.value, [None, None, None, None, -3, None, 0, -2])

    def test_ideated_three_sum_executes_correctly(self) -> None:
        drafted = IdeationEngine().ideate(
            "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that nums[i] + nums[j] + nums[k] == 0 and the solution set does not contain duplicate triplets.",
            {"nums": [-1, 0, 1, 2, -1, -4]},
        )
        spec = parse_rendered_program(drafted.source)
        plan = Compiler().compile(spec)
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(plan, {"nums": [-1, 0, 1, 2, -1, -4]})
        normalized = sorted(sorted(item) for item in result.value)
        self.assertEqual(normalized, [[-1, -1, 2], [-1, 0, 1]])

    def test_algorithm_family_hashmap_counting(self) -> None:
        drafted = IdeationEngine().ideate("Group the anagrams together.", {"words": ["eat", "tea", "tan", "ate", "nat", "bat"]})
        plan = Compiler().compile(parse_rendered_program(drafted.source))
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(plan, {"words": ["eat", "tea", "tan", "ate", "nat", "bat"]})
        normalized = sorted(sorted(group) for group in result.value)
        self.assertEqual(normalized, [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]])

    def test_algorithm_family_stack_queue_heap(self) -> None:
        drafted = IdeationEngine().ideate("Check whether a parentheses string is valid.", {"s": "()[]{}"})
        plan = Compiler().compile(parse_rendered_program(drafted.source))
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(plan, {"s": "()[]{}"})
        self.assertTrue(result.value)

    def test_algorithm_family_linked_list(self) -> None:
        drafted = IdeationEngine().ideate("Reverse a linked list.", {"head": [1, 2, 3, 4, 5]})
        plan = Compiler().compile(parse_rendered_program(drafted.source))
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(plan, {"head": [1, 2, 3, 4, 5]})
        self.assertEqual(result.value, [5, 4, 3, 2, 1])

    def test_algorithm_family_tree_graph(self) -> None:
        drafted = IdeationEngine().ideate("Return the binary tree level order traversal.", {"root": [3, 9, 20, None, None, 15, 7]})
        plan = Compiler().compile(parse_rendered_program(drafted.source))
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(plan, {"root": [3, 9, 20, None, None, 15, 7]})
        self.assertEqual(result.value, [[3], [9, 20], [15, 7]])

    def test_algorithm_family_dp_backtracking(self) -> None:
        drafted = IdeationEngine().ideate("Solve coin change and return the minimum number of coins.", {"coins": [1, 2, 5], "amount": 11})
        plan = Compiler().compile(parse_rendered_program(drafted.source))
        result = ExecutionEngine(workspace_root=str(REPO_ROOT)).execute(plan, {"coins": [1, 2, 5], "amount": 11})
        self.assertEqual(result.value, 3)


if __name__ == "__main__":
    unittest.main()
