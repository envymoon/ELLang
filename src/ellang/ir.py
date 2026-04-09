from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeKind(StrEnum):
    INPUT = "input"
    INFER = "infer"
    PROJECT = "project"
    OBJECT = "object"
    MODULE = "module"
    CONDITION = "condition"
    LOOP = "loop"
    TRANSFORM = "transform"
    SORT = "sort"
    FILTER = "filter"
    VALIDATE = "validate"
    TEST = "test"
    DEBUG = "debug"
    OUTPUT = "output"
    MERGE = "merge"


@dataclass(slots=True)
class IRNode:
    node_id: str
    kind: NodeKind
    title: str
    config: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class IREdge:
    source: str
    target: str
    label: str = ""


@dataclass(slots=True)
class ExecutionPlan:
    program_name: str
    intent: str
    nodes: list[IRNode]
    edges: list[IREdge]
    entrypoint: str
    outputs: list[str]
    diagnostics: list[str] = field(default_factory=list)
    suggested_tests: list[dict[str, object]] = field(default_factory=list)
    typed_program: Any | None = None
    bytecode: Any | None = None
    replay_schema_version: str = "ellang-replay-v1"

    def node_map(self) -> dict[str, IRNode]:
        return {node.node_id: node for node in self.nodes}

    def outgoing(self) -> dict[str, list[IREdge]]:
        graph: dict[str, list[IREdge]] = {}
        for edge in self.edges:
            graph.setdefault(edge.source, []).append(edge)
        return graph

    def incoming(self) -> dict[str, list[IREdge]]:
        graph: dict[str, list[IREdge]] = {}
        for edge in self.edges:
            graph.setdefault(edge.target, []).append(edge)
        return graph
