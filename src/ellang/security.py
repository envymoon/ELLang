from __future__ import annotations

from dataclasses import dataclass, field

from .typed_ir import Capability, ResourceBudget, TypedProgram


@dataclass(slots=True)
class CapabilityPolicy:
    granted: list[Capability] = field(default_factory=list)

    def allows(self, capability: Capability) -> bool:
        return capability in self.granted


@dataclass(slots=True)
class RuntimeQuota:
    budget: ResourceBudget
    tokens_used: int = 0
    cpu_ms_used: int = 0
    wall_ms_used: int = 0
    writes_used: int = 0
    network_calls_used: int = 0

    def charge_tokens(self, count: int) -> None:
        self.tokens_used += count
        if self.tokens_used > self.budget.max_tokens:
            raise RuntimeError("Token budget exceeded.")

    def charge_cpu_ms(self, count: int) -> None:
        self.cpu_ms_used += count
        if self.cpu_ms_used > self.budget.max_cpu_ms:
            raise RuntimeError("CPU budget exceeded.")

    def charge_wall_ms(self, count: int) -> None:
        self.wall_ms_used += count
        if self.wall_ms_used > self.budget.max_wall_ms:
            raise RuntimeError("Wall-clock budget exceeded.")

    def charge_network(self) -> None:
        self.network_calls_used += 1
        if self.network_calls_used > self.budget.max_network_calls:
            raise RuntimeError("Network budget exceeded.")


class CapabilityVerifier:
    def verify(self, program: TypedProgram, policy: CapabilityPolicy) -> list[str]:
        diagnostics: list[str] = []
        for capability in program.required_capabilities:
            if not policy.allows(capability):
                raise PermissionError(f"Capability not granted: {capability.value}")
            diagnostics.append(f"Capability granted: {capability.value}")
        return diagnostics
