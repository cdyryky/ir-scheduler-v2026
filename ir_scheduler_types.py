from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Resident:
    resident_id: str
    track: str


@dataclass(frozen=True)
class ScheduleInput:
    block_labels: List[str]
    residents: List[Resident]
    blocked: Dict[Tuple[str, int, str], bool]
    forced: Dict[Tuple[str, int, str], bool]
    weights: Dict[str, int]
    num_solutions: int
    constraint_modes: Dict[str, str]
    soft_priority: List[str]
    constraint_params: Dict[str, dict]
    requirements: Dict[str, Dict[str, int]]
    warnings: Tuple[str, ...] = ()


@dataclass
class Solution:
    assignments: Dict[str, Dict[str, Dict[str, float]]]
    objective: Dict[str, int]


@dataclass
class Diagnostic:
    status: str
    conflicting_constraints: List[Dict[str, str]]
    suggestions: List[Dict[str, str]]


@dataclass
class SolveResult:
    solutions: List[Solution]
    diagnostic: Optional[Diagnostic] = None
    warnings: Tuple[str, ...] = ()


class ScheduleError(Exception):
    pass
