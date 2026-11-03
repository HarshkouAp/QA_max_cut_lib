"""Small, typed decision and quantisation components for final controllers.

The components in this module keep physical correlation observations,
decision evidence, and heuristic proxies separate.  Controllers that
consume them are responsible for planning all observations before work.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np

from ..core.graph import WeightedGraph
from ..core.reduction import GraphReducer

__all__ = [
    "DecisionEvidence",
    "odds_fusion",
    "selected_observation",
    "quantize_angles",
    "ParityForest",
    "restriction_cut_proxy",
    "legacy_history_mix",
    "positive_history_mix",
    "HierarchicalCorrelationProxy",
    "hierarchical_proxy",
    "repeat_while_improving",
]


@dataclass
class DecisionEvidence:
    kind: str
    value: float | None
    domain: str
    origin: str
    valid: bool = True
    reason: str | None = None
    physical_observations: tuple = ()
    observations: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "kind": self.kind,
            "value": self.value,
            "domain": self.domain,
            "origin": self.origin,
            "valid": self.valid,
            "reason": self.reason,
            "physical_observations": list(self.physical_observations),
            "observations": deepcopy(self.observations),
        }


def odds_fusion(correlations, *, saturated_conflict="exclude_pair"):
    """Product-of-experts decision from physical pair correlations.

    ``q=tanh(sum atanh(c))``.  A +1/-1 conflict is a saturated 0/0 case,
    not a NaN winner.  Missing observations are excluded without changing
    the physical mean.
    """
    values = []
    for item in correlations:
        if isinstance(item, dict):
            raw = item.get("physical_correlation", item.get("c"))
        else:
            raw = item
        if raw is None:
            continue
        c = float(raw)
        if not math.isfinite(c) or abs(c) > 1.0 + 1e-12:
            raise ValueError("correlation must be finite in [-1,1].")
        values.append(min(1.0, max(-1.0, c)))
    if not values:
        return DecisionEvidence("odds_product", None, "correlation", "odds_fusion", False, "no_observations")
    if any(value >= 1.0 for value in values) and any(value <= -1.0 for value in values):
        if saturated_conflict != "exclude_pair":
            raise ValueError("saturated conflict policy is not supported.")
        return DecisionEvidence(
            "odds_product", None, "correlation", "odds_fusion", False,
            "saturated_conflict", tuple(values),
        )
    finite = [value for value in values if -1.0 < value < 1.0]
    if any(value >= 1.0 for value in finite) or any(value <= -1.0 for value in finite):
        return DecisionEvidence("odds_product", None, "correlation", "odds_fusion", False, "saturated")
    q = math.tanh(math.fsum(math.atanh(value) for value in finite))
    p_plus = (q + 1.0) / 2.0
    return DecisionEvidence(
        "odds_product", float(q), "probability", "odds_fusion", True,
        observations={"p_plus": float(p_plus), "physical_mean": float(np.mean(values)), "count": len(values)},
    )


def selected_observation(observations, *, weight_power=1):
    """Choose one real observation by weighted magnitude, with stable ties."""
    if weight_power < 0 or not math.isfinite(float(weight_power)):
        raise ValueError("weight_power must be finite and nonnegative.")
    best = None
    for item in observations:
        c = float(item["physical_correlation"])
        weight = abs(float(item.get("weight", 1.0)))
        score = (weight ** weight_power) * abs(c)
        key = (-score, item.get("round", 0), str(item.get("task_id", "")))
        if best is None or key < best[0]:
            best = (key, item)
    if best is None:
        return DecisionEvidence("selected_observation", None, "correlation", "round_max", False, "no_observations")
    item = best[1]
    return DecisionEvidence(
        "selected_observation", float(item["physical_correlation"]), "correlation", "round_max",
        observations={"observation": deepcopy(item), "score": float(-best[0][0])},
    )


def quantize_angles(theta, *, delta=math.pi / 8):
    values = np.asarray(theta, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("theta must be a finite vector.")
    if not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError("delta must be positive and finite.")
    quantized = np.rint(values / delta) * delta
    return {
        "continuous": [float(value) for value in values],
        "quantized": [float(value) for value in quantized],
        "delta": float(delta),
        "units": "radians",
        "tie_rule": "nearest_ties_to_even",
        "integer_grid": [int(round(value / delta)) for value in values],
    }


class ParityForest:
    """Parity union-find for frozen batch proposals.

    ``parity[x]`` stores ``s_x = parity[x] * s_parent[x]``.
    """

    def __init__(self, labels):
        self.parent = {int(node): int(node) for node in labels}
        self.parity = {int(node): 1 for node in labels}
        self.rank = {int(node): 0 for node in labels}

    def find(self, node):
        node = int(node)
        if node not in self.parent:
            raise KeyError(node)
        if self.parent[node] == node:
            return node, 1
        parent = self.parent[node]
        root, parent_parity = self.find(parent)
        total = self.parity[node] * parent_parity
        self.parent[node] = root
        self.parity[node] = total
        return root, total

    def add(self, u, v, sign, *, task_id=None):
        if sign not in (-1, 1):
            raise ValueError("proposal sign must be +/-1.")
        ru, eu = self.find(u)
        rv, ev = self.find(v)
        if ru == rv:
            if eu * ev == sign:
                return {"status": "redundant", "removed": None, "retained": None, "sign": sign, "task_id": task_id, "root": ru}
            return {"status": "conflict", "removed": None, "retained": None, "sign": sign, "task_id": task_id, "root": ru}
        # Attach the smaller root to the larger root, keeping canonical
        # orientation and one stable root per component.
        if ru < rv:
            child, root, child_parity, root_parity = ru, rv, eu, ev
        else:
            child, root, child_parity, root_parity = rv, ru, ev, eu
        # s_child = sign * child_parity * root_parity * s_root
        self.parent[child] = root
        self.parity[child] = sign * child_parity * root_parity
        self.rank[root] = max(self.rank[root], self.rank[child] + 1)
        return {
            "status": "accepted", "removed": min(ru, rv), "retained": max(ru, rv),
            "sign": sign, "task_id": task_id, "root": max(ru, rv),
        }

def restriction_cut_proxy(graph, correlations, pair, sign, *, zero_sign=1):
    """Read-only survivor-moment proxy for one oriented reduction."""
    if not isinstance(graph, WeightedGraph):
        raise TypeError("graph must be a WeightedGraph.")
    u, v = sorted(int(node) for node in pair)
    if u not in graph.nodes or v not in graph.nodes:
        raise ValueError("pair labels must be active.")
    if sign not in (-1, 1):
        raise ValueError("sign must be +/-1.")
    total = 0.0
    for a, b, weight in graph.edges:
        if u in (a, b):
            continue
        c = correlations.get((min(a, b), max(a, b)))
        if c is None:
            c = 0.0
        total += float(weight) * (1.0 - float(c)) / 2.0
    c_uv = correlations.get((u, v))
    if c_uv is None:
        c_uv = 0.0
    total += graph.weight(u, v) * (1.0 - sign) / 2.0
    for k in sorted(graph.nodes):
        if k in (u, v):
            continue
        c_vk = correlations.get((min(v, k), max(v, k)))
        if c_vk is None:
            continue
        total += graph.weight(u, k) * (1.0 - sign * float(c_vk)) / 2.0
    return {
        "value": float(total),
        "pair": [u, v],
        "sign": int(sign),
        "kind": "restriction_cut_proxy_v1",
    }


def legacy_history_mix(old_cut, new_cut, old_weight, new_weight):
    """Literal signed affine source statistic, not a convex mixture."""
    old_weight = float(old_weight)
    new_weight = float(new_weight)
    denominator = old_weight + new_weight
    if denominator == 0:
        return {"value": None, "reason": "zero_denominator"}
    return {"value": (old_weight * float(old_cut) + new_weight * float(new_cut)) / denominator}


def positive_history_mix(old_cut, new_cut, old_weight, new_weight):
    old_weight = max(0.0, float(old_weight))
    new_weight = max(0.0, float(new_weight))
    if old_weight + new_weight == 0:
        return {"value": 0.0, "rule": "uniform_all_zero"}
    return {"value": (old_weight * float(old_cut) + new_weight * float(new_cut)) / (old_weight + new_weight)}


def repeat_while_improving(callback, *, max_calls, min_improvement=0.0, initial=None):
    best = initial
    calls = []
    for index in range(int(max_calls)):
        value = callback(index)
        calls.append(value)
        if best is None or value > best + float(min_improvement):
            best = value
        else:
            break
    return {"best": best, "calls": calls, "stopped": "improvement" if len(calls) <= int(max_calls) else "max_calls"}


@dataclass
class HierarchicalCorrelationProxy:
    raw: float
    value: float | None
    valid: bool
    reason: str | None = None
    path: tuple = ()
    projected: bool = False

    def to_dict(self):
        return {
            "kind": "hierarchical_covariance_proxy",
            "raw": self.raw,
            "value": self.value,
            "valid": self.valid,
            "reason": self.reason,
            "path": list(self.path),
            "projected": self.projected,
        }


def hierarchical_proxy(parent_m_a, parent_m_b, c_parent, leaf_m_a, leaf_m_b, *, path=()):
    """Covariance-transfer proxy with leaf Fréchet validity check.

    Parent means define the correlation coefficient; leaf means define the
    proxy pair marginal.  A valid pair marginal is not a global state.
    """
    parent_m_a = float(parent_m_a)
    parent_m_b = float(parent_m_b)
    leaf_m_a = float(leaf_m_a)
    leaf_m_b = float(leaf_m_b)
    parent_denominator = math.sqrt(
        max(0.0, (1.0 - parent_m_a ** 2) * (1.0 - parent_m_b ** 2))
    )
    if parent_denominator <= 1e-15:
        return HierarchicalCorrelationProxy(None, None, False, "zero_variance", tuple(path))
    rho = (float(c_parent) - parent_m_a * parent_m_b) / parent_denominator
    raw = leaf_m_a * leaf_m_b + rho * math.sqrt(
        max(0.0, (1.0 - leaf_m_a ** 2) * (1.0 - leaf_m_b ** 2))
    )
    low = -1.0 + abs(leaf_m_a + leaf_m_b)
    high = 1.0 - abs(leaf_m_a - leaf_m_b)
    valid = -1.0 - 1e-12 <= raw <= 1.0 + 1e-12 and low - 1e-12 <= raw <= high + 1e-12
    return HierarchicalCorrelationProxy(
        raw, raw if valid else None, valid, None if valid else "frechet_violation", tuple(path)
    )
