"""Share the run lifecycle without sharing solver mathematics."""

from copy import deepcopy
from time import perf_counter
from uuid import uuid4
import numpy as np
from .._version import __version__
from ..core.graph import WeightedGraph
from ..core.records import RunRecord
from ..core.saving import save_result, utc_now
from ..core.checkpoint import RunControl, CheckpointManager, validate_checkpoint_config


def positive_int(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


class BaseSolver:
    """Create one fresh result record for each solve call.

    Attributes
    ----------
    result : dict or None
        Complete JSON-compatible data for the last run.
    last_run : RunRecord or None
        Independent native result record.
    solution : dict or None
        Spins indexed by the original integer node labels.
    cut_value, solution_energy : float or None
        C(solution) and -C(solution), evaluated on the input snapshot.
    """

    algorithm_type = None
    supports_parallel = False
    supports_checkpoint = False
    extra_config_fields = ()
    optional_fields = (
        "solution",
        "cut_value",
        "solution_energy",
        "expected_cut",
        "expected_energy",
        "objective_value",
        "optimizer_loss",
        "optimal_angles",
        "optimizer_history",
        "probabilities",
        "statevector",
        "correlations",
        "worst_cut_value",
    )

    def __init__(self, config=None):
        self._config = deepcopy(config or {})
        if self._config.get("type", self.algorithm_type) != self.algorithm_type:
            raise ValueError("The configured type does not match this solver class.")
        allowed = {
            "schema_version",
            "type",
            "seed",
            "num_treads",
            "parameters",
            "objective",
            "estimation",
            "optimizer",
            "initialization",
            "readout",
            "limits",
            "collect",
            "output",
            "save",
            "checkpoint",
        } | set(self.extra_config_fields)
        if set(self._config) - allowed:
            raise ValueError(f"Unknown solver fields: {set(self._config) - allowed}")
        if self._config.get("schema_version", 1) != 1:
            raise ValueError("Unsupported configuration schema.")
        if self._config.get("checkpoint", {}).get("enabled", False):
            if not self.supports_checkpoint:
                raise ValueError(
                    f"Checkpoint/resume is not implemented for {self.algorithm_type} in {__version__}."
                )
            self._config["checkpoint"] = validate_checkpoint_config(self._config["checkpoint"])
        self._last_run = None
        self._active = False
        self._run_id = None
        self._resume_state = None
        self._resume_arrays = {}
        self._resume_run_id = None
        self._resume_graph = None
        self._resume_control = None
        self._resume_mode = False
        self.last_checkpoint = None
        self._checkpoint_manager = None

    @property
    def config(self):
        return (
            self._last_run.config
            if self._last_run is not None
            else deepcopy(self._config)
        )

    @property
    def result(self):
        return None if self._last_run is None else self._last_run.result

    @property
    def last_run(self):
        return self._last_run

    def __getattr__(self, name):
        record = self.__dict__.get("_last_run")
        if record is not None and name in record.field_names:
            return record.get(name)
        if name in self.optional_fields:
            return None
        raise AttributeError(name)

    def solve(self, graph, *, report=None, logs=None, num_treads=None, control=None):
        """Run on an independent input snapshot and return a spin dictionary.

        Parameters
        ----------
        graph : WeightedGraph
            The input graph. The solver does not modify it.
        report, logs : {0, 1, 2}, optional
            Independent output levels. Configuration defaults are zero.
        num_treads : int, optional
            CPU budget for this run.
        control : RunControl, optional
            Explicit pause/cancel control.  A normal solve never resumes an
            old checkpoint automatically.

        Returns
        -------
        dict of int to int
            One ordinary integer spin, +1 or -1, per original node.

        Notes
        -----
        A fresh run resets the configured seed.  A restored object continues
        only through ``resume()``, never through ``solve()``.
        """
        if self._active:
            raise RuntimeError("This solver already has an active run.")
        if control is not None and not isinstance(control, RunControl):
            raise TypeError("control must be a RunControl.")
        if control is not None and not self.supports_checkpoint:
            raise ValueError(f"RunControl is not supported by {self.algorithm_type}.")
        resume_mode = bool(self._resume_mode)
        if resume_mode and control is None:
            control = self._resume_control
        self._last_run = None
        if not isinstance(graph, WeightedGraph):
            raise TypeError("graph must be a WeightedGraph.")
        config = deepcopy(self._config)
        config.update(type=self.algorithm_type, schema_version=1)
        config.setdefault("seed", None)
        output = config.setdefault("output", {})
        if set(output) - {"report", "logs", "color"}:
            raise ValueError("Unknown output field.")
        output.setdefault("report", 0)
        output.setdefault("logs", 0)
        output.setdefault("color", "auto")
        for key, value in (("report", report), ("logs", logs)):
            if value is not None:
                output[key] = value
            if type(output[key]) is not int or output[key] not in (0, 1, 2):
                raise ValueError(f"{key} must be 0, 1, or 2.")
        threads = config.get("num_treads", 1) if num_treads is None else num_treads
        positive_int(threads, "num_treads")
        if threads != 1 and not self.supports_parallel:
            raise ValueError(
                "Parallel CPU execution is not implemented for this solver; use num_treads=1."
            )
        if output["color"] not in ("auto", "always", "never"):
            raise ValueError("output.color must be auto, always, or never.")
        config["num_treads"] = threads
        self._validate(config, graph)
        self._last_run = None
        self._active = True
        started = utc_now()
        clock = perf_counter()
        snapshot = graph.copy()
        self._run_id = self._resume_run_id if resume_mode and self._resume_run_id else str(uuid4())
        if resume_mode and self._resume_state is not None:
            saved_rng = self._resume_state.get("rng_state")
            rng = np.random.default_rng()
            if saved_rng is not None:
                rng.bit_generator.state = deepcopy(saved_rng)
        else:
            rng = np.random.default_rng(config.get("seed"))
        self._control = control
        try:
            fields = self._run(snapshot, config, rng)
            solution = {int(k): int(v) for k, v in fields["solution"].items()}
            cut = snapshot.cut_value(solution)
            elapsed = perf_counter() - clock
            fields.update(
                schema_version=1,
                algorithm_type=self.algorithm_type,
                package_version=__version__,
                run_id=self._run_id,
                config=config,
                graph=snapshot.snapshot(),
                node_order=list(snapshot.nodes),
                started_at=started,
                finished_at=utc_now(),
                timing={"compute_seconds": elapsed},
                solution=solution,
                cut_value=cut,
                solution_energy=-cut,
                seed=config.get("seed"),
                num_treads=fields.get("num_treads", config["num_treads"]),
            )
            fields.setdefault("status", "completed")
            fields.setdefault("termination_reason", "completed")
            fields.setdefault("resources", {"shots_used": 0, "num_treads": 1})
            extra_report = fields.pop("report", None)
            fields["report"] = {
                "status": fields["status"],
                "termination_reason": fields["termination_reason"],
                "cut_value": cut,
                "solution_energy": -cut,
                "compute_seconds": elapsed,
                "resources": deepcopy(fields["resources"]),
            }
            if isinstance(extra_report, dict):
                fields["report"].update(deepcopy(extra_report))
            for key in self.optional_fields:
                fields.setdefault(key, None)
            self._last_run = RunRecord(fields)
        except Exception as error:
            self._last_run = RunRecord(
                {
                    "schema_version": 1,
                    "algorithm_type": self.algorithm_type,
                    "package_version": __version__,
                    "run_id": self._run_id,
                    "status": "failed",
                    "termination_reason": type(error).__name__,
                    "error": str(error),
                    "config": config,
                    "graph": snapshot.snapshot(),
                    "started_at": started,
                    "finished_at": utc_now(),
                    "timing": {"compute_seconds": perf_counter() - clock},
                    "solution": None,
                }
            )
            raise
        finally:
            self._active = False
            if resume_mode:
                self._resume_mode = False
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.release()
                self._checkpoint_manager = None
        if output["report"]:
            from ..tools.reporting import print_report

            print_report(self._last_run, output["report"], output["color"])
        if "save" in config:
            save_result(self._last_run, config["save"])
        return deepcopy(solution)

    def _validate(self, config, graph):
        raise NotImplementedError

    def _run(self, graph, config, rng):
        raise NotImplementedError
