"""Keep result snapshots separate from solver work buffers."""

from copy import deepcopy
from types import MappingProxyType
import json
import math
from pathlib import Path
import numpy as np


def to_json_data(value):
    """Convert numeric values without losing array shape or dtype."""
    if isinstance(value, np.ndarray):
        data = {
            "__type__": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
        if np.iscomplexobj(value):
            data.update(
                real=to_json_data(value.real.tolist()),
                imag=to_json_data(value.imag.tolist()),
            )
        else:
            data["data"] = to_json_data(value.tolist())
        return data
    if isinstance(value, np.generic):
        return to_json_data(value.item())
    if isinstance(value, complex):
        return {
            "__type__": "complex",
            "real": to_json_data(value.real),
            "imag": to_json_data(value.imag),
        }
    if isinstance(value, dict):
        return {str(k): to_json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_data(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Non-finite values require an explicit unavailable-data status."
            )
        return value
    raise TypeError(f"No JSON encoder for {type(value).__name__}.")


def from_json_data(value):
    """Restore the documented numeric JSON representation."""
    if isinstance(value, list):
        return [from_json_data(v) for v in value]
    if not isinstance(value, dict):
        return value
    if value.get("__type__") == "ndarray":
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not supported.")
        if "real" in value:
            array = np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
        else:
            array = np.asarray(value["data"])
        return array.astype(dtype).reshape(value["shape"])
    if value.get("__type__") == "complex":
        return complex(value["real"], value["imag"])
    result = {k: from_json_data(v) for k, v in value.items()}
    if isinstance(result.get("solution"), dict):
        result["solution"] = {int(k): int(v) for k, v in result["solution"].items()}
    return result


class RunRecord:
    """Store an independent result for one completed or stopped run.

    Parameters
    ----------
    fields : dict
        Native result values. The constructor takes an independent copy.

    Attributes
    ----------
    result : dict
        A fresh, complete JSON-compatible snapshot.
    field_names : tuple of str
        Names available through direct attribute access.

    Notes
    -----
    Direct attributes return copies. A new solve call cannot change this
    record. Array encodings preserve complex values, shape, and dtype.
    """

    def __init__(self, fields):
        self._fields = MappingProxyType(deepcopy(fields))

    @property
    def result(self):
        return to_json_data(dict(self._fields))

    @property
    def field_names(self):
        return tuple(self._fields)

    def get(self, name, default=None):
        """Return an independent native value for one field."""
        return deepcopy(self._fields.get(name, default))

    def __getattr__(self, name):
        if name in self._fields:
            return self.get(name)
        raise AttributeError(name)


def resolve_record(source):
    """Accept a solver, a RunRecord, or its JSON-compatible result."""
    if isinstance(source, RunRecord):
        return source
    if isinstance(source, dict):
        return RunRecord(from_json_data(source))
    record = getattr(source, "last_run", None)
    if isinstance(record, RunRecord):
        return record
    raise ValueError("Supply a completed solver, a RunRecord, or a result dictionary.")


def load_result(path, *, load_original_data=False):
    """Read a full result or a selected export.

    Parameters
    ----------
    path : path-like
        The main JSON result file.
    load_original_data : bool, default False
        Also read referenced .npy files from the same export directory.

    Returns
    -------
    RunRecord
        A result usable by the same analysis tools as a solver object.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("export_type") == "selected_result":
        fields = from_json_data(data["metadata"])
        fields.update(from_json_data(data["attributes"]))
        fields["original_data"] = data["original_data"]
        if load_original_data:
            for name, item in data["original_data"].items():
                file = Path(item["file"])
                if file.name != str(file) or item["format"] != "npy":
                    raise ValueError("Unsupported original-data reference.")
                fields[name] = np.load(path.parent / file, allow_pickle=False)
    else:
        fields = from_json_data(data)
    return RunRecord(fields)
