"""Publish new result files without replacing existing user data."""

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .records import resolve_record, to_json_data


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _name(value):
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise ValueError("filename must be a nonempty base name.")
    if any(c in value for c in '/\\<>:"|?*'):
        raise ValueError("filename must not contain a path or reserved characters.")
    return value


@dataclass(frozen=True)
class SavedFiles:
    """Describe the completed export files."""

    primary: Path
    files: tuple


def publish_files(config, payloads, primary):
    """Publish a prepared file set. Publish the main record last.

    Hard-link creation is exclusive: it fails if the destination already
    exists. Keep temporary files on the destination filesystem. This
    avoids a check-then-overwrite race between simultaneous exports.
    """
    allowed = {
        "path",
        "filename",
        "makedir",
        "savedir",
        "atts",
        "save_original_data",
        "format",
        "DPI",
        "metadata",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown save fields: {sorted(set(config) - allowed)}")
    base = Path(config["path"]).expanduser()
    stem = _name(config["filename"])
    for flag in ("makedir", "savedir", "metadata"):
        if flag in config and not isinstance(config[flag], bool):
            raise TypeError(f"save.{flag} must be boolean.")
    for name in payloads:
        _name(name)
    if not base.exists():
        if not config.get("makedir", False):
            raise FileNotFoundError(base)
        base.mkdir(parents=True, exist_ok=True)
    if not base.is_dir():
        raise NotADirectoryError(base)
    folder = base / stem if config.get("savedir", False) else base
    if config.get("savedir", False) and os.path.lexists(folder):
        raise FileExistsError(f"Export folder already exists: {folder}")
    conflicts = [
        str(folder / name) for name in payloads if os.path.lexists(folder / name)
    ]
    if conflicts:
        raise FileExistsError("Export targets already exist: " + ", ".join(conflicts))
    stage = Path(tempfile.mkdtemp(prefix=".qa_export_", dir=base))
    created = []
    owned_folder = False
    try:
        # Complete serialization before publishing any result file.
        for name, content in payloads.items():
            with (stage / name).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        if config.get("savedir", False):
            folder.mkdir()
            owned_folder = True
        order = [name for name in payloads if name != primary] + [primary]
        for name in order:
            target = folder / name
            os.link(stage / name, target)
            created.append((target, (stage / name).stat().st_ino))
        return SavedFiles(folder / primary, tuple(folder / name for name in order))
    except BaseException:
        # Remove only the files published by this export attempt.
        for path, inode in reversed(created):
            if path.exists() and path.stat().st_ino == inode:
                path.unlink()
        if owned_folder:
            try:
                folder.rmdir()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(stage)


def _result_payloads(source, config):
    """Serialize selected attributes and optional original NumPy arrays.

    Parameters
    ----------
    source : solver, RunRecord, or dict
        A completed or stopped run.
    config : dict
        Fields path, filename, atts, save_original_data, makedir, savedir.
        An omitted atts field selects all collected result fields.

    Returns
    -------
    tuple of dict and str
        File contents indexed by base name, and the main JSON base name.
    """
    record = resolve_record(source)
    stem = _name(config["filename"])
    selected = config.get("atts", list(record.field_names))
    originals = config.get("save_original_data", [])
    for names in (selected, originals):
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise TypeError("Attribute selections must be lists of unique names.")
        for name in names:
            _name(name)
            if name not in record.field_names:
                raise ValueError(f"Unknown result attribute: {name}")
    mandatory = [
        "schema_version",
        "algorithm_type",
        "package_version",
        "run_id",
        "status",
        "termination_reason",
        "started_at",
        "finished_at",
        "config",
        "graph",
        "resources",
        "timing",
        "report",
        "data_status",
    ]
    metadata = {
        name: record.get(name) for name in mandatory if name in record.field_names
    }
    payloads, descriptors = {}, {}
    for name in originals:
        array = record.get(name)
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValueError(f"{name} is not an available numeric NumPy array.")
        filename = f"{stem}_{name}.npy"
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payloads[filename] = buffer.getvalue()
        descriptors[name] = {
            "file": filename,
            "format": "npy",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    document = {
        "schema_version": 1,
        "export_type": "selected_result",
        "saved_at": utc_now(),
        "metadata": to_json_data(metadata),
        "attributes": {name: to_json_data(record.get(name)) for name in selected},
        "original_data": descriptors,
        "export_complete": True,
    }
    primary = stem + ".json"
    payloads[primary] = json.dumps(
        document, ensure_ascii=False, allow_nan=False, indent=2
    ).encode("utf-8")
    return payloads, primary


def save_result(source, config):
    """Publish a JSON result and the selected original arrays.

    Parameters
    ----------
    source : BaseSolver, RunRecord, or dict
        A completed or stopped run. The source is not modified.
    config : dict
        ``path`` and ``filename`` are required. ``atts`` selects JSON
        attributes. ``save_original_data`` selects numeric arrays for
        separate NPY files. ``makedir`` creates missing parent folders.
        ``savedir`` creates one exclusive folder named ``filename``.

    Returns
    -------
    SavedFiles
        The main JSON path and every published file path.

    Raises
    ------
    FileExistsError
        Any destination file or the requested export folder exists.
    ValueError
        An attribute is unknown or its native format is not supported.

    Notes
    -----
    Publish the JSON record after all native arrays. This makes the
    JSON file the completion marker for this export. Existing files
    are never replaced. This implementation requires filesystem hard
    links within the destination filesystem.
    """
    payloads, primary = _result_payloads(source, config)
    return publish_files(config, payloads, primary)
