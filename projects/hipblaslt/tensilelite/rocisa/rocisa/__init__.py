# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import types

from pathlib import Path as _Path

if any(_Path(__file__).parent.glob("_rocisa.abi3.*")) and sys.version_info < (3, 12):
    raise ImportError(
        f"rocisa stable-ABI extension requires Python >= 3.12 "
        f"(running {sys.version_info.major}.{sys.version_info.minor}). "
        f"Install a non-stable-ABI build or upgrade Python."
    )
del _Path

# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------
# ``ROCISA_BACKEND=stinkytofu`` redirects ``import rocisa`` to the
# ``rocisa_stinkytofu_adaptor`` shim (a rocisa-shaped facade backed by the
# stinkytofu Python binding ``_stinkytofu.so``). Anything else (or unset)
# keeps the original nanobind bindings in ``_rocisa``.

_BACKEND = os.environ.get("ROCISA_BACKEND", "").strip().lower()

_ADAPTER_PKG = "rocisa_stinkytofu_adaptor"


def _load_stinkytofu_adapter() -> bool:
    """Try to install the rocisa_stinkytofu_adaptor as the ``rocisa`` module.

    Returns True iff we successfully rewired sys.modules; on any failure
    we fall back to the nanobind bindings silently (the caller decides
    whether that is acceptable).
    """

    # Locate ``<repo_root>/projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor``
    # by walking up from this file until we find an ancestor whose basename
    # is ``projects``; the adapter then lives at
    # ``<that_parent>/projects/hipblaslt/tensilelite/<_ADAPTER_PKG>/``.
    #
    # The adapter is a sibling of ``tensilelite/rocisa/`` on purpose — it is
    # a *consumer* of the stinkytofu Python binding (a tensilelite-internal
    # alternative backend), not a piece of ``shared/stinkytofu/`` itself.
    # Sibling-of-rocisa keeps ``ROCISA_BACKEND=stinkytofu`` purely a
    # tensilelite concern.
    #
    # Works for both the source tree (``<repo>/projects/hipblaslt/
    # tensilelite/rocisa/rocisa/__init__.py``) and CMake-staged copies
    # under ``<repo>/projects/hipblaslt/tensilelite/<build_dir>/tensilelite/
    # rocisa/rocisa/__init__.py`` because in either case walking up from
    # ``__file__`` eventually hits the ``projects`` directory.
    repo_root = None
    cur = os.path.dirname(os.path.abspath(__file__))
    adapter_rel = os.path.join("projects", "hipblaslt", "tensilelite", _ADAPTER_PKG)
    while True:
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            break
        if os.path.basename(cur) == "projects":
            candidate = parent
            if os.path.isdir(os.path.join(candidate, adapter_rel)):
                repo_root = candidate
                break
        cur = parent
    if repo_root is None:
        return False
    adapter_parent = os.path.join(repo_root, adapter_rel)

    if not os.path.isdir(os.path.join(adapter_parent, _ADAPTER_PKG)):
        return False

    if adapter_parent not in sys.path:
        sys.path.insert(0, adapter_parent)

    try:
        import rocisa_stinkytofu_adaptor as _adapter  # noqa: F401
    except Exception:
        return False

    # Install the adapter as ``rocisa`` and re-export each
    # ``rocisa_stinkytofu_adaptor.*`` submodule under ``rocisa.*`` in
    # ``sys.modules``.
    sys.modules["rocisa"] = _adapter
    _prefix = f"{_ADAPTER_PKG}."
    for _name, _obj in vars(_adapter).items():
        if isinstance(_obj, types.ModuleType) and _obj.__name__.startswith(_prefix):
            short = _obj.__name__[len(_prefix):]
            sys.modules[f"rocisa.{short}"] = _obj

    return True


def _find_stale_sources(so_path, source_roots, build_dir):
    """Return source files newer than so_path, excluding files under build_dir.

    Extracted from the module-level staleness check so it can be unit-tested
    without requiring a real _rocisa.so or touching actual source files.
    """
    from pathlib import Path

    so_mtime = Path(so_path).stat().st_mtime
    build_dir = Path(build_dir).resolve()
    stale = []
    for root in source_roots:
        for pattern in ("*.[ch]pp", "*.h", "*.def", "*.inc"):
            for p in Path(root).rglob(pattern):
                if p.stat().st_mtime > so_mtime and not p.resolve().is_relative_to(build_dir):
                    stale.append(str(p))
    return stale


if _BACKEND == "stinkytofu" and _load_stinkytofu_adapter():
    # stinkytofu adapter active; wiring done inside _load_stinkytofu_adapter.
    pass
else:
    # Default path: original nanobind bindings.
    from ._rocisa import *  # noqa: F401,F403
    from . import _rocisa

    # Register nanobind submodules under the rocisa.* namespace so that
    # `from rocisa.enum import X` and `import rocisa.instruction as ri` work.
    for _name, _obj in vars(_rocisa).items():
        if isinstance(_obj, types.ModuleType) and not _name.startswith("_"):
            sys.modules.setdefault(f"rocisa.{_name}", _obj)

    # Staleness check: only active in source builds.
    # Pre-built packages (wheels, apt) lack _build_info.py — the import is
    # silently skipped. Catching ImportError (not just ModuleNotFoundError)
    # because Python 3.10 raises ImportError for missing relative submodules.
    # The intentional staleness ImportError is raised outside the try/except
    # so it is never swallowed.
    _bi = None
    try:
        from . import _build_info as _bi
    except ImportError:
        pass  # Pre-built package — no source tree, skip check

    if _bi is not None:
        from pathlib import Path

        _so = Path(_rocisa.__file__)
        # Scan rocisa sources and stinkytofu asm-IR sources (since _rocisa.so
        # links libstinkytofu.so for the toStinkyTofuModule / emitAssembly path).
        # Both roots are populated by CMake; an empty one signals a malformed
        # _build_info.py. Warn (rather than scan Path("") == the CWD) and skip it,
        # so a regression surfaces instead of silently disabling the check.
        # Excluded from the stinkytofu scan:
        #   - tests/         — test code is never compiled into .so
        #   - python_module/ — only compiled into _stinkytofu.so (Python bindings)
        #   - src/ir/logical  — logical IR is only used by _stinkytofu.so (left
        #                       path); _rocisa.so never touches logical modules.
        _roots = []
        for _name, _root in (("rocisa", _bi.SOURCE_ROOT), ("stinkytofu", _bi.STINKYTOFU_SOURCE_ROOT)):
            if _root:
                _roots.append(Path(_root))
            else:
                import warnings

                warnings.warn(
                    f"rocisa staleness check: {_name} source root is unset in "
                    f"_build_info.py; skipping it. Rebuild with: invoke rocisa",
                    stacklevel=2,
                )
        _st_root = Path(_bi.STINKYTOFU_SOURCE_ROOT) if _bi.STINKYTOFU_SOURCE_ROOT else None
        _st_skip = {_st_root / "tests", _st_root / "src" / "ir" / "logical", _st_root / "python_module"} if _st_root else set()
        _all_stale = _find_stale_sources(_so, _roots, _bi.BUILD_DIR)
        _stale = [s for s in _all_stale if not any(Path(s).is_relative_to(sk) for sk in _st_skip)]
        if _stale:
            _preview = _stale[:3] + (["..."] if len(_stale) > 3 else [])
            raise ImportError(
                "rocisa C++ sources are newer than the built _rocisa.so — bindings are stale.\n"
                f"  Modified: {', '.join(_preview)}\n"
                "  Rebuild:  invoke rocisa"
            )
        del _bi, _so, _stale, _all_stale, _roots, _name, _root, _st_root, _st_skip, Path


def hasStinkyTofuBackend() -> bool:
    """Return True if rocisa was built with StinkyTofu backend support."""
    return hasattr(_rocisa, "isSupportedByStinkyTofu")
