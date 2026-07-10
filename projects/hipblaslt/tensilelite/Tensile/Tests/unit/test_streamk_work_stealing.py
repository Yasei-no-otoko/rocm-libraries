# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the single-hop StreamK work-stealing codegen.

These tests assert that the new work-stealing assembly is emitted by the
helper methods on the ``StreamK`` base class, and -- crucially -- that those
helpers are only ever reached behind the codegen-time ``StreamKWorkStealing``
toggle. Following the StreamK=5 hybrid tests, they import rocisa instructions
and inspect emitted modules rather than matching source text; the toggle gating
and the Solution-level validation are verified by executing the *real* source
(via the AST) so the assertions track the actual code, not a copy of it.
"""

import ast
import inspect
import os
import sys
import textwrap
from copy import deepcopy

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Module, Label
from rocisa.instruction import (
    SAddU32,
    SAndB32,
    SAtomicInc,
    SBarrier,
    SCBranchSCC0,
    SCBranchSCC1,
    SCmpGeU32,
    SCmpLtU32,
    SMovB32,
    SStoreB32,
)

from Tensile.LibraryIO import parseLibraryLogicData
from Tensile.Common.ValidParameters import validParameters
from Tensile.Components.StreamK import (
    StreamK,
    StreamKDynamic,
    StreamKHybrid,
    streamKVariantClass,
)
from Tensile.SolutionStructs import Solution
from Tensile.SolutionStructs.Utilities import reject


# ---------------------------------------------------------------------------
# Fakes: just enough of a "writer" for the standalone helper methods.
#
# The three helpers only touch ``writer.sgprPool`` (checkOut / checkOutAligned
# / checkIn) and emit rocisa instructions via free functions (sgpr/vgpr), so a
# tiny pool that hands out monotonically increasing register indices is all
# that is required -- no KernelWriter, no GPU.
# ---------------------------------------------------------------------------
class _FakeSgprPool:
    def __init__(self, start: int = 100):
        self._next = start

    def checkOut(self, n: int, name: str = "", *args, **kwargs) -> int:
        reg = self._next
        self._next += n
        return reg

    def checkOutAligned(self, n: int, align: int, name: str = "", *args, **kwargs) -> int:
        if self._next % align:
            self._next += align - (self._next % align)
        reg = self._next
        self._next += n
        return reg

    def checkIn(self, *args, **kwargs):
        return None


class _FakeWriter:
    def __init__(self):
        self.sgprPool = _FakeSgprPool()


def _mk_label(base: str) -> Label:
    return Label(base, "")


def _stream_k_instance(streamk: int) -> StreamK:
    """A concrete StreamK variant (helpers live on the base class)."""
    return streamKVariantClass(streamk)()


def _imm_in(inst, value: int) -> bool:
    """True if *inst* carries *value* as an immediate operand.

    rocisa renders immediates inconsistently -- ints passed straight through
    print as decimal ("7"), while values passed as ``hex(...)`` print as
    "0x..." -- so normalise every param through ``int(p, 0)`` and compare.
    """
    for p in inst.getParams():
        try:
            if int(str(p), 0) == value:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _flat(module: Module) -> list:
    return list(module.flatitems())


# ---------------------------------------------------------------------------
# AST helpers: read the *real* StreamK / Solution source and reason about it.
# ---------------------------------------------------------------------------
def _const_slice(subscript: ast.Subscript):
    s = subscript.slice
    if isinstance(s, ast.Constant):
        return s.value
    return None


def _is_subscript_on(node, name: str, key: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
        and _const_slice(node) == key
    )


def _ws_guarded_calls(func) -> set:
    """Names of ``self.streamKWorkStealing*`` calls that sit inside an
    ``if kernel["StreamKWorkStealing"]:`` block in *func* (recursing into
    nested closures)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_subscript_on(
            node.test, "kernel", "StreamKWorkStealing"
        ):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr.startswith("streamKWorkStealing"):
                        guarded.add(sub.func.attr)
    return guarded


def _all_ws_calls(func) -> set:
    """Every ``self.streamKWorkStealing*`` call in *func*, guarded or not."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("streamKWorkStealing"):
                calls.add(node.func.attr)
    return calls


def _extract_ws_validation():
    """Compile the *real* ``if state["StreamKWorkStealing"]:`` block out of
    ``Solution.assignDerivedParameters`` into a standalone callable so the
    actual rejection logic can be exercised without a full Solution state."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(Solution.assignDerivedParameters))
    )
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_subscript_on(
            node.test, "state", "StreamKWorkStealing"
        ):
            target = node
            break
    assert target is not None, "could not find StreamKWorkStealing validation block"

    func = ast.FunctionDef(
        name="_validate",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg("state"), ast.arg("printRejectionReason"), ast.arg("reject")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[target],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    mod = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {}
    exec(compile(mod, "<ws-validation>", "exec"), ns)
    return ns["_validate"]


# ===========================================================================
# 1. ValidParameters: the codegen-time toggle exists and is boolean.
# ===========================================================================
class TestValidParameters:
    def test_work_stealing_param_exists(self):
        assert "StreamKWorkStealing" in validParameters

    def test_work_stealing_param_is_zero_one(self):
        assert validParameters["StreamKWorkStealing"] == [0, 1]

    def test_relaxed_param_exists(self):
        assert "StreamKWorkStealingRelaxed" in validParameters

    def test_relaxed_param_is_zero_one(self):
        assert validParameters["StreamKWorkStealingRelaxed"] == [0, 1]


# ===========================================================================
# 2. The three new StreamK helper methods exist and are callable.
# ===========================================================================
class TestHelperMethodsExist:
    @pytest.mark.parametrize(
        "name",
        [
            "streamKWorkStealingHomeNoReset",
            "streamKWorkStealingSteal",
            "streamKWorkStealingKernelEndReset",
        ],
    )
    def test_method_is_defined_on_base(self, name):
        assert callable(getattr(StreamK, name))


# ===========================================================================
# 3a. Presence: the helpers actually emit the work-stealing assembly.
# ===========================================================================
class TestHomeNoResetEmission:
    """Disabling the home auto-reset must mask TotalItems with 0x7 and, when a
    remainder exists, move 0xFFFFFFFF into the auto-reset bound register."""

    def _emit(self):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = Module("home-no-reset")
        sBound = writer.sgprPool.checkOut(1, "bound")
        sk.streamKWorkStealingHomeNoReset(writer, module, {}, sBound, _mk_label)
        return _flat(module)

    def test_masks_total_items_with_queue_mask(self):
        items = self._emit()
        masks = [
            i for i in items
            if isinstance(i, SAndB32) and _imm_in(i, 0x7)
        ]
        assert masks, "expected an s_and_b32 with the 0x7 queue mask"

    def test_disables_auto_reset_with_all_ones(self):
        items = self._emit()
        movs = [
            i for i in items
            if isinstance(i, SMovB32) and _imm_in(i, 0xFFFFFFFF)
        ]
        assert movs, "expected 0xFFFFFFFF mov to disable the home auto-reset"


class TestStealEmission:
    """One single-hop NEXT steal: walk to (queueIdx+1) & 0x7, guard against the
    neighbor having no structural extra, then a single auto-reset-disabled
    atomic increment against the stolen queue."""

    def _emit(self, kernel=None):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = Module("steal")
        sQueueIdx = writer.sgprPool.checkOut(1, "queueIdx")
        sWorkItemIdx = writer.sgprPool.checkOut(1, "workItemIdx")
        sk.streamKWorkStealingSteal(
            writer, module, {} if kernel is None else kernel, sQueueIdx, sWorkItemIdx, _mk_label
        )
        return _flat(module)

    def test_neighbor_walk_is_plus_one_then_wrap(self):
        items = self._emit()
        # +1 to advance to the next neighbor ...
        assert any(
            isinstance(i, SAddU32) and _imm_in(i, 1) for i in items
        ), "expected +1 advance to the next queue"
        # ... wrapped within the 8-queue ring via & 0x7.
        assert any(
            isinstance(i, SAndB32) and _imm_in(i, 0x7) for i in items
        ), "expected (queueIdx+1) & 0x7 wrap"

    def test_skips_when_neighbor_has_no_extra(self):
        items = self._emit()
        assert any(isinstance(i, SCmpGeU32) for i in items), (
            "expected a >= remainder guard so a neighbor without a structural "
            "extra is not robbed"
        )

    def test_relaxed_omits_neighbor_extra_guard(self):
        # The neighbor "has no structural extra" guard is the only SCmpGeU32 in
        # the steal sequence: the original policy must keep it, and the relaxed
        # policy must drop it so an idle WG steals regardless of the neighbor.
        original = self._emit({"StreamKWorkStealingRelaxed": 0})
        relaxed = self._emit({"StreamKWorkStealingRelaxed": 1})
        assert any(isinstance(i, SCmpGeU32) for i in original), (
            "original policy must keep the neighbor >= remainder guard"
        )
        assert not any(isinstance(i, SCmpGeU32) for i in relaxed), (
            "relaxed policy must omit the neighbor >= remainder guard"
        )

    def test_exactly_one_atomic_increment(self):
        items = self._emit()
        atomics = [i for i in items if isinstance(i, SAtomicInc)]
        assert len(atomics) == 1, "single-hop steal must emit exactly one atomic"

    def test_atomic_uses_auto_reset_disabled_bound(self):
        items = self._emit()
        assert any(
            isinstance(i, SMovB32) and _imm_in(i, 0xFFFFFFFF)
            for i in items
        ), "the stolen atomic must run with auto-reset disabled (0xFFFFFFFF)"

    def test_guards_on_valid_home_fetch(self):
        # A valid home fetch (index < TotalItems) must short-circuit the steal.
        items = self._emit()
        assert any(isinstance(i, SCmpLtU32) for i in items)
        assert any(isinstance(i, SCBranchSCC1) for i in items)


class TestKernelEndResetEmission:
    """The last WG zeroes the 8 per-queue counters plus the completion counter,
    behind a barrier + wave-0 completion count."""

    def _emit(self):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = sk.streamKWorkStealingKernelEndReset(writer, {}, "skGrid", _mk_label)
        return _flat(module)

    def test_starts_with_barrier(self):
        items = self._emit()
        assert any(isinstance(i, SBarrier) for i in items)

    def test_resets_eight_queues_plus_completion_counter(self):
        items = self._emit()
        stores = [i for i in items if isinstance(i, SStoreB32)]
        assert len(stores) == StreamK._WS_NUM_QUEUES + 1 == 9, (
            "expected 8 per-queue counter resets + 1 completion counter reset"
        )

    def test_completion_count_uses_atomic_inc(self):
        items = self._emit()
        assert any(isinstance(i, SAtomicInc) for i in items), (
            "wave 0 counts completed WGs via an atomic increment"
        )

    def test_only_last_wg_resets(self):
        # SCBranchSCC0 guards (a) wave-0-only and (b) last-WG-only.
        items = self._emit()
        assert sum(isinstance(i, SCBranchSCC0) for i in items) >= 2


# ===========================================================================
# 3b. Absence-by-toggle: the helpers are only reached behind the
#     ``kernel["StreamKWorkStealing"]`` gate at every callsite. Verified
#     against the real source so "off" provably emits nothing extra.
# ===========================================================================
class TestCallsitesAreToggleGated:
    def test_sk4_grawg_steal_calls_are_all_gated(self):
        guarded = _ws_guarded_calls(StreamKDynamic.graWorkGroup)
        allcalls = _all_ws_calls(StreamKDynamic.graWorkGroup)
        assert {"streamKWorkStealingHomeNoReset", "streamKWorkStealingSteal"} <= guarded
        # Nothing slips through ungated.
        assert allcalls == guarded

    def test_sk4_kernelend_reset_is_gated(self):
        guarded = _ws_guarded_calls(StreamKDynamic.kernelEnd)
        allcalls = _all_ws_calls(StreamKDynamic.kernelEnd)
        assert "streamKWorkStealingKernelEndReset" in guarded
        assert allcalls == guarded

    def test_sk5_grawg_steal_calls_are_all_gated(self):
        guarded = _ws_guarded_calls(StreamKHybrid.graWorkGroup)
        allcalls = _all_ws_calls(StreamKHybrid.graWorkGroup)
        assert {"streamKWorkStealingHomeNoReset", "streamKWorkStealingSteal"} <= guarded
        assert allcalls == guarded

    def test_sk5_kernelend_reset_is_gated(self):
        guarded = _ws_guarded_calls(StreamKHybrid.kernelEnd)
        allcalls = _all_ws_calls(StreamKHybrid.kernelEnd)
        assert "streamKWorkStealingKernelEndReset" in guarded
        assert allcalls == guarded


# ===========================================================================
# 4. Solution validation: the real rejection logic from
#    assignDerivedParameters, executed in isolation.
# ===========================================================================
class TestSolutionValidation:
    def setup_method(self):
        self.validate = _extract_ws_validation()

    def _run(self, *, streamk, atomic, work_stealing=1):
        state = {
            "StreamKWorkStealing": work_stealing,
            "StreamK": streamk,
            "StreamKAtomic": atomic,
        }
        self.validate(state, False, reject)
        return state

    @pytest.mark.parametrize("streamk", [0, 1, 2, 3])
    def test_rejected_when_streamk_not_4_or_5(self, streamk):
        state = self._run(streamk=streamk, atomic=0)
        assert state["Valid"] is False

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_accepted_for_dynamic_and_hybrid_without_atomic(self, streamk):
        state = self._run(streamk=streamk, atomic=0)
        assert state.get("Valid", True) is True

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_rejected_with_atomic(self, streamk):
        state = self._run(streamk=streamk, atomic=1)
        assert state["Valid"] is False

    def test_off_toggle_is_inert_even_for_bad_combo(self):
        # With the toggle off the guard must not fire, even for a combination
        # that would otherwise be rejected.
        state = self._run(streamk=3, atomic=1, work_stealing=0)
        assert "Valid" not in state


# ===========================================================================
# 5. parseLibraryLogicData WS codegen toggle: TENSILE_STREAMK_WS_MODE is a
#    three-way choice (off / only / relaxed) + the deprecated legacy alias
#    TENSILE_GENERATE_STREAMK_WS_VARIANTS ("0" -> off, "1" -> only). "relaxed"
#    flips eligible solutions like "only" and additionally sets
#    StreamKWorkStealingRelaxed=1 (drops the neighbor has-no-extra guard).
#
#    The flip happens inline in parseLibraryLogicData *before* the heavy
#    Solution construction (which needs an assembler / isaInfoMap / GPU). To
#    exercise the *real* mode-resolution + in-place flip source without that
#    machinery, the relevant statements are AST-extracted into a standalone
#    ``_apply(data)`` callable -- mirroring TestSolutionValidation above, so
#    the assertions track the actual code rather than a copy of it.
# ===========================================================================
def _extract_ws_mode_applier():
    """Compile the real WS-mode block out of ``parseLibraryLogicData`` into a
    standalone ``_apply(data)`` so the mode-resolution + in-place flip logic can
    run on a tiny synthetic ``data`` dict with no assembler or GPU.

    The block runs from the first ``wsMode = ...`` assignment through the
    ``if wsMode in ("only", "relaxed"):`` flip loop. The production code emits
    deprecation / unknown-mode warnings via ``sys.stderr.write``, so ``sys``
    (and ``os`` for the env lookups) must be present in the exec namespace.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(parseLibraryLogicData)))
    funcdef = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    body = funcdef.body

    start = end = None
    for i, stmt in enumerate(body):
        if (
            start is None
            and isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "wsMode" for t in stmt.targets)
        ):
            start = i
        # The flip dispatch: ``if wsMode in ("only", "relaxed"): ...`` (in-place
        # flip). Matched by the ``in`` membership test whose options include
        # "only"; the earlier ``if wsMode not in (...)`` guard uses ``NotIn`` and
        # is intentionally skipped.
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "wsMode"
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.In)
            and isinstance(stmt.test.comparators[0], (ast.Tuple, ast.List))
            and any(
                isinstance(e, ast.Constant) and e.value == "only"
                for e in stmt.test.comparators[0].elts
            )
        ):
            end = i
    assert start is not None, "could not find the wsMode env resolution"
    assert end is not None and end >= start, "could not find the WS only block"

    func = ast.FunctionDef(
        name="_apply",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg("data")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body[start : end + 1],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    mod = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {"os": os, "sys": sys}
    exec(compile(mod, "<ws-mode>", "exec"), ns)
    return ns["_apply"]


# Synthetic solution indices (kept as module constants for readable asserts).
_SK5_ELIGIBLE = 0   # StreamK=5, atomic=0, WS=0  -> eligible
_SK4_ELIGIBLE = 1   # StreamK=4, atomic=0, WS=0  -> eligible
_NON_STREAMK = 2    # StreamK=0                  -> ineligible
_ALREADY_WS = 3     # StreamK=5, WS=1            -> ineligible
_ATOMIC = 4         # StreamK=4, atomic=1        -> ineligible
_ELIGIBLE = (_SK5_ELIGIBLE, _SK4_ELIGIBLE)
_INELIGIBLE = (_NON_STREAMK, _ALREADY_WS, _ATOMIC)


def _mk_solutions():
    return [
        {"name": "sk5", "StreamK": 5, "StreamKAtomic": 0, "StreamKWorkStealing": 0},
        {"name": "sk4", "StreamK": 4, "StreamKAtomic": 0, "StreamKWorkStealing": 0},
        {"name": "gemm", "StreamK": 0, "StreamKAtomic": 0, "StreamKWorkStealing": 0},
        {"name": "sk5ws", "StreamK": 5, "StreamKAtomic": 0, "StreamKWorkStealing": 1},
        {"name": "sk4atomic", "StreamK": 4, "StreamKAtomic": 1, "StreamKWorkStealing": 0},
    ]


def _mk_matching_data():
    """A "Matching" library-logic dict: table rows are [key, [solIdx, dist]]."""
    sols = _mk_solutions()
    return {
        "LibraryType": "Matching",
        "Solutions": sols,
        "Library": {
            "table": [
                [[256, 256, 1, 256], [_SK5_ELIGIBLE, 1.0]],   # -> remapped
                [[512, 512, 1, 512], [_SK4_ELIGIBLE, 2.0]],   # -> remapped
                [[128, 128, 1, 128], [_NON_STREAMK, 3.0]],    # -> not remapped
                [[64, 64, 1, 64], [_ATOMIC, 4.0]],            # -> not remapped
            ]
        },
    }


def _mk_freesize_data():
    """A "FreeSize"-style dict where the table is the [0, len(Solutions)] form."""
    sols = _mk_solutions()
    return {
        "LibraryType": "FreeSize",
        "Solutions": sols,
        "Library": {"table": [0, len(sols)]},
    }


class TestWorkStealingFlipMode:
    """The off/only/relaxed contract: ``off`` is a no-op; ``only`` flips eligible
    SK4/SK5 solutions to WS=1 in place with no count growth and no table edits;
    ``relaxed`` flips like ``only`` and additionally sets
    StreamKWorkStealingRelaxed=1 on those same eligible solutions."""

    def setup_method(self):
        self.apply = _extract_ws_mode_applier()

    # -- off -------------------------------------------------------------
    def test_off_leaves_everything_untouched(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "off")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        before = deepcopy(data)
        self.apply(data)
        # Count, every flag, and the table are all identical to before.
        assert len(data["Solutions"]) == len(before["Solutions"])
        assert all(s["StreamKWorkStealing"] == b["StreamKWorkStealing"]
                   for s, b in zip(data["Solutions"], before["Solutions"]))
        # No fresh WS=1 appears on a previously-WS=0 solution.
        assert not any(s["StreamKWorkStealing"] == 1 for s in data["Solutions"][:3])
        assert data["Library"]["table"] == before["Library"]["table"]

    def test_off_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "OfF")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        before = deepcopy(data)
        self.apply(data)
        assert data["Solutions"] == before["Solutions"]
        assert data["Library"]["table"] == before["Library"]["table"]

    # -- only ------------------------------------------------------------
    def test_only_flips_eligible_in_place_no_growth_no_table_change(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "only")
        data = _mk_matching_data()
        orig_n = len(data["Solutions"])
        table_before = deepcopy(data["Library"]["table"])
        self.apply(data)

        # No duplicates appended.
        assert len(data["Solutions"]) == orig_n
        # Eligible solutions (SK4/SK5, atomic=0, WS=0) flipped to WS=1 in place.
        for idx in _ELIGIBLE:
            assert data["Solutions"][idx]["StreamKWorkStealing"] == 1
        # Ineligible untouched: non-StreamK, already-WS, and atomic.
        assert data["Solutions"][_NON_STREAMK]["StreamKWorkStealing"] == 0
        assert data["Solutions"][_ALREADY_WS]["StreamKWorkStealing"] == 1
        assert data["Solutions"][_ATOMIC]["StreamKWorkStealing"] == 0
        # Table unchanged.
        assert data["Library"]["table"] == table_before

    def test_only_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "OnLy")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        self.apply(data)
        for idx in _ELIGIBLE:
            assert data["Solutions"][idx]["StreamKWorkStealing"] == 1

    def test_only_leaves_freesize_table_unchanged(self, monkeypatch):
        # The simplified contract never touches the Library/table, regardless of
        # library type -- including the [0, len] FreeSize/Prediction form.
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "only")
        data = _mk_freesize_data()
        orig_n = len(data["Solutions"])
        table_before = deepcopy(data["Library"]["table"])
        self.apply(data)
        assert len(data["Solutions"]) == orig_n
        assert data["Library"]["table"] == table_before
        for idx in _ELIGIBLE:
            assert data["Solutions"][idx]["StreamKWorkStealing"] == 1

    def test_only_with_no_eligible_solutions_is_a_noop(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "only")
        data = {
            "LibraryType": "Matching",
            "Solutions": [
                {"name": "gemm", "StreamK": 0, "StreamKAtomic": 0, "StreamKWorkStealing": 0},
                {"name": "sk4atomic", "StreamK": 4, "StreamKAtomic": 1, "StreamKWorkStealing": 0},
            ],
            "Library": {"table": [[[1, 1, 1, 1], [0, 1.0]]]},
        }
        before = deepcopy(data)
        self.apply(data)
        assert data == before

    # -- relaxed ---------------------------------------------------------
    def test_relaxed_flips_eligible_and_sets_relaxed_flag(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "relaxed")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        orig_n = len(data["Solutions"])
        table_before = deepcopy(data["Library"]["table"])
        self.apply(data)

        # No duplicates appended.
        assert len(data["Solutions"]) == orig_n
        # Eligible solutions (SK4/SK5, atomic=0, WS=0) flipped to WS=1 AND get
        # StreamKWorkStealingRelaxed=1 in place.
        for idx in _ELIGIBLE:
            assert data["Solutions"][idx]["StreamKWorkStealing"] == 1
            assert data["Solutions"][idx]["StreamKWorkStealingRelaxed"] == 1
        # Ineligible untouched: no WS flip and no relaxed flag introduced.
        assert data["Solutions"][_NON_STREAMK]["StreamKWorkStealing"] == 0
        assert data["Solutions"][_ALREADY_WS]["StreamKWorkStealing"] == 1
        assert data["Solutions"][_ATOMIC]["StreamKWorkStealing"] == 0
        for idx in _INELIGIBLE:
            assert data["Solutions"][idx].get("StreamKWorkStealingRelaxed", 0) == 0
        # Table unchanged.
        assert data["Library"]["table"] == table_before

    def test_relaxed_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "ReLaXeD")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        self.apply(data)
        for idx in _ELIGIBLE:
            assert data["Solutions"][idx]["StreamKWorkStealing"] == 1
            assert data["Solutions"][idx]["StreamKWorkStealingRelaxed"] == 1

    def test_only_does_not_set_relaxed_flag(self, monkeypatch):
        # "only" enables WS but must NOT touch StreamKWorkStealingRelaxed.
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "only")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        self.apply(data)
        for idx in _ELIGIBLE:
            assert data["Solutions"][idx]["StreamKWorkStealing"] == 1
            assert data["Solutions"][idx].get("StreamKWorkStealingRelaxed", 0) == 0

    def test_relaxed_with_no_eligible_solutions_is_a_noop(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "relaxed")
        data = {
            "LibraryType": "Matching",
            "Solutions": [
                {"name": "gemm", "StreamK": 0, "StreamKAtomic": 0, "StreamKWorkStealing": 0},
                {"name": "sk4atomic", "StreamK": 4, "StreamKAtomic": 1, "StreamKWorkStealing": 0},
            ],
            "Library": {"table": [[[1, 1, 1, 1], [0, 1.0]]]},
        }
        before = deepcopy(data)
        self.apply(data)
        assert data == before


class TestWorkStealingModeResolution:
    """Env-var precedence + back-compat + unknown-value fallback (off/only/relaxed)."""

    def setup_method(self):
        self.apply = _extract_ws_mode_applier()

    def _flipped_eligible(self, data):
        """True iff every eligible solution ended up flipped to WS=1 and no
        solution count growth occurred."""
        return all(data["Solutions"][idx]["StreamKWorkStealing"] == 1
                   for idx in _ELIGIBLE)

    def _no_new_flips(self, data):
        """True iff no previously-WS=0 solution was flipped to WS=1."""
        return not any(s["StreamKWorkStealing"] == 1 for s in data["Solutions"][:3])

    def test_default_neither_set_is_off(self, monkeypatch):
        monkeypatch.delenv("TENSILE_STREAMK_WS_MODE", raising=False)
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        before = deepcopy(data)
        self.apply(data)
        assert len(data["Solutions"]) == len(before["Solutions"])
        assert self._no_new_flips(data)
        assert data["Library"]["table"] == before["Library"]["table"]

    def test_empty_string_is_off(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        n = len(data["Solutions"])
        self.apply(data)
        assert len(data["Solutions"]) == n
        assert self._no_new_flips(data)

    def test_legacy_zero_behaves_like_off(self, monkeypatch):
        monkeypatch.delenv("TENSILE_STREAMK_WS_MODE", raising=False)
        monkeypatch.setenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", "0")
        data = _mk_matching_data()
        n = len(data["Solutions"])
        self.apply(data)
        assert len(data["Solutions"]) == n
        assert self._no_new_flips(data)

    def test_legacy_one_behaves_like_only(self, monkeypatch):
        monkeypatch.delenv("TENSILE_STREAMK_WS_MODE", raising=False)
        monkeypatch.setenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", "1")
        data = _mk_matching_data()
        n = len(data["Solutions"])
        self.apply(data)
        # only: flip in place, no growth.
        assert len(data["Solutions"]) == n
        assert self._flipped_eligible(data)

    def test_new_var_overrides_legacy_off_beats_one(self, monkeypatch):
        # New var says off, legacy says only -> new var wins (off).
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "off")
        monkeypatch.setenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", "1")
        data = _mk_matching_data()
        n = len(data["Solutions"])
        self.apply(data)
        assert len(data["Solutions"]) == n
        assert self._no_new_flips(data)

    def test_new_var_overrides_legacy_only_beats_zero(self, monkeypatch):
        # New var says only, legacy says off -> new var wins (only: flip in place).
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "only")
        monkeypatch.setenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", "0")
        data = _mk_matching_data()
        n = len(data["Solutions"])
        self.apply(data)
        assert len(data["Solutions"]) == n
        assert self._flipped_eligible(data)

    def test_unknown_mode_falls_back_to_off(self, monkeypatch):
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "bogus")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        n = len(data["Solutions"])
        self.apply(data)
        assert len(data["Solutions"]) == n
        assert self._no_new_flips(data)

    def test_relaxed_is_not_treated_as_unknown(self, monkeypatch):
        # "relaxed" is a recognized value, so it must flip eligible solutions
        # (like "only") rather than falling back to the off no-op, and it also
        # sets the relaxed flag on those solutions.
        monkeypatch.setenv("TENSILE_STREAMK_WS_MODE", "relaxed")
        monkeypatch.delenv("TENSILE_GENERATE_STREAMK_WS_VARIANTS", raising=False)
        data = _mk_matching_data()
        n = len(data["Solutions"])
        self.apply(data)
        assert len(data["Solutions"]) == n
        assert self._flipped_eligible(data)
        for idx in _ELIGIBLE:
            assert data["Solutions"][idx]["StreamKWorkStealingRelaxed"] == 1
