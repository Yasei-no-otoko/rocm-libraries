################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the ``TDMStoreInst`` solution parameter.

``TDMStoreInst`` gates the gfx1250 full-tile Tensor Data Mover epilogue
store (the entire padded tile runs through the epilogue, is staged to LDS, and
is flushed with a single ``tensor_store_from_lds`` whose descriptor clamps the
OOB/dummy region at write-out).

These tests cover the parameter's registration (valid values, default off,
kernel-name / no-dedup token) and the ``HasTDM`` transport guard in
``Solution.assignDerivedParameters`` -- exercised through the lifted-out
:func:`_validateTDMStoreInst` helper, mirroring ``test_MXScaleLayoutDerivation``.
All CPU-only; no GPU required.
"""

import pytest

from Tensile.SolutionStructs.Solution import _validateTDMStoreInst
from Tensile.Common.ValidParameters import validParameters
from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
from Tensile.Common.RequiredParameters import getRequiredParametersMin


def _caps(hasTDM=True):
    return {"HasTDM": hasTDM}


# ---------------------------------------------------------------------------
# Parameter registration
# ---------------------------------------------------------------------------

class TestTDMStoreInstRegistration:

    def test_valid_values(self):
        assert validParameters["TDMStoreInst"] == [False, True]

    def test_default_is_false(self):
        defaults = {list(d.keys())[0]: list(d.values())[0]
                    for d in defaultBenchmarkCommonParameters}
        assert "TDMStoreInst" in defaults
        assert defaults["TDMStoreInst"] == [False]

    def test_in_required_parameters_min(self):
        # Presence in RequiredParametersMin makes TDMStoreInst part of the kernel
        # name token so solutions differing only in TDMStoreInst are not
        # deduplicated (mirrors TDMSplit).
        assert "TDMStoreInst" in getRequiredParametersMin()


# ---------------------------------------------------------------------------
# HasTDM transport guard (mirrors the TDMInst reject)
# ---------------------------------------------------------------------------

class TestTDMStoreInstHasTDMGuard:

    def test_reject_without_hastdm(self):
        state = {"TDMStoreInst": True}
        assert _validateTDMStoreInst(state, _caps(hasTDM=False), False) is False
        assert state["Valid"] is False

    def test_pass_with_hastdm(self):
        state = {"TDMStoreInst": True}
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is True
        assert state.get("Valid") is not False

    def test_disabled_bypasses_guard(self):
        # TDMStoreInst=False must never reject, even on an arch without HasTDM.
        state = {"TDMStoreInst": False}
        assert _validateTDMStoreInst(state, _caps(hasTDM=False), False) is True
        assert state.get("Valid") is not False


# ---------------------------------------------------------------------------
# Correctness-hardening guards (P0/P1)
#
# The full-tile flush stages the entire MT0*MT1 tile M-contiguous into an
# LDS scratch and writes AddressD with one non-atomic tensor_store_from_lds.
# These guards reject the combinations that would break that assumption. All
# reads use state.get with valid-config defaults, so a "valid" base state must
# populate every field a guard inspects; each test flips exactly one field.
# ---------------------------------------------------------------------------

class _FakeDataType:
    """Minimal DestDataType stub exposing numBytes()."""
    def __init__(self, numBytes):
        self._numBytes = numBytes

    def numBytes(self):
        return self._numBytes


def _valid_state(**overrides):
    """A fully-valid TDMStoreInst state (bf16 out, 16x16 MI, GSU1, no StreamK,
    256x256 MacroTile that fits in a 160 KiB LDS). Override single fields to
    exercise one guard at a time."""
    state = {
        "TDMStoreInst": True,
        "UseSubtileImpl": True,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "ProblemType": {"DestDataType": _FakeDataType(2)},  # bf16 -> 2 bytes
        "GlobalSplitU": 1,
        "StreamK": 0,
        "StreamKAtomic": 0,
        "ClusterDim": [1, 1],
        "MacroTile0": 256,
        "MacroTile1": 256,
        "MaxLDS": 163840,  # gfx1250 device LDS (160 KiB): 256*256*2 = 128 KiB fits
    }
    state.update(overrides)
    return state


class TestTDMStoreInstHardeningGuards:

    def test_valid_full_state_passes(self):
        state = _valid_state()
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is True
        assert state.get("Valid") is not False

    def test_reject_without_usesubtileimpl(self):
        state = _valid_state(UseSubtileImpl=False)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_reject_non_16x16_mi(self):
        state = _valid_state(MatrixInstN=32)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_reject_fractional_byte_dest(self):
        # F4 output (0.5 B) -> data_size encoding would KeyError in codegen.
        state = _valid_state(ProblemType={"DestDataType": _FakeDataType(0.5)})
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_reject_8byte_dest(self):
        state = _valid_state(ProblemType={"DestDataType": _FakeDataType(8)})
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    @pytest.mark.parametrize("bpe", [1, 2, 4])
    def test_pass_supported_dest_bytes(self, bpe):
        # 1/2/4-byte outputs are allowed; keep the tile small so LDS always fits.
        state = _valid_state(ProblemType={"DestDataType": _FakeDataType(bpe)},
                             MacroTile0=128, MacroTile1=128)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is True
        assert state.get("Valid") is not False

    def test_reject_globalsplitu_gt1(self):
        state = _valid_state(GlobalSplitU=2)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_reject_streamk(self):
        state = _valid_state(StreamK=3)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_reject_atomic(self):
        state = _valid_state(StreamKAtomic=1)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_reject_clusterdim_multicast(self):
        state = _valid_state(ClusterDim=[2, 1])
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_reject_lds_capacity_overflow(self):
        # 256*256*2 = 128 KiB > 64 KiB MaxLDS -> reject.
        state = _valid_state(MaxLDS=65536)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is False
        assert state["Valid"] is False

    def test_pass_lds_capacity_fits(self):
        # Same tile, 160 KiB LDS -> fits.
        state = _valid_state(MaxLDS=163840)
        assert _validateTDMStoreInst(state, _caps(hasTDM=True), False) is True
        assert state.get("Valid") is not False