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
"""Unit tests for the ``TDMStore`` solution parameter.

``TDMStore`` gates the gfx1250 whole-MacroTile Tensor Data Mover epilogue
store (the whole padded tile runs through the epilogue, is staged to LDS, and
is flushed with a single ``tensor_store_from_lds`` whose descriptor clamps the
OOB/dummy region at write-out).

These tests cover the parameter's registration (valid values, default off,
kernel-name / no-dedup token) and the ``HasTDM`` transport guard in
``Solution.assignDerivedParameters`` -- exercised through the lifted-out
:func:`_validateTDMStore` helper, mirroring ``test_MXScaleLayoutDerivation``.
All CPU-only; no GPU required.
"""

import pytest

from Tensile.SolutionStructs.Solution import _validateTDMStore
from Tensile.Common.ValidParameters import validParameters
from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
from Tensile.Common.RequiredParameters import getRequiredParametersMin


def _caps(hasTDM=True):
    return {"HasTDM": hasTDM}


# ---------------------------------------------------------------------------
# Parameter registration
# ---------------------------------------------------------------------------

class TestTDMStoreRegistration:

    def test_valid_values(self):
        assert validParameters["TDMStore"] == [False, True]

    def test_default_is_false(self):
        defaults = {list(d.keys())[0]: list(d.values())[0]
                    for d in defaultBenchmarkCommonParameters}
        assert "TDMStore" in defaults
        assert defaults["TDMStore"] == [False]

    def test_in_required_parameters_min(self):
        # Presence in RequiredParametersMin makes TDMStore part of the kernel
        # name token so solutions differing only in TDMStore are not
        # deduplicated (mirrors TDMSplit).
        assert "TDMStore" in getRequiredParametersMin()


# ---------------------------------------------------------------------------
# HasTDM transport guard (mirrors the TDMInst reject)
# ---------------------------------------------------------------------------

class TestTDMStoreHasTDMGuard:

    def test_reject_without_hastdm(self):
        state = {"TDMStore": True}
        assert _validateTDMStore(state, _caps(hasTDM=False), False) is False
        assert state["Valid"] is False

    def test_pass_with_hastdm(self):
        state = {"TDMStore": True}
        assert _validateTDMStore(state, _caps(hasTDM=True), False) is True
        assert state.get("Valid") is not False

    def test_disabled_bypasses_guard(self):
        # TDMStore=False must never reject, even on an arch without HasTDM.
        state = {"TDMStore": False}
        assert _validateTDMStore(state, _caps(hasTDM=False), False) is True
        assert state.get("Valid") is not False