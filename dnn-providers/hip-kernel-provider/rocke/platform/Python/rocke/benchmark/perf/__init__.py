# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.benchmark.perf - rocKE kernel performance primitives (pure, produce side).

LAYER 1 - primitives. Pure: they RETURN records / values and write nothing, so any
consumer (a local dev/agent tool OR an external perf framework) can use them without
inheriting file-writing behavior. The measurement-record ``schema`` is the seam every
consumer speaks.

This change lands the contract:
  schema.py  - measurement-record schema + validate (the seam)
  report.py  - serialize a record; diagnostic panel; diff two records

Further primitives (counters, harness, occupancy, aggregate) and the separate user
tool (``rocke.benchmark.perf.tool``) land in follow-on changes; the primitives never
import the tool.
"""

from . import report, schema  # noqa: F401

__all__ = ["report", "schema"]
