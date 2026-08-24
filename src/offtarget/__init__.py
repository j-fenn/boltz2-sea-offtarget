"""Boltz-2 against the Lounkine 2012 off-target benchmark.

A re-analysis of two course projects. The finding is about the readout, not the
model: on this dataset Boltz-2's structural confidence outputs do not separate
confirmed binders from confirmed non-binders, while its affinity outputs do.
"""
__version__ = "0.2.0"

from . import constructs, figures, metrics, pipeline, transforms  # noqa: F401
