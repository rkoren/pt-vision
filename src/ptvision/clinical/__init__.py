"""Clinical protocols, segmenters, metric registry and normative comparisons.

Importing this package registers the built-in segmenters and metrics.
"""

from ptvision.clinical import registry
from ptvision.clinical.metrics import gait as _gait_metrics  # noqa: F401  (registers metrics)
from ptvision.clinical.metrics import sts as _sts_metrics  # noqa: F401  (registers metrics)
from ptvision.clinical.segmenters import (
    gait_zeni as _gait_segmenter,  # noqa: F401  (registers segmenter)
)
from ptvision.clinical.segmenters import sts as _sts_segmenter  # noqa: F401  (registers segmenter)

__all__ = ["registry"]
