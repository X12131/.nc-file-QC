"""ncparse：NC 刀轨（G 代码）只读解析与沉头孔底面平台识别。"""

from .counterbore import (
    DEFAULT_CLUSTER_DEPTH_MM,
    DEFAULT_DEPTH_EPSILON_MM,
    DEFAULT_DEPTH_MATCH_TOLERANCE_MM,
    DEFAULT_DIAMETER_MATCH_TOLERANCE_MM,
    DEFAULT_SWEEP_MIN_ARCS,
    DEFAULT_SWEEP_MIN_FULL_CIRCLES,
    DEFAULT_SWEEP_MIN_STEP_MM,
    DEFAULT_XY_TOLERANCE_MM,
    MATCHED_BY_ARC_FLAT_SWEEP,
    MATCHED_BY_SAME_CLUSTER,
    MATCHED_BY_SIGNATURE,
    BoreCluster,
    CounterboreResult,
    analyze_nc_counterbores,
    find_counterbores,
)
from .nc_parser import (
    SEG_ARC_FLAT,
    SEG_ARC_HELICAL,
    SEG_DRILL,
    SEG_LINEAR,
    NCParseResult,
    NCSegment,
    parse_nc,
)

__version__ = "0.1.0"

__all__ = [
    "BoreCluster",
    "CounterboreResult",
    "NCParseResult",
    "NCSegment",
    "SEG_ARC_FLAT",
    "SEG_ARC_HELICAL",
    "SEG_DRILL",
    "SEG_LINEAR",
    "MATCHED_BY_ARC_FLAT_SWEEP",
    "MATCHED_BY_SAME_CLUSTER",
    "MATCHED_BY_SIGNATURE",
    "DEFAULT_CLUSTER_DEPTH_MM",
    "DEFAULT_DEPTH_EPSILON_MM",
    "DEFAULT_DEPTH_MATCH_TOLERANCE_MM",
    "DEFAULT_DIAMETER_MATCH_TOLERANCE_MM",
    "DEFAULT_SWEEP_MIN_ARCS",
    "DEFAULT_SWEEP_MIN_FULL_CIRCLES",
    "DEFAULT_SWEEP_MIN_STEP_MM",
    "DEFAULT_XY_TOLERANCE_MM",
    "analyze_nc_counterbores",
    "find_counterbores",
    "parse_nc",
    "__version__",
]
