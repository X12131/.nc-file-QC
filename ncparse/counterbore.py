"""沉头孔（counterbore）底面平台识别。

从 :mod:`ncparse.nc_parser` 的段列表里找**沉头孔底面平台**（沉头上部的扩孔平底）：

1. 按**圆弧圆心** XY 聚类（容差 ``xy_tolerance_mm``，默认 1.0mm；空间网格哈希 + 并查集）；
   为什么用圆心而不是起点：同轴心的整圈圆弧起点会沿圆周漂移（相邻段起点最多差一个弦长），
   用起点聚类会把一圈圆弧拆成多簇、或把相邻的不同孔串成一簇；圆心恒定，是可靠键。
2. 每簇取「不带 Z 的圆弧段（``ARC_FLAT``）」的**最深 Z**（最负）作为候选平台；
3. **双判据，任一命中即候选**：
   - 判据 A（``same_cluster_drill``）：该簇还存在更深的下钻 / 螺旋段——平底圆弧在螺旋下刀的
     每层收尾处都会出现，真正的平台底面是「平底圆弧比同轴心的螺旋/下钻更深」的那种；
   - 判据 B（``depth_diameter_signature``，需传入要求值）：平底圆弧段同时满足
     ``|Z| ≈ required_depth``（± ``depth_match_tolerance_mm``，默认 0.05）且
     **有效孔径** = ``刀具直径 + 2×圆弧半径 ≈ required_diameter``
     （± ``diameter_match_tolerance_mm``，默认 1.5）。
     常见布局：平台铣削与通孔钻削分属两份 NC（每份只有一把刀），同一份 NC 里
     同簇凑不齐判据 A；此时用「深度 + 直径联合签名」识别。
4. 输出 ``[(X, Y, 平台Z, 刀具直径, 圆弧半径, 有效孔径, 命中判据, 簇内点数)]``；
5. 同一 NC 内多簇平台 Z 不一致时，输出全部簇并标记 ``inconsistent=True``（不自行挑选）。

**判据 C（``arc_flat_sweep``，可开关）**

第三种平台形态——「定深平底弧扫掠」：

    1 条带 Z 螺旋弧进刀 → 数十条定深平底弧（G2/G3 不带 Z，圆心逐段漂移、半径递增）
    → 2 条整圆收尾 → 1 条带 Z 螺旋弧退出

该形态里平底圆弧同簇**没有更深的下钻/螺旋**（判据 A 永不命中），且平台 Z 与要求深度的
偏差可超出判据 B 的容差，所以两条既有判据都可能落空。判据 C 用**形态 + 包络孔径**识别：

- 按**模态 Z 分层**（同一 Z 的平底弧才可能属于同一平台底面）；
- 层内按**圆心 1mm 聚类**（扫掠时圆心逐段漂移，1mm 是「同一平台」与「相邻孔」的分界）；
- 簇内要求 ``n_arcs ≥ min_arcs`` 且至少 ``min_full_circles`` 条**整圆**（终点 == 上一段终点）；
- 簇首段的前一段必须是**同 Z 螺旋弧**（螺旋进刀）；
- 簇的参考圆心 1mm 内必须存在**更深**（Z 更负）的圆弧/下钻段（平台下面还有通孔）；
- **包络半径** ``env = max_i(|center_i - ref| + r_i)``，``ref`` = 半径最大那条弧的圆心；
  有效孔径 = 刀具直径 + 2×env，须与要求直径相差 ≤ ``diameter_tolerance_mm``。

判据 C 只负责「量出平台 Z」，不改变容差判定（容差判定在上层规则完成）。
阈值来源：形态与聚类口径来自带金标核对的实际刀轨样本；包络孔径 ±1.5mm 沿用判据 B 容差。
平底圆弧在螺旋孔的每层收尾都会出现，必须用「平底比螺旋更深」这一硬判据区分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .nc_parser import (
    SEG_ARC_FLAT,
    SEG_ARC_HELICAL,
    SEG_DRILL,
    NCSegment,
)

# 默认 XY 聚类容差（mm）
DEFAULT_XY_TOLERANCE_MM = 1.0
# 判定「更深」时允许的数值噪声（mm）：平台 Z 必须比螺旋/下钻至少深这么多
DEFAULT_DEPTH_EPSILON_MM = 0.001
# 圆弧段没有圆心（缺 I/J）时回退用起点聚类
DEFAULT_CLUSTER_DEPTH_MM = 0.05
# 判据 B：|实测平台Z| 与要求深度的匹配容差（mm）
DEFAULT_DEPTH_MATCH_TOLERANCE_MM = 0.05
# 判据 B：有效孔径与要求直径的匹配容差（mm）
DEFAULT_DIAMETER_MATCH_TOLERANCE_MM = 1.5
# 判据 C（arc_flat_sweep）：一个扫掠块最少平底弧段数。
# 实际刀轨里短收尾变体只有 2 条平底弧 + 1 条整圆，长扫掠变体可达 20~30 条；
# 取 2 是为了不漏短变体，同时靠「螺旋进刀 + 整圆收尾 + 同 XY 更深」三项把普通清角圆弧排除。
DEFAULT_SWEEP_MIN_ARCS = 2
# 判据 C：簇内最少整圆（终点 == 上一段终点）弧段数
DEFAULT_SWEEP_MIN_FULL_CIRCLES = 1
# 判据 C：「上大下小」最小台阶（mm）。平台底面半径必须比同 XY 下方更深特征的最大半径
# 至少大这么多（即平台直径比下方通孔直径大 ≥1mm）；
# 螺旋钻削中间层的平底收尾台阶恒为 0（同半径继续下钻）→ 被这一条排除。
DEFAULT_SWEEP_MIN_STEP_MM = 0.5

# 命中判据标记（BoreCluster.matched_by）
MATCHED_BY_SAME_CLUSTER = "same_cluster_drill"        # 判据 A：同簇平底 + 更深螺旋/下钻
MATCHED_BY_SIGNATURE = "depth_diameter_signature"     # 判据 B：深度 + 直径联合签名
MATCHED_BY_ARC_FLAT_SWEEP = "arc_flat_sweep"          # 判据 C：定深平底弧扫掠模板


@dataclass
class BoreCluster:
    """一个平台候选簇（按圆弧圆心聚类）。"""

    x: float
    y: float
    platform_z: Optional[float] = None          # 平底圆弧最深 Z
    helical_z: Optional[float] = None           # 螺旋段最深 Z
    drill_z: Optional[float] = None             # 下钻段最深 Z
    tool_diameter: Optional[float] = None       # 刀具直径（来自刀具头）
    arc_radius: Optional[float] = None          # 平台处平底圆弧半径
    # 判据 C 专用：簇内平底弧的**原始**半径范围（未加圆心漂移的包络）。
    # 只用于人工复核文案（「半径恒定」= 型腔底面，「半径递增」= 真平台扫掠），
    # 不参与任何命中判定。
    arc_radius_min: Optional[float] = None
    arc_radius_max: Optional[float] = None
    point_count: int = 0                        # 簇内段数
    # 命中判据：""（未命中，不输出）/ same_cluster_drill（A）/ depth_diameter_signature（B）
    matched_by: str = ""

    @property
    def deeper_z(self) -> Optional[float]:
        """同簇里比平台更深的下钻 / 螺旋 Z（取最负）。"""

        values = [z for z in (self.helical_z, self.drill_z) if z is not None]
        return min(values) if values else None

    @property
    def effective_hole_diameter(self) -> Optional[float]:
        """有效孔径 = 刀具直径 + 2×平台圆弧半径（平台走完整圈时铣出的孔径）。"""

        if self.tool_diameter is None or self.arc_radius is None:
            return None
        return self.tool_diameter + 2.0 * self.arc_radius

    @property
    def is_counterbore(self) -> bool:
        """是否命中判据 A：有平底圆弧，且**同簇还存在更深的下钻/螺旋段**。

        物理含义：平台底面（平底圆弧，如 Z=-4）下面还有继续往下钻的通孔（如 Z=-20）。
        普通平底清角的底面就是该处最深点，下面没有更深的钻孔 → 不是平台。

        注意平底圆弧在螺旋下刀的**每一层收尾**处也会出现，那些层的平底 Z 与螺旋 Z 相同，
        不满足「有更深的下钻/螺旋」，会被这里排除。
        """

        if self.platform_z is None:
            return False
        deeper = self.deeper_z
        if deeper is None:
            return False
        return deeper < self.platform_z - DEFAULT_DEPTH_EPSILON_MM

    def to_dict(self) -> Dict[str, object]:
        effective = self.effective_hole_diameter
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "platform_z": None if self.platform_z is None else round(self.platform_z, 3),
            "helical_z": None if self.helical_z is None else round(self.helical_z, 3),
            "drill_z": None if self.drill_z is None else round(self.drill_z, 3),
            "tool_diameter": self.tool_diameter,
            "arc_radius": None if self.arc_radius is None else round(self.arc_radius, 3),
            "arc_radius_min": (
                None if self.arc_radius_min is None else round(self.arc_radius_min, 3)
            ),
            "arc_radius_max": (
                None if self.arc_radius_max is None else round(self.arc_radius_max, 3)
            ),
            "effective_hole_diameter": None if effective is None else round(effective, 3),
            "matched_by": self.matched_by,
            "point_count": self.point_count,
        }


@dataclass
class CounterboreResult:
    """一份 NC 的沉头孔平台识别结果。"""

    path: str
    tool_diameter: Optional[float] = None
    clusters: List[BoreCluster] = field(default_factory=list)
    # 多个平台 Z 不一致（相差超过 depth_tolerance_mm）
    inconsistent: bool = False
    # 参与统计的平台 Z 集合（去重后）
    platform_depths: List[float] = field(default_factory=list)
    error: str = ""

    @property
    def has_counterbore(self) -> bool:
        return bool(self.clusters)

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "tool_diameter": self.tool_diameter,
            "inconsistent": bool(self.inconsistent),
            "platform_depths": [round(z, 3) for z in self.platform_depths],
            "clusters": [c.to_dict() for c in self.clusters],
            "error": self.error,
        }


def _cluster_key(seg: NCSegment) -> Tuple[float, float]:
    """聚类键：优先圆弧圆心，缺圆心时用起点。"""

    if seg.center_x is not None and seg.center_y is not None:
        return seg.center_x, seg.center_y
    return seg.x, seg.y


def _aggregate(segments: Sequence[NCSegment]) -> List[Dict[str, object]]:
    """把段按「(段类型, 圆心/起点, 半径)」聚合，只保留每处的最深 Z 与段数。

    螺旋孔在同一圆心会堆几千段；先聚合再聚类，把 O(n²) 压成 O(k²)（k = 唯一位置数，通常 < 200）。
    聚合键按 0.05mm 量化，既不会把同一圆心拆开，也不会误合并相邻孔。
    """

    buckets: Dict[Tuple[str, float, float, float], Dict[str, object]] = {}
    for seg in segments:
        kx, ky = _cluster_key(seg)
        quant = 0.05
        key = (
            seg.kind,
            round(kx / quant),
            round(ky / quant),
            round((seg.radius or 0.0) / quant),
        )
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "kind": seg.kind,
                "x": kx,
                "y": ky,
                "radius": seg.radius,
                "z_min": seg.z,
                "count": 1,
            }
            buckets[key] = bucket
        else:
            bucket["count"] = int(bucket["count"]) + 1
            if seg.z is not None:
                current = bucket["z_min"]
                if current is None or seg.z < current:
                    bucket["z_min"] = seg.z
    return list(buckets.values())


def _cluster_by_xy(
    points: Sequence[Dict[str, object]], tolerance: float
) -> List[List[Dict[str, object]]]:
    """按 XY 距离 ≤ tolerance 做并查集聚类（空间网格哈希）。

    输入是 :func:`_aggregate` 的输出（唯一位置），数量小，网格里同格点很少。
    """

    if not points:
        return []

    cell = max(tolerance, 1e-6)
    grid: Dict[Tuple[int, int], List[int]] = {}
    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for idx, point in enumerate(points):
        cx = int(math.floor(float(point["x"]) / cell))
        cy = int(math.floor(float(point["y"]) / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((cx + dx, cy + dy), ()):
                    op = points[other]
                    if math.hypot(
                        float(point["x"]) - float(op["x"]),
                        float(point["y"]) - float(op["y"]),
                    ) <= tolerance:
                        union(idx, other)
        grid.setdefault((cx, cy), []).append(idx)

    groups: Dict[int, List[Dict[str, object]]] = {}
    for idx, point in enumerate(points):
        groups.setdefault(find(idx), []).append(point)
    return list(groups.values())


def _cluster_stats(
    members: Sequence[Dict[str, object]],
    tool_diameter: Optional[float],
    *,
    required_depth: Optional[float] = None,
    required_diameter: Optional[float] = None,
    depth_match_tolerance_mm: float = DEFAULT_DEPTH_MATCH_TOLERANCE_MM,
    diameter_match_tolerance_mm: float = DEFAULT_DIAMETER_MATCH_TOLERANCE_MM,
) -> BoreCluster:
    """汇总一个簇的平台/螺旋/下钻深度，簇坐标取圆心均值。

    同时做判据 A / B 命中判定并写进 ``matched_by``：

    - 判据 A：:attr:`BoreCluster.is_counterbore`（同簇平底 + 更深螺旋/下钻）；
    - 判据 B（需 ``required_depth`` / ``required_diameter`` 都给出且刀具直径已知）：
      扫簇内全部平底圆弧桶，取「|Z|≈要求深度 且 有效孔径≈要求直径」的最深命中桶
      作为平台（同深取半径更大者）。判据 A 命中时 A 优先，B 信息不覆盖。
    """

    cluster = BoreCluster(
        x=sum(float(m["x"]) for m in members) / len(members) if members else 0.0,
        y=sum(float(m["y"]) for m in members) / len(members) if members else 0.0,
        tool_diameter=tool_diameter,
        point_count=sum(int(m["count"]) for m in members),
    )

    flat = [m for m in members if m["kind"] == SEG_ARC_FLAT and m["z_min"] is not None]
    helical = [m for m in members if m["kind"] == SEG_ARC_HELICAL and m["z_min"] is not None]
    drill = [m for m in members if m["kind"] == SEG_DRILL and m["z_min"] is not None]

    if flat:
        deepest_z = min(float(m["z_min"]) for m in flat)
        # 同深平底圆弧取半径最大者：平台底面环形区是同深处半径最大的圆弧，
        # 小半径同深圆弧多为清角；固定取最大保证 arc_radius 判定确定性。
        deepest = max(
            (m for m in flat if float(m["z_min"]) == deepest_z),
            key=lambda m: float(m["radius"] or 0.0),
        )
        cluster.platform_z = deepest_z
        cluster.arc_radius = deepest["radius"]
    if helical:
        cluster.helical_z = min(float(m["z_min"]) for m in helical)
    if drill:
        cluster.drill_z = min(float(m["z_min"]) for m in drill)

    if cluster.is_counterbore:
        # 判据 A 命中（更强的同簇证据）；即使判据 B 也满足也保持 A 标记
        cluster.matched_by = MATCHED_BY_SAME_CLUSTER
        return cluster

    # 判据 B：深度 + 直径联合签名（要求值或刀具直径缺失时退回纯几何模式）
    if required_depth is None or required_diameter is None or tool_diameter is None:
        return cluster

    match: Optional[Dict[str, object]] = None
    match_z = 0.0
    match_radius = 0.0
    for bucket in flat:
        radius = bucket["radius"]
        if radius is None:
            continue
        z = float(bucket["z_min"])
        effective = float(tool_diameter) + 2.0 * float(radius)
        if (
            abs(abs(z) - float(required_depth)) > float(depth_match_tolerance_mm) + 1e-9
            or abs(effective - float(required_diameter)) > float(diameter_match_tolerance_mm) + 1e-9
        ):
            continue
        # 取最深（z 最小）的命中桶；同深取半径更大者
        if (
            match is None
            or z < match_z
            or (z == match_z and float(radius) > match_radius)
        ):
            match = bucket
            match_z = z
            match_radius = float(radius)
    if match is not None:
        cluster.matched_by = MATCHED_BY_SIGNATURE
        # 平台 Z 与半径取命中桶：该桶才是签名对应的平台底面，
        # 簇内其它深度的平底桶属于别的特征，不参与量测
        cluster.platform_z = match_z
        cluster.arc_radius = match["radius"]
    return cluster


def _arc_xy(seg: NCSegment) -> Tuple[float, float]:
    """圆弧段的参考坐标：优先圆心（同轴心圆弧起点会漂移），缺圆心时用终点。"""

    if seg.center_x is not None and seg.center_y is not None:
        return seg.center_x, seg.center_y
    return float(seg.x or 0.0), float(seg.y or 0.0)


def _sweep_clusters(
    segments: Sequence[NCSegment],
    tool_diameter: Optional[float],
    *,
    required_diameter: Optional[float],
    xy_tolerance_mm: float,
    diameter_match_tolerance_mm: float,
    min_arcs: int,
    min_full_circles: int,
    min_step_mm: float = DEFAULT_SWEEP_MIN_STEP_MM,
) -> List[BoreCluster]:
    """判据 C：找「定深平底弧扫掠」平台底面（第三种形态）。

    只在**同一模态 Z 层**内聚类，因为不同 Z 的平底弧属于不同加工层，
    混在一起会把整个孔的所有层并成一个簇、平台 Z 变成最深层（= 判据 A 的盲区）。

    每簇必须同时满足（缺一不可，均为几何硬证据）：

    1. 平底弧段数 ≥ ``min_arcs``；
    2. 至少 ``min_full_circles`` 条**整圆**（本段终点 == 上一段终点）——「整圆收尾」；
    3. 簇首段的前一段是**同 Z 螺旋弧**——「螺旋进刀」；
    4. 参考圆心 1mm 内存在**更深**（Z 更负）的圆弧/下钻段——平台下面还有通孔；
    5. **上大下小**：包络半径比上述更深特征的最大半径至少大 ``min_step_mm``。
       这是平台的定义性几何（平台直径 > 下方通孔直径）；螺旋钻削的**中间层平底收尾**
       与螺旋同半径继续下钻，台阶恒为 0 → 被本条排除（纯螺旋孔 100% 被拦住）；
    6. 有效孔径（刀具直径 + 2×包络半径）与要求直径相差 ≤ ``diameter_match_tolerance_mm``。
       要求直径缺失时（纯几何模式）本条不启用。

    返回的 ``BoreCluster`` 已写好 ``matched_by = MATCHED_BY_ARC_FLAT_SWEEP``，
    ``platform_z`` = 该层 Z，``arc_radius`` = 包络半径，``point_count`` = 簇内平底弧段数。
    """

    if not segments:
        return []

    # 整圆判定：本段终点 == 上一段终点（只给 I/J 的整圆行会沿用上一 XY，因此成立）
    full_circle = [False] * len(segments)
    for index in range(1, len(segments)):
        current, previous = segments[index], segments[index - 1]
        if (
            current.x is not None
            and previous.x is not None
            and current.y is not None
            and previous.y is not None
            and abs(current.x - previous.x) < 1e-6
            and abs(current.y - previous.y) < 1e-6
        ):
            full_circle[index] = True

    arcs = [
        (index, seg)
        for index, seg in enumerate(segments)
        if seg.kind in (SEG_ARC_FLAT, SEG_ARC_HELICAL, SEG_DRILL)
        and seg.x is not None
        and seg.y is not None
        and seg.z is not None
    ]
    if not arcs:
        return []

    # 按模态 Z 分层（0.001mm 量化；同层 Z 完全相同）
    by_z: Dict[float, List[Tuple[int, NCSegment]]] = {}
    for index, seg in arcs:
        if seg.kind != SEG_ARC_FLAT:
            continue
        by_z.setdefault(round(float(seg.z), 3), []).append((index, seg))

    tolerance = float(xy_tolerance_mm) if xy_tolerance_mm else DEFAULT_XY_TOLERANCE_MM
    found: List[BoreCluster] = []
    for level_z in sorted(by_z):
        items = by_z[level_z]
        if len(items) < int(min_arcs):
            continue
        points = []
        for index, seg in items:
            cx, cy = _arc_xy(seg)
            points.append(
                {
                    "kind": SEG_ARC_FLAT,
                    "x": cx,
                    "y": cy,
                    "radius": seg.radius,
                    "z_min": float(seg.z),
                    "count": 1,
                }
            )
        for members in _cluster_by_xy(points, tolerance):
            if len(members) < int(min_arcs):
                continue
            # 该簇对应的原始段（按圆心 1mm 反查；同 Z 层内成员数少，直接过滤）
            member_indices = [
                index
                for index, seg in items
                if any(
                    math.hypot(_arc_xy(seg)[0] - float(m["x"]), _arc_xy(seg)[1] - float(m["y"]))
                    <= tolerance
                    for m in members
                )
            ]
            if len(member_indices) < int(min_arcs):
                continue
            n_full = sum(1 for index in member_indices if full_circle[index])
            if n_full < int(min_full_circles):
                continue
            # 螺旋进刀：簇首段的前一段是同 Z 螺旋弧
            first_index = min(member_indices)
            previous = segments[first_index - 1] if first_index > 0 else None
            if not (
                previous is not None
                and previous.kind == SEG_ARC_HELICAL
                and previous.z is not None
                and abs(float(previous.z) - level_z) < 0.01
            ):
                continue
            # 包络半径：ref = 半径最大那条弧的圆心
            member_segs = [segments[index] for index in member_indices]
            radii = [seg.radius for seg in member_segs if seg.radius is not None]
            if not radii:
                continue
            ref_seg = max(member_segs, key=lambda s: s.radius or 0.0)
            ref_x, ref_y = _arc_xy(ref_seg)
            env_radius = max(
                math.hypot(_arc_xy(seg)[0] - ref_x, _arc_xy(seg)[1] - ref_y) + (seg.radius or 0.0)
                for seg in member_segs
            )
            # 同 XY 更深特征（平台下面的通孔）
            deeper = [
                other
                for other_index, other in arcs
                if other.z is not None
                and float(other.z) < level_z - DEFAULT_DEPTH_EPSILON_MM
                and math.hypot(_arc_xy(other)[0] - ref_x, _arc_xy(other)[1] - ref_y) <= tolerance
            ]
            if not deeper:
                continue
            # 上大下小：平台底面半径必须明显大于下方最深特征的最大半径
            # （螺旋钻削中间层的平底收尾与螺旋同半径 → 台阶 0 → 排除）
            deeper_radius = max((float(other.radius or 0.0) for other in deeper), default=0.0)
            if env_radius <= deeper_radius + float(min_step_mm) - 1e-9:
                continue
            deeper_values = [float(other.z) for other in deeper]
            # 有效孔径签名（要求直径缺失时不启用，保留纯几何命中）
            effective = None
            if tool_diameter is not None:
                effective = float(tool_diameter) + 2.0 * env_radius
            if required_diameter is not None:
                if effective is None:
                    continue
                if abs(effective - float(required_diameter)) > float(diameter_match_tolerance_mm) + 1e-9:
                    continue
            found.append(
                BoreCluster(
                    x=ref_x,
                    y=ref_y,
                    platform_z=level_z,
                    helical_z=min(deeper_values),
                    tool_diameter=tool_diameter,
                    arc_radius=env_radius,
                    arc_radius_min=min(float(radius) for radius in radii),
                    arc_radius_max=max(float(radius) for radius in radii),
                    point_count=len(member_indices),
                    matched_by=MATCHED_BY_ARC_FLAT_SWEEP,
                )
            )
    return found


def find_counterbores(
    segments: Sequence[NCSegment],
    *,
    tool_diameter: Optional[float] = None,
    xy_tolerance_mm: float = DEFAULT_XY_TOLERANCE_MM,
    depth_tolerance_mm: float = DEFAULT_CLUSTER_DEPTH_MM,
    required_depth: Optional[float] = None,
    required_diameter: Optional[float] = None,
    depth_match_tolerance_mm: float = DEFAULT_DEPTH_MATCH_TOLERANCE_MM,
    diameter_match_tolerance_mm: float = DEFAULT_DIAMETER_MATCH_TOLERANCE_MM,
    sweep_enabled: bool = True,
    sweep_min_arcs: int = DEFAULT_SWEEP_MIN_ARCS,
    sweep_min_full_circles: int = DEFAULT_SWEEP_MIN_FULL_CIRCLES,
    sweep_min_step_mm: float = DEFAULT_SWEEP_MIN_STEP_MM,
) -> Tuple[List[BoreCluster], bool, List[float]]:
    """从段列表里找沉头孔底面平台。

    返回 ``(平台簇列表, inconsistent, 去重后的平台Z列表)``。

    - 判据 A「平底圆弧比同圆心/同位置的螺旋/下钻更深」与判据 B「|Z|≈要求深度 且
      有效孔径≈要求直径」（传入要求值时启用）先跑；
    - **两者都落空时**才跑判据 C「定深平底弧扫掠模板」（``sweep_enabled`` 控制，
      见 :func:`_sweep_clusters`）。C 是兜底判据，因此不会改变任何 A/B 已给出的结论；
    - 要求值缺失时判据 B 退回纯几何模式；判据 C 只在**给出要求直径**时启用
      （没有要求值就不产生结论）；
    - 多个平台 Z 相差超过 ``depth_tolerance_mm`` → ``inconsistent=True``。
    """

    tolerance = float(xy_tolerance_mm) if xy_tolerance_mm else DEFAULT_XY_TOLERANCE_MM
    # 只把「圆弧 / 下钻」段拿来聚类：直线段（刀路连接）会污染圆心键
    arc_segments = [
        s for s in segments
        if s.kind in (SEG_ARC_FLAT, SEG_ARC_HELICAL, SEG_DRILL)
        and s.x is not None and s.y is not None
    ]
    points = _aggregate(arc_segments)
    clusters = [
        _cluster_stats(
            members,
            tool_diameter,
            required_depth=required_depth,
            required_diameter=required_diameter,
            depth_match_tolerance_mm=depth_match_tolerance_mm,
            diameter_match_tolerance_mm=diameter_match_tolerance_mm,
        )
        for members in _cluster_by_xy(points, tolerance)
    ]
    counterbores = [c for c in clusters if c.matched_by]

    # 判据 C 的启用条件（两条，都是误报/噪声控制）：
    # ① **有要求直径**：判据 C 靠「包络孔径 ≈ 要求直径」定位平台；没有要求值时启用
    #    只会让无要求的工件多出一批候选记录；
    # ② 判据 A/B **在本份 NC 内全部落空**（兜底语义）：
    #    保证所有既有判定逐条不变；同一份 NC 若既有 A/B 平台、又有 C 形态平台，
    #    只保留 A/B（更强证据），深度判定取最深平台，漏掉较浅的第二处不影响结论。
    if sweep_enabled and required_diameter is not None and not counterbores:
        counterbores = _sweep_clusters(
            segments,
            tool_diameter,
            required_diameter=required_diameter,
            xy_tolerance_mm=tolerance,
            diameter_match_tolerance_mm=diameter_match_tolerance_mm,
            min_arcs=sweep_min_arcs,
            min_full_circles=sweep_min_full_circles,
            min_step_mm=sweep_min_step_mm,
        )

    depths = sorted({round(c.platform_z, 3) for c in counterbores if c.platform_z is not None})
    inconsistent = False
    if len(depths) > 1:
        spread = max(depths) - min(depths)
        inconsistent = spread > float(depth_tolerance_mm)

    # 按深度升序（最浅的在前），便于人工核对
    counterbores.sort(key=lambda c: (c.platform_z if c.platform_z is not None else 0.0, c.x, c.y))
    return counterbores, inconsistent, depths


def analyze_nc_counterbores(
    nc_result,
    *,
    xy_tolerance_mm: float = DEFAULT_XY_TOLERANCE_MM,
    depth_tolerance_mm: float = DEFAULT_CLUSTER_DEPTH_MM,
    required_depth: Optional[float] = None,
    required_diameter: Optional[float] = None,
    depth_match_tolerance_mm: float = DEFAULT_DEPTH_MATCH_TOLERANCE_MM,
    diameter_match_tolerance_mm: float = DEFAULT_DIAMETER_MATCH_TOLERANCE_MM,
    sweep_enabled: bool = True,
    sweep_min_arcs: int = DEFAULT_SWEEP_MIN_ARCS,
    sweep_min_full_circles: int = DEFAULT_SWEEP_MIN_FULL_CIRCLES,
    sweep_min_step_mm: float = DEFAULT_SWEEP_MIN_STEP_MM,
) -> CounterboreResult:
    """对一份 :class:`ncparse.nc_parser.NCParseResult` 做沉头孔平台识别。

    ``required_depth`` / ``required_diameter`` 是要求值（可缺，如订单/规格单给定）：
    都给出时启用判据 B（深度+直径联合签名），缺失时退回纯几何判据 A。
    ``sweep_enabled`` / ``sweep_min_arcs`` / ``sweep_min_full_circles`` 控制判据 C
    （定深平底弧扫掠）；默认启用，关闭后行为与新增判据 C 之前逐条一致。
    """

    out = CounterboreResult(
        path=getattr(nc_result, "path", ""),
        tool_diameter=getattr(nc_result, "tool_diameter", None),
    )
    errors = list(getattr(nc_result, "errors", []) or [])
    if errors:
        out.error = "；".join(errors)
        return out

    clusters, inconsistent, depths = find_counterbores(
        getattr(nc_result, "segments", []) or [],
        tool_diameter=out.tool_diameter,
        xy_tolerance_mm=xy_tolerance_mm,
        depth_tolerance_mm=depth_tolerance_mm,
        required_depth=required_depth,
        required_diameter=required_diameter,
        depth_match_tolerance_mm=depth_match_tolerance_mm,
        diameter_match_tolerance_mm=diameter_match_tolerance_mm,
        sweep_enabled=sweep_enabled,
        sweep_min_arcs=sweep_min_arcs,
        sweep_min_full_circles=sweep_min_full_circles,
        sweep_min_step_mm=sweep_min_step_mm,
    )
    out.clusters = clusters
    out.inconsistent = inconsistent
    out.platform_depths = depths
    return out
