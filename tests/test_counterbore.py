"""沉头孔底面平台识别测试（``ncparse.counterbore``）。

覆盖：

1. 平底圆弧 + 更深螺旋/下钻 → 识别为平台（判据 A）；
2. **螺旋每层收尾的平底圆弧不被误判**（平底 Z == 螺旋 Z，不满足「有更深的下钻/螺旋」）；
3. 普通平底清角（没有更深的下钻/螺旋）→ 不是平台；
4. XY 聚类：同圆心整圈圆弧聚成一簇，不同孔分开；
5. 同一 NC 多簇 Z 不一致 → ``inconsistent=True``；
6. 判据 B（深度 + 直径联合签名）：跨文件平台（平台铣削与通孔钻削是两份 NC）——
   平底圆弧 |Z|≈要求深度 且 刀径+2×半径≈要求直径 → 命中；要求值缺失退回纯几何；
7. 判据 C（定深平底弧扫掠）：形态 + 包络孔径识别第三种平台；
8. 端到端：脱敏合成 NC 文本经 ``parse_nc()`` 解析后走完整识别链路；
   设置 ``NCPARSE_SAMPLE_DIR`` 后可对本地真实样本做冒烟回归（样本不随仓库发布）。
"""

from __future__ import annotations

import os

import pytest

from ncparse.counterbore import (
    MATCHED_BY_ARC_FLAT_SWEEP,
    MATCHED_BY_SAME_CLUSTER,
    MATCHED_BY_SIGNATURE,
    analyze_nc_counterbores,
    find_counterbores,
)
from ncparse.nc_parser import (
    SEG_ARC_FLAT,
    SEG_ARC_HELICAL,
    SEG_DRILL,
    SEG_LINEAR,
    NCSegment,
    NCParseResult,
    parse_nc,
)

# 真实样本目录通过环境变量提供（样本不随仓库发布；未设置时相关用例自动跳过）
SAMPLE_DIR = os.environ.get("NCPARSE_SAMPLE_DIR", "")


def _seg(kind, x, y, z, radius=5.0, center=None, has_z=False, line_no=0):
    cx, cy = center if center else (x, y)
    return NCSegment(
        kind=kind,
        x=x,
        y=y,
        z=z,
        i=radius,
        j=0.0,
        radius=radius,
        center_x=cx,
        center_y=cy,
        has_z=has_z,
        line_no=line_no,
    )


def _sample_dir_files(limit=None):
    if not SAMPLE_DIR or not os.path.isdir(SAMPLE_DIR):
        return []
    files = sorted(
        os.path.join(SAMPLE_DIR, name)
        for name in os.listdir(SAMPLE_DIR)
        if name.casefold().endswith(".nc")
    )
    return files[:limit] if limit else files


# --------------------------------------------------------------------------- #
# 判定逻辑（合成数据）
# --------------------------------------------------------------------------- #


def test_flat_arc_with_deeper_helical_is_counterbore():
    """平台底面 Z=-4，下面还有更深的螺旋通孔 Z=-20 → 平台命中。"""

    segs = [
        _seg(SEG_ARC_FLAT, 100.0, 0.0, -4.0, radius=12.5, center=(100.0, 0.0)),
        _seg(SEG_ARC_HELICAL, 100.0, 0.0, -20.0, radius=6.0, center=(100.0, 0.0)),
    ]
    clusters, inconsistent, depths = find_counterbores(segs, tool_diameter=10.0)
    assert len(clusters) == 1
    assert clusters[0].platform_z == pytest.approx(-4.0)
    assert clusters[0].is_counterbore is True
    assert depths == [-4.0]
    assert inconsistent is False


def test_helical_layer_finish_flat_is_not_counterbore():
    """螺旋下刀的每层收尾也会有不带 Z 的平底圆弧，但平底 Z == 螺旋 Z → 不是平台。"""

    segs = []
    for depth in (-2.0, -4.0, -6.0, -8.0):
        segs.append(_seg(SEG_ARC_FLAT, 0.0, 0.0, depth, radius=5.0))
        segs.append(_seg(SEG_ARC_HELICAL, 0.0, 0.0, depth, radius=5.0, has_z=True))
    clusters, inconsistent, depths = find_counterbores(segs, tool_diameter=10.0)
    assert clusters == []
    assert inconsistent is False
    assert depths == []


def test_plain_flat_clearing_is_not_counterbore():
    """普通平底清角：底面就是该处最深点，下面没有更深的下钻/螺旋 → 不是平台。"""

    segs = [
        _seg(SEG_ARC_FLAT, 5.0, 5.0, -3.0, radius=2.0),
        _seg(SEG_LINEAR, 5.0, 5.0, -3.0),
    ]
    clusters, inconsistent, depths = find_counterbores(segs, tool_diameter=10.0)
    assert clusters == []


def test_drill_below_flat_makes_counterbore():
    """平底 Z=-4 + 固定循环下钻 Z=-25 → 平台命中。"""

    segs = [
        _seg(SEG_ARC_FLAT, 0.0, 0.0, -4.0, radius=8.0),
        _seg(SEG_DRILL, 0.0, 0.0, -25.0),
    ]
    clusters, inconsistent, depths = find_counterbores(segs, tool_diameter=16.0)
    assert len(clusters) == 1
    assert clusters[0].drill_z == pytest.approx(-25.0)
    assert clusters[0].is_counterbore is True


def test_xy_clustering_merges_same_center_and_separates_holes():
    """同圆心整圈圆弧（起点沿圆周漂移）聚成一簇；相距 > 容差的孔分开。"""

    segs = []
    # 孔 A：圆心 (0,0)，起点在圆周不同角度（整圆被拆成多段）
    for start in ((5.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (0.0, -5.0)):
        segs.append(_seg(SEG_ARC_FLAT, start[0], start[1], -4.0, radius=5.0, center=(0.0, 0.0)))
    segs.append(_seg(SEG_ARC_HELICAL, 0.0, 0.0, -20.0, radius=2.0, center=(0.0, 0.0)))
    # 孔 B：圆心 (50,0)
    segs.append(_seg(SEG_ARC_FLAT, 55.0, 0.0, -6.0, radius=5.0, center=(50.0, 0.0)))
    segs.append(_seg(SEG_ARC_HELICAL, 50.0, 0.0, -20.0, radius=2.0, center=(50.0, 0.0)))

    clusters, inconsistent, depths = find_counterbores(segs, tool_diameter=10.0)
    assert len(clusters) == 2
    xs = sorted(round(c.x, 1) for c in clusters)
    assert xs == [0.0, 50.0]
    # 两簇 Z 相差 2mm > 0.05 → 不一致
    assert inconsistent is True
    assert depths == [-6.0, -4.0]


def test_consistent_depths_are_not_inconsistent():
    segs = []
    for x in (0.0, 50.0, 100.0):
        segs.append(_seg(SEG_ARC_FLAT, x, 0.0, -4.0, radius=8.0, center=(x, 0.0)))
        segs.append(_seg(SEG_ARC_HELICAL, x, 0.0, -20.0, radius=3.0, center=(x, 0.0)))
    clusters, inconsistent, depths = find_counterbores(segs, tool_diameter=10.0)
    assert len(clusters) == 3
    assert inconsistent is False
    assert depths == [-4.0]


def test_analyze_nc_counterbores_reports_parse_errors():
    result = NCParseResult(path="bad.nc", errors=["未找到刀具头"])
    out = analyze_nc_counterbores(result)
    assert out.error
    assert out.clusters == []
    assert out.has_counterbore is False


def test_tolerance_configurable():
    """XY 容差放大后，相距 1.5mm 的两个圆点会被合并成一簇。"""

    segs = [
        _seg(SEG_ARC_FLAT, 0.0, 0.0, -4.0, radius=5.0, center=(0.0, 0.0)),
        _seg(SEG_ARC_HELICAL, 1.5, 0.0, -20.0, radius=2.0, center=(1.5, 0.0)),
    ]
    tight, _, _ = find_counterbores(segs, xy_tolerance_mm=1.0)
    loose, _, _ = find_counterbores(segs, xy_tolerance_mm=2.0)
    # 1.0mm 容差下两个圆心分开 → 平底簇没有更深的螺旋（不是平台），螺旋簇没有平底
    assert tight == []
    # 2.0mm 容差下合并成一簇 → 平底 + 更深螺旋 → 平台
    assert len(loose) == 1


# --------------------------------------------------------------------------- #
# 判据 B：深度 + 直径联合签名（跨文件平台）
# --------------------------------------------------------------------------- #


def _signature_segments():
    """四处对称位置的平台平底（r=7.5，Z=-4.0），无螺旋/下钻（跨文件场景）。"""

    segs = []
    for x, y in ((100.0, 50.0), (-100.0, 50.0), (100.0, -50.0), (-100.0, -50.0)):
        segs.append(
            _seg(SEG_ARC_FLAT, x + 7.5, y, -4.0, radius=7.5, center=(x, y))
        )
    return segs


def test_signature_matches_cross_file_counterbore():
    """平台与通孔是两份 NC：单文件只有平底圆弧，按「深度+直径」签名命中。"""

    clusters, inconsistent, depths = find_counterbores(
        _signature_segments(),
        tool_diameter=6.0,
        required_depth=4.0,
        required_diameter=20.0,
    )
    assert len(clusters) == 4
    assert all(c.matched_by == MATCHED_BY_SIGNATURE for c in clusters)
    assert all(c.platform_z == pytest.approx(-4.0) for c in clusters)
    # 有效孔径 = 6 + 2×7.5 = 21.0（与要求 20 差 1.0 ≤ 1.5）
    assert all(c.effective_hole_diameter == pytest.approx(21.0) for c in clusters)
    assert depths == [-4.0]
    assert inconsistent is False


def test_signature_without_requirement_is_pure_geometry():
    """不传要求值 → 退回判据 A：平底-only 文件没有更深的螺旋/下钻 → 不算平台。"""

    clusters, _, depths = find_counterbores(_signature_segments(), tool_diameter=6.0)
    assert clusters == []
    assert depths == []


def test_signature_depth_mismatch_is_rejected():
    """|Z| 与要求深度差超容差 → 判据 B 不命中。"""

    segs = [_seg(SEG_ARC_FLAT, 0.0, 0.0, -5.5, radius=7.5, center=(0.0, 0.0))]
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=6.0, required_depth=4.0, required_diameter=20.0
    )
    assert clusters == []


def test_signature_diameter_mismatch_is_rejected():
    """有效孔径与要求直径差超容差 → 判据 B 不命中（防通孔平台/清角混入）。"""

    segs = [_seg(SEG_ARC_FLAT, 0.0, 0.0, -4.0, radius=3.725, center=(0.0, 0.0))]
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=17.0, required_depth=4.0, required_diameter=30.0
    )
    assert clusters == []


def test_signature_partial_requirement_falls_back_to_geometry():
    """只传深度不传直径（或反之）→ 无法做签名，退回纯几何。"""

    segs = _signature_segments()
    assert find_counterbores(
        segs, tool_diameter=6.0, required_depth=4.0, required_diameter=None
    )[0] == []
    assert find_counterbores(
        segs, tool_diameter=6.0, required_depth=None, required_diameter=20.0
    )[0] == []
    assert find_counterbores(
        segs, tool_diameter=None, required_depth=4.0, required_diameter=20.0
    )[0] == []


def test_criterion_a_wins_when_both_match():
    """同文件场景（平底 + 更深螺旋）：A、B 同时命中时保持 A 标记，不改平台。"""

    segs = [
        _seg(SEG_ARC_FLAT, 0.0, 0.0, -4.0, radius=9.5, center=(0.0, 0.0)),
        _seg(SEG_ARC_HELICAL, 0.0, 0.0, -20.0, radius=6.0, center=(0.0, 0.0)),
    ]
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=6.0, required_depth=4.0, required_diameter=25.0
    )
    assert len(clusters) == 1
    assert clusters[0].matched_by == MATCHED_BY_SAME_CLUSTER
    assert clusters[0].platform_z == pytest.approx(-4.0)
    assert clusters[0].effective_hole_diameter == pytest.approx(25.0)


def test_cluster_dict_exposes_effective_diameter_and_matched_by():
    import json

    clusters, _, _ = find_counterbores(
        _signature_segments(),
        tool_diameter=6.0,
        required_depth=4.0,
        required_diameter=20.0,
    )
    data = clusters[0].to_dict()
    assert data["effective_hole_diameter"] == pytest.approx(21.0)
    assert data["matched_by"] == MATCHED_BY_SIGNATURE
    json.dumps(data, ensure_ascii=False)  # 必须可序列化


def test_signature_picks_deepest_matching_bucket_not_smaller_arcs():
    """同簇里混着清角小圆弧与平台大圆弧（同 Z）：平台与有效孔径取签名命中的大圆弧。"""

    segs = [
        # 清角小圆弧（有效孔径 6.4/12，不满足要求直径 20）
        _seg(SEG_ARC_FLAT, 0.2, 0.0, -4.0, radius=0.2, center=(0.0, 0.0)),
        _seg(SEG_ARC_FLAT, 3.0, 0.0, -4.0, radius=3.0, center=(0.0, 0.0)),
        # 平台底面大圆弧
        _seg(SEG_ARC_FLAT, 7.5, 0.0, -4.0, radius=7.5, center=(0.0, 0.0)),
    ]
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=6.0, required_depth=4.0, required_diameter=20.0
    )
    assert len(clusters) == 1
    assert clusters[0].arc_radius == pytest.approx(7.5)
    assert clusters[0].effective_hole_diameter == pytest.approx(21.0)


def test_analyze_nc_counterbores_passes_requirement_through():
    """analyze_nc_counterbores 把要求值透传给识别层（判据 B 生效）。"""

    result = NCParseResult(
        path="synthetic.nc", tool_diameter=6.0, segments=_signature_segments()
    )
    out = analyze_nc_counterbores(
        result, required_depth=4.0, required_diameter=20.0
    )
    assert out.has_counterbore
    assert all(c.matched_by == MATCHED_BY_SIGNATURE for c in out.clusters)


# --------------------------------------------------------------------------- #
# 端到端：脱敏合成 NC 文本 → parse_nc() → 识别
# （替代不随仓库发布的真实样本用例；等效覆盖单文件判据 A 与跨文件判据 B）
# --------------------------------------------------------------------------- #

SYNTHETIC_ONE_FILE_NC = """%
O0100
(demo_part.nc)
G91 G28 Z0.0
G21 G40 G49 G80
G00 G90 G54 X0.0 Y0.0
(Tool=FLAT D=16.00 R=0.50)
S1200 M03
G90 G0 X5.0 Y0.0
Z100.
Z3.
G1 Z-10. F250.
G2 X-5. Y0. I-5. J0.
G2 X5. Y0. I5. J0.
G1 Z3.
G1 Z-16. F100.
G3 X5. Y0. I-5. J0. Z-18.
M05
M09
G91 G28 Z0.0
M30
%
"""

SYNTHETIC_CROSS_FILE_NC = """%
O0100
(demo_part_flat.nc)
G91 G28 Z0.0
G21 G40 G49 G80
G00 G90 G54 X0.0 Y0.0
(Tool=FLAT D=6.00 R=0.30)
S2000 M03
G90 G0 X107.5 Y50.0
Z100.
Z3.
G1 Z-4. F200.
G2 X107.5 Y50. I-7.5 J0.
G0 Z3.
G0 X107.5 Y-50.0
G1 Z-4. F200.
G2 X107.5 Y-50. I-7.5 J0.
G0 Z3.
G0 X-107.5 Y50.0
G1 Z-4. F200.
G2 X-107.5 Y50. I7.5 J0.
G0 Z3.
G0 X-107.5 Y-50.0
G1 Z-4. F200.
G2 X-107.5 Y-50. I7.5 J0.
M05
M09
G91 G28 Z0.0
M30
%
"""


def _write_nc(tmp_path, name, text):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="gbk", errors="replace") as handle:
        handle.write(text)
    return path


def test_synthetic_nc_file_criterion_a_end_to_end(tmp_path):
    """同一份 NC：平台平底（Z=-10）+ 同圆心更深螺旋（Z=-18）→ 判据 A 命中。

    中途的直线下刀不参与圆弧聚类（LINEAR 段被排除），不影响量测。
    """

    result = parse_nc(_write_nc(tmp_path, "one_file.nc", SYNTHETIC_ONE_FILE_NC))
    assert result.errors == []
    kinds = {seg.kind for seg in result.segments}
    assert kinds == {SEG_LINEAR, SEG_ARC_FLAT, SEG_ARC_HELICAL}

    cb = analyze_nc_counterbores(result)
    assert cb.has_counterbore
    assert len(cb.clusters) == 1
    cluster = cb.clusters[0]
    assert cluster.matched_by == MATCHED_BY_SAME_CLUSTER
    assert cluster.platform_z == pytest.approx(-10.0)
    assert cluster.helical_z == pytest.approx(-18.0)
    assert cluster.tool_diameter == pytest.approx(16.0)
    assert cluster.effective_hole_diameter == pytest.approx(26.0)
    assert cb.inconsistent is False


def test_synthetic_nc_file_cross_file_signature(tmp_path):
    """平台与通孔分两份 NC：单文件只有平底圆弧，按「深度+直径」签名命中 4 处。"""

    result = parse_nc(_write_nc(tmp_path, "flat_only.nc", SYNTHETIC_CROSS_FILE_NC))
    assert result.errors == []
    cb = analyze_nc_counterbores(result, required_depth=4.0, required_diameter=20.0)
    assert cb.has_counterbore
    assert len(cb.clusters) == 4
    assert all(c.matched_by == MATCHED_BY_SIGNATURE for c in cb.clusters)
    assert all(c.platform_z == pytest.approx(-4.0) for c in cb.clusters)
    # 有效孔径 = 6 + 2×7.5 = 21.0（与要求 20 差 1.0 ≤ 1.5）
    assert all(c.effective_hole_diameter == pytest.approx(21.0) for c in cb.clusters)
    assert cb.platform_depths == [-4.0]
    assert cb.inconsistent is False
    # 不传要求值：退回纯几何判据 A，该文件不再出平台（跨文件场景）
    pure = analyze_nc_counterbores(result)
    assert not pure.has_counterbore


# --------------------------------------------------------------------------- #
# 判据 C：定深平底弧扫掠模板
# --------------------------------------------------------------------------- #


def _sweep_block(cx, cy, z, radii, full_circles=2, entry_radius=0.477):
    """一个「螺旋进刀 → 平底弧扫掠 → 整圆收尾」块（扫掠模板的单层）。

    圆心逐段漂移 0.1mm（远小于 1mm 聚类容差），半径递增（旋风扫掠）。
    """

    segs = [
        _seg(SEG_ARC_HELICAL, cx + 0.045, cy + 0.475, z, radius=entry_radius,
             center=(cx, cy), has_z=True),
    ]
    for index, radius in enumerate(radii):
        drift = 0.1 * (index % 3)
        segs.append(
            _seg(SEG_ARC_FLAT, cx + radius, cy + drift, z, radius=radius,
                 center=(cx + drift, cy), has_z=False)
        )
    for _ in range(full_circles):
        segs.append(
            _seg(SEG_ARC_FLAT, cx + radii[-1], cy, z, radius=radii[-1],
                 center=(cx, cy), has_z=False)
        )
    return segs


def _sweep_segments(
    *,
    platform_z=-4.2,
    center=(180.0, -82.0),
    sweep_radii=(5.626, 6.2, 7.0, 8.0, 9.5),
    deeper_z=-21.0,
    deeper_radii=(3.122, 4.0, 5.1),
    full_circles=2,
):
    """构造「定深平底弧扫掠」双层模板。

    上层 = 平台底面（platform_z，大半径）；下层 = 通孔（deeper_z，小半径）。
    两层都有各自的扫掠块，因此按圆心聚合后「最深平底 = 下层、同簇无更深螺旋」，
    判据 A 落空；|platform_z| 与要求深度差 0.2mm 超判据 B 的 ±0.05 → 判据 B 也落空。
    这正是判据 C 要覆盖的场景。
    """

    cx, cy = center
    segs = _sweep_block(cx, cy, deeper_z, deeper_radii, full_circles)
    segs += _sweep_block(cx, cy, platform_z, sweep_radii, full_circles)
    return segs


def test_arc_flat_sweep_detected_as_criterion_c():
    """判据 C：螺旋进刀 + 平底弧扫掠 + 整圆收尾 + 同 XY 更深 + 上大下小 → 命中。"""

    segs = _sweep_segments()
    clusters, inconsistent, depths = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.matched_by == MATCHED_BY_ARC_FLAT_SWEEP
    assert cluster.platform_z == pytest.approx(-4.2)
    # 包络半径 = 圆心漂移 0.1 + 最大半径 9.5 = 9.6；有效孔径 = 12 + 2×9.6 = 31.2
    assert cluster.arc_radius == pytest.approx(9.6)
    assert cluster.effective_hole_diameter == pytest.approx(31.2)
    # 平底弧段数：平台层 5 条扫掠弧 + 2 条整圆收尾（不含两层各 1 条螺旋进刀）
    assert cluster.point_count == 7
    assert inconsistent is False
    assert depths == [pytest.approx(-4.2)]


def test_arc_flat_sweep_requires_step_over_deeper_feature():
    """上大下小：平台包络半径不大于下方特征半径（螺旋中间层）→ 判据 C 不命中。"""

    # 下方通孔包络 9.6 ≈ 平台包络 9.6 → 台阶 0，不是「上大下小」
    segs = _sweep_segments(deeper_radii=(3.122, 9.5))
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert clusters == []


def test_arc_flat_sweep_requires_full_circle():
    """没有整圆收尾（终点都不等于上一段终点）→ 判据 C 不命中。"""

    segs = _sweep_segments(full_circles=0)
    # 让所有平底弧终点各不相同
    for index, seg in enumerate(segs):
        seg.x = float(index) + 0.5
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert clusters == []


def test_arc_flat_sweep_requires_helical_entry():
    """簇首段前一段不是同 Z 螺旋弧（这里换成直线段）→ 判据 C 不命中。"""

    segs = _sweep_segments()
    # 平台块（第二块）的首段索引 = 下层块段数（1 进刀 + 3 弧 + 2 整圆 = 6）
    platform_entry = 6
    assert segs[platform_entry].kind == SEG_ARC_HELICAL
    assert segs[platform_entry].z == pytest.approx(-4.2)
    segs[platform_entry] = _seg(
        SEG_LINEAR, segs[platform_entry].x, segs[platform_entry].y, -4.2,
        radius=0.0, center=(180.0, -82.0),
    )
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert clusters == []


def test_arc_flat_sweep_requires_deeper_feature():
    """同 XY 没有更深特征（平台下面没有通孔）→ 判据 C 不命中。"""

    segs = _sweep_segments()
    segs = [s for s in segs if s.z is None or s.z >= -4.2]
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert clusters == []


def test_arc_flat_sweep_rejects_diameter_mismatch():
    """包络孔径与要求直径不符 → 判据 C 不命中。"""

    segs = _sweep_segments()
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=20.0
    )
    assert clusters == []


def test_arc_flat_sweep_min_arcs_threshold():
    """扫掠最少平底弧数可配：抬高门槛后短扫掠不再命中。"""

    segs = _sweep_segments(sweep_radii=(5.626, 9.5))
    # 默认门槛 2 → 命中
    hit, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert len(hit) == 1
    # 门槛抬到 8 → 不命中
    miss, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0,
        sweep_min_arcs=8,
    )
    assert miss == []


def test_arc_flat_sweep_disabled_restores_previous_behaviour():
    """关闭判据 C：同一份段列表不再产出平台（与新增判据前逐条一致）。"""

    segs = _sweep_segments()
    clusters, _, depths = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0,
        sweep_enabled=False,
    )
    assert clusters == []
    assert depths == []


def test_arc_flat_sweep_does_not_override_criteria_ab():
    """判据 A/B 已命中时不追加判据 C（兜底语义：既有结论逐条不变）。"""

    # 同文件既有 A 命中的平台（平底 -4.0 + 更深螺旋 -20），又有 C 形态（-6.5）
    segs = [
        _seg(SEG_ARC_FLAT, 0.0, 0.0, -4.0, radius=9.5, center=(0.0, 0.0)),
        _seg(SEG_ARC_HELICAL, 0.0, 0.0, -20.0, radius=6.0, center=(0.0, 0.0)),
    ]
    segs += _sweep_segments(platform_z=-6.5, center=(100.0, 0.0))
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert len(clusters) == 1
    assert clusters[0].matched_by == MATCHED_BY_SAME_CLUSTER
    assert clusters[0].platform_z == pytest.approx(-4.0)


def test_arc_flat_sweep_clusters_by_1mm_not_by_center_identity():
    """圆心逐段漂移（0.1mm 级）仍在同一簇；相邻孔（>1mm）分开。"""

    segs = _sweep_segments(center=(180.0, -82.0))
    segs += _sweep_segments(center=(180.0, -60.0))  # 22mm 外的第二个平台
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert len(clusters) == 2
    ys = sorted(round(c.y, 1) for c in clusters)
    assert ys == [-82.0, -60.0]
    assert all(c.matched_by == MATCHED_BY_ARC_FLAT_SWEEP for c in clusters)


def test_arc_flat_sweep_not_used_without_required_diameter():
    """没有要求直径时不启用判据 C（避免无要求工件多出候选记录）。"""

    segs = _sweep_segments()
    clusters, _, depths = find_counterbores(segs, tool_diameter=12.0)
    assert clusters == []
    assert depths == []


def test_arc_flat_sweep_exposes_raw_radius_range():
    """判据 C 的簇必须暴露**原始**半径范围（复核靠它区分「半径恒定」与「递增扫掠」）。

    只读字段：不参与命中判定，命中集合与加字段前逐条一致。
    """

    # 最大半径 9.5 → 包络孔径 12 + 2×9.6 = 31.2，落在要求 Φ30 的 ±1.5 内（判据 C 才命中）
    segs = _sweep_segments(sweep_radii=(3.127, 5.0, 7.0, 9.5))
    clusters, _, _ = find_counterbores(
        segs, tool_diameter=12.0, required_depth=4.0, required_diameter=30.0
    )
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.arc_radius_min == pytest.approx(3.127)
    assert cluster.arc_radius_max == pytest.approx(9.5)
    # 包络半径（含圆心漂移）不小于原始最大半径；两者语义不同，不可互换
    assert cluster.arc_radius is not None and cluster.arc_radius >= cluster.arc_radius_max
    data = cluster.to_dict()
    assert data["arc_radius_min"] == pytest.approx(3.127)
    assert data["arc_radius_max"] == pytest.approx(9.5)


# --------------------------------------------------------------------------- #
# 真实样本冒烟回归（设置 NCPARSE_SAMPLE_DIR 后启用；样本不随仓库发布）
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _sample_dir_files(), reason="未设置 NCPARSE_SAMPLE_DIR 或目录里没有 .nc"
)
def test_real_samples_smoke():
    """真实样本（如有）：识别全程不抛异常，命中的簇字段完整。"""

    checked = 0
    for path in _sample_dir_files():
        result = parse_nc(path)
        cb = analyze_nc_counterbores(result)
        for cluster in cb.clusters:
            assert cluster.matched_by in (
                MATCHED_BY_SAME_CLUSTER,
                MATCHED_BY_SIGNATURE,
                MATCHED_BY_ARC_FLAT_SWEEP,
            )
            assert cluster.platform_z is not None
            assert cluster.point_count > 0
        checked += 1
        if checked >= 20:
            break
    assert checked > 0
