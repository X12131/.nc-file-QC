"""NC 刀轨解析测试（``ncparse.nc_parser``）。

未设置 ``NCPARSE_SAMPLE_DIR`` 环境变量（或目录里没有 .nc）时跳过真实样本用例，
只用内联的最小 NC 断言解析契约。真实产线样本不随仓库发布。

覆盖：

1. 刀具头解析（``(Tool=名 D=直径 R=圆角)``）；
2. 段类型区分：``ARC_FLAT`` / ``ARC_HELICAL`` / ``DRILL`` / ``LINEAR``；
3. 平底 vs 螺旋的**硬判据**：圆弧行不带 Z = 平底，带 Z = 螺旋；
4. 平底圆弧继承模态 Z（``z`` 是实际执行 Z，``has_z`` 只标记本行是否写了 Z）；
5. 单份 ≤ 50ms；
6. 解析失败不抛异常，错误进 ``errors``。
"""

from __future__ import annotations

import os
import time

import pytest

from ncparse.nc_parser import (
    SEG_ARC_FLAT,
    SEG_ARC_HELICAL,
    SEG_DRILL,
    SEG_LINEAR,
    parse_nc,
)

# 真实样本目录通过环境变量提供（样本不随仓库发布；未设置时相关用例自动跳过）
SAMPLE_DIR = os.environ.get("NCPARSE_SAMPLE_DIR", "")

MINIMAL_NC = """%
O0100
(test.nc)
G91 G28 Z0.0
G21 G40 G49 G80
G00 G90 G54 X0.0 Y0.0
(Tool=TEST D=10.00 R=0.50)
S1000 M03
G90 G0 X0.0 Y0.0
Z100.
Z3.
G1 Z-4. F250.
G2 X10. Y0. I5. J0.
G3 X0. Y10. Z-6. I0. J5.
G98 G81 X50. Y50. Z-20. R2. F250.
M05
M09
G91 G28 Z0.0
M30
%
"""


def _write(tmp_path, name: str, text: str) -> str:
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="gbk", errors="replace") as handle:
        handle.write(text)
    return path


def _real_samples(limit=None):
    if not SAMPLE_DIR or not os.path.isdir(SAMPLE_DIR):
        return []
    files = sorted(
        os.path.join(SAMPLE_DIR, name)
        for name in os.listdir(SAMPLE_DIR)
        if name.casefold().endswith(".nc")
    )
    return files[:limit] if limit else files


# --------------------------------------------------------------------------- #
# 内联最小 NC：解析契约
# --------------------------------------------------------------------------- #


def test_tool_header_parsed(tmp_path):
    result = parse_nc(_write(tmp_path, "t.nc", MINIMAL_NC))
    assert result.errors == []
    assert result.tool_name == "TEST"
    assert result.tool_diameter == 10.0
    assert result.tool_radius == 0.5


def test_segment_kinds_are_separated(tmp_path):
    result = parse_nc(_write(tmp_path, "t.nc", MINIMAL_NC))
    kinds = [seg.kind for seg in result.segments]
    assert SEG_LINEAR in kinds
    assert SEG_ARC_FLAT in kinds
    assert SEG_ARC_HELICAL in kinds
    assert SEG_DRILL in kinds


def test_flat_vs_helical_hard_criterion(tmp_path):
    """圆弧行不带 Z = 平底；带 Z = 螺旋。"""

    result = parse_nc(_write(tmp_path, "t.nc", MINIMAL_NC))
    flat = [s for s in result.segments if s.kind == SEG_ARC_FLAT]
    helical = [s for s in result.segments if s.kind == SEG_ARC_HELICAL]
    assert len(flat) == 1 and len(helical) == 1
    # 平底圆弧本行没写 Z，但有效 Z 继承上一段（Z-4）
    assert flat[0].has_z is False
    assert flat[0].z == pytest.approx(-4.0)
    assert helical[0].has_z is True
    assert helical[0].z == pytest.approx(-6.0)


def test_arc_radius_from_ij(tmp_path):
    result = parse_nc(_write(tmp_path, "t.nc", MINIMAL_NC))
    flat = [s for s in result.segments if s.kind == SEG_ARC_FLAT][0]
    assert flat.radius == pytest.approx(5.0)
    assert flat.center_x == pytest.approx(5.0)
    assert flat.center_y == pytest.approx(0.0)


def test_drill_cycle_z_read_from_parameters(tmp_path):
    result = parse_nc(_write(tmp_path, "t.nc", MINIMAL_NC))
    drills = [s for s in result.segments if s.kind == SEG_DRILL]
    assert len(drills) == 1
    assert drills[0].x == pytest.approx(50.0)
    assert drills[0].y == pytest.approx(50.0)
    assert drills[0].z == pytest.approx(-20.0)


def test_missing_tool_header_is_reported_not_raised(tmp_path):
    text = MINIMAL_NC.replace("(Tool=TEST D=10.00 R=0.50)\n", "")
    result = parse_nc(_write(tmp_path, "t.nc", text))
    assert result.errors
    assert result.tool_diameter is None


def test_unreadable_file_returns_error_not_raises(tmp_path):
    result = parse_nc(str(tmp_path / "missing.nc"))
    assert result.errors
    assert result.segments == []


# --------------------------------------------------------------------------- #
# 真实样本回归（设置 NCPARSE_SAMPLE_DIR 环境变量后启用；样本不随仓库发布）
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _real_samples(), reason="未设置 NCPARSE_SAMPLE_DIR 或目录里没有 .nc"
)
def test_real_samples_tool_header_and_timing():
    """真实样本（如有）：全部可解析、刀具头齐全，单份耗时中位数 ≤ 50ms。"""

    files = _real_samples()
    timings = []
    for path in files:
        result = parse_nc(path)
        assert result.errors == [], (path, result.errors)
        assert result.tool_diameter is not None, path
        timings.append(result.elapsed_ms)
    timings.sort()
    median = timings[len(timings) // 2]
    assert median <= 50.0, median


@pytest.mark.skipif(
    not _real_samples(), reason="未设置 NCPARSE_SAMPLE_DIR 或目录里没有 .nc"
)
def test_real_samples_flat_arcs_carry_modal_z():
    """平底圆弧必须继承模态 Z（否则平台深度量测拿不到值）。"""

    checked = 0
    for path in _real_samples():
        result = parse_nc(path)
        for seg in (s for s in result.segments if s.kind == SEG_ARC_FLAT):
            assert seg.has_z is False
            assert seg.z is not None, (path, seg.line_no)
            checked += 1
        if checked >= 100:
            break
    if checked == 0:
        pytest.skip("样本语料中没有平底圆弧段")


@pytest.mark.skipif(
    not _real_samples(), reason="未设置 NCPARSE_SAMPLE_DIR 或目录里没有 .nc"
)
def test_real_samples_geometry_is_finite():
    """抽 10 份真实样本：最深平底 Z 必须是有限数且在合理范围内（不出现 NaN）。"""

    checked_files = 0
    for path in _real_samples(limit=10):
        result = parse_nc(path)
        assert result.tool_diameter and result.tool_diameter > 0, path
        flat = [s for s in result.segments if s.kind == SEG_ARC_FLAT]
        depths = [s.z for s in flat if s.z is not None]
        if not depths:
            continue
        deepest = min(depths)
        assert isinstance(deepest, float)
        assert -1000.0 < deepest < 100.0
        checked_files += 1
    if checked_files == 0:
        pytest.skip("样本语料中没有带平底圆弧的文件")
