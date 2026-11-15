"""NC 刀轨只读解析（GBK 文本）。

智模后处理产物的 NC 文件结构：

- 首行 ``%``，第二行 ``O0100``；
- **每份 NC 只有一把刀**：一行 ``(Tool=<名> D=<直径> R=<圆角>)``；
- 坐标系恒为 ``G54`` 绝对坐标，``G91`` 仅用于 ``G28 Z0`` 回零；
- 安全高度 ``G0 Z100.``；
- 固定循环 ``G83/G81 X.. Y.. Z.. R.. F..`` 的 Z 在参数里；
- **平底圆弧** = ``G2/G3 X.. Y.. I.. J..``（不带 Z）；
  **螺旋下刀** = ``G2/G3 X.. Y.. Z.. I.. J..``（带 Z）；
- 圆弧 ``I``/``J`` 是**相对起点的圆心偏移**，半径 = ``sqrt(I² + J²)``。

产出 ``NCParseResult``：

- ``tool_name / tool_diameter / tool_radius``：刀具头；
- ``segments``：段列表 ``[NCSegment, ...]``，段类型：
  ``ARC_FLAT``（平底圆弧）、``ARC_HELICAL``（螺旋下钻）、``DRILL``（G81/G82/G83）、``LINEAR``；
- ``errors``：单文件解析异常（不抛，留给调用方写入 parse_errors.json）。

只读，不修改源文件。任何单文件解析失败都不会中断外层循环——返回 ``NCParseResult``
且 ``errors`` 非空即可。

性能：单份 NC 解析通常 ≤ 50ms。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# 段类型常量
SEG_ARC_FLAT = "ARC_FLAT"        # 平底圆弧：不带 Z 变化
SEG_ARC_HELICAL = "ARC_HELICAL"  # 螺旋下钻：带 Z 变化
SEG_DRILL = "DRILL"              # 固定循环：G81/G82/G83
SEG_LINEAR = "LINEAR"            # 直线插补：G0/G1 等

_TOOL_RE = re.compile(
    r"\(Tool=(?P<name>[^)]*?)\s+D=(?P<d>[\d.]+)\s+R=(?P<r>[\d.]+)\s*\)",
    re.IGNORECASE,
)

# 一行坐标段：X.. Y.. [Z..] [I.. J..] [R..] [F..]；只要 X/Y 出现就算段
_NUM_RE = re.compile(r"([A-Za-z])(-?\d+(?:\.\d+)?|-?\.\d+)")

_G2_RE = re.compile(r"^G0?2\b")
_G3_RE = re.compile(r"^G8[123]\b")

# 一行里的所有「字母+数字」令牌；单次扫描同时取坐标与 G 指令（热路径，避免两次 findall）。
_LETTER_NUM_RE = re.compile(r"([A-Za-z])(-?(?:\d+(?:\.\d*)?|\.\d+))")
# 注释与程序段分隔
_PAREN = "("
_SEMI = ";"
# 回零 / 坐标系切换 / 取消固定循环：不产生刀轨段，且清掉固定循环模态
_CANCEL_CODES = frozenset({28, 30, 53, 80})
_MOTION_CODES = frozenset({0, 1, 2, 3, 81, 82, 83})
_END_RE = re.compile(r"\bM0?30\b", re.IGNORECASE)


@dataclass(slots=True)
class NCSegment:
    """一条刀轨段。

    ``z`` 是该段执行时的**实际模态 Z**（本行没写 Z 就沿用上一段的 Z），
    ``has_z`` 表示本行**是否显式写了 Z**（用于区分平底圆弧 / 螺旋圆弧）。
    """

    kind: str  # SEG_*
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    i: Optional[float] = None
    j: Optional[float] = None
    r: Optional[float] = None
    f: Optional[float] = None
    # 段的圆弧半径（仅 ARC_* 段有效；= sqrt(i² + j²)）
    radius: Optional[float] = None
    # 圆弧圆心绝对坐标（起点 + I/J）；仅 ARC_* 段有效。同轴心的整圈圆弧圆心恒定，
    # 是平台聚类的可靠键（起点会沿圆周漂移）。
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    # 是否带 Z 变化（仅 ARC_* 段有意义；LINEAR 段 z 是坐标不是「变化」）
    has_z: bool = False
    line_no: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "i": self.i,
            "j": self.j,
            "r": self.r,
            "f": self.f,
            "radius": self.radius,
            "has_z": self.has_z,
            "line_no": self.line_no,
        }


@dataclass(slots=True)
class NCParseResult:
    """一份 NC 的解析结果。"""

    path: str
    tool_name: str = ""
    tool_diameter: Optional[float] = None
    tool_radius: Optional[float] = None
    segments: List[NCSegment] = field(default_factory=list)
    # 单文件解析异常（不抛）
    errors: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "tool_name": self.tool_name,
            "tool_diameter": self.tool_diameter,
            "tool_radius": self.tool_radius,
            "segments": [s.to_dict() for s in self.segments],
            "errors": list(self.errors),
            "elapsed_ms": self.elapsed_ms,
        }


def _parse_gcode_line(line: str) -> dict:
    """从一行 G-code 里抽取坐标/参数：X/Y/Z/I/J/R/F。"""

    out: dict = {}
    for letter, num in _NUM_RE.findall(line):
        try:
            value = float(num)
        except ValueError:
            continue
        key = letter.upper()
        # 同行多值时后写覆盖（与 G-code 解释器一致）
        out[key] = value
    return out


def parse_nc(path: str) -> NCParseResult:
    """只读解析一份 NC 文件。"""

    started = time.perf_counter()
    result = NCParseResult(path=path)

    try:
        with open(path, "r", encoding="gbk", errors="replace") as fh:
            lines = fh.readlines()
    except (OSError, UnicodeDecodeError) as exc:
        result.errors.append(f"打开/读取失败：{type(exc).__name__}: {exc}")
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    # 解析刀具头
    for line in lines:
        match = _TOOL_RE.search(line)
        if match:
            result.tool_name = match.group("name").strip()
            try:
                result.tool_diameter = float(match.group("d"))
            except ValueError:
                pass
            try:
                result.tool_radius = float(match.group("r"))
            except ValueError:
                pass
            break

    if not result.tool_name and result.tool_diameter is None:
        result.errors.append("未找到刀具头 (Tool=... D=... R=...)")

    # 当前模态（G0/G1/G2/G3/G81-83）与坐标
    modal_g: Optional[str] = None
    last_x: Optional[float] = None
    last_y: Optional[float] = None
    last_z: Optional[float] = None

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        # 去除注释 ( ... ) 和 ; 之后的内容
        if _PAREN in line:
            line = line.split(_PAREN, 1)[0].strip()
        if _SEMI in line:
            line = line.split(_SEMI, 1)[0].strip()
        if not line:
            continue
        if _END_RE.search(line):
            break

        # 单次分词：同时取坐标/参数（X/Y/Z/I/J/R/F）与 G 指令（热路径）。
        # G 指令取本行**最后一个**运动指令；G80/G28/G30/G53 清掉固定循环模态。
        coords: dict = {}
        cancel_cycle = False
        for letter, num in _LETTER_NUM_RE.findall(line):
            key = letter.upper()
            if key == "G":
                try:
                    gnum = int(float(num))
                except ValueError:
                    continue
                if gnum in _CANCEL_CODES:
                    cancel_cycle = True
                elif gnum in _MOTION_CODES:
                    modal_g = "G%d" % gnum
                continue
            # 坐标/参数：同行多值时后写覆盖（与 G-code 解释器一致）
            try:
                coords[key] = float(num)
            except ValueError:
                continue
        if cancel_cycle and modal_g in ("G81", "G82", "G83"):
            modal_g = None

        x = coords.get("X")
        y = coords.get("Y")
        z = coords.get("Z")
        i = coords.get("I")
        j = coords.get("J")
        r = coords.get("R")
        f = coords.get("F")

        # 本行没有任何 X/Y/Z/I/J：纯模态行（如 `G90`）不产生段
        if x is None and y is None and z is None and i is None and j is None:
            continue

        # 无 X/Y（只 Z）：沿用上一次的 XY；如果还没有，跳过
        if x is None and y is None:
            if modal_g in ("G81", "G82", "G83") and last_x is not None and last_y is not None:
                x = last_x
                y = last_y
            elif modal_g in ("G2", "G3") and (i is not None or j is not None) \
                    and last_x is not None and last_y is not None:
                # 整圆圆弧（只给 I/J，终点 = 起点）：沿用上一 XY
                x = last_x
                y = last_y
            else:
                # 保留 Z 供后续参考，但不产段
                if z is not None:
                    last_z = z
                continue

        # 补全未给出的坐标（沿用上一位置）
        if x is None:
            x = last_x
        if y is None:
            y = last_y

        # 段类型判定
        if modal_g in ("G81", "G82", "G83"):
            kind = SEG_DRILL
        elif modal_g in ("G2", "G3"):
            # 硬判据：圆弧行不带 Z = 平底，带 Z = 螺旋
            kind = SEG_ARC_HELICAL if z is not None else SEG_ARC_FLAT
        else:
            # modal_g 是 None / G0 / G1 或未知：都记 LINEAR
            kind = SEG_LINEAR

        # 圆弧半径与圆心
        radius: Optional[float] = None
        center_x: Optional[float] = None
        center_y: Optional[float] = None
        if i is not None or j is not None:
            ii = i if i is not None else 0.0
            jj = j if j is not None else 0.0
            radius = math.hypot(ii, jj)
            if x is not None and y is not None:
                # I/J 是相对**本段起点**的圆心偏移；本段起点 = 上一段终点（last_x/last_y）。
                start_x = last_x if last_x is not None else x
                start_y = last_y if last_y is not None else y
                center_x = start_x + ii
                center_y = start_y + jj

        result.segments.append(
            NCSegment(
                kind=kind,
                x=x,
                y=y,
                # 段的有效 Z = 本行 Z 或沿用上一段（模态 Z）；has_z 只标记本行是否显式写 Z
                z=(z if z is not None else last_z),
                i=i,
                j=j,
                r=r,
                f=f,
                radius=radius,
                center_x=center_x,
                center_y=center_y,
                has_z=(z is not None),
                line_no=line_no,
            )
        )

        # 更新位置
        if x is not None:
            last_x = x
        if y is not None:
            last_y = y
        if z is not None:
            last_z = z

    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result
