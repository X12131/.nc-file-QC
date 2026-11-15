# ncparse — NC 刀轨只读解析与沉头孔底面平台识别

从 CNC 后处理输出的 NC 程序（G 代码文本，GBK 编码）中**只读**解析刀轨段，
并识别**沉头孔底面平台**（沉头上部的扩孔平底）的 XY 位置与平台 Z 深度。
零第三方运行时依赖，纯 Python 标准库实现。

## 安装

要求 Python ≥ 3.10。

```bash
git clone <本仓库地址>
cd ncparse
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"      # 运行时零依赖；[dev] 仅安装 pytest
```

不想安装也可以：仓库根目录有 `conftest.py`，直接在根目录运行 `pytest -q` 即可。

## 用法

最小示例（与 [examples/demo.py](examples/demo.py) 相同，可直接 `python examples/demo.py` 运行）：

```python
"""最小示例：解析一份 NC 文件并识别沉头孔底面平台。"""

from __future__ import annotations

import json
import os
import tempfile

from ncparse import analyze_nc_counterbores, parse_nc

NC_TEXT = """%
O0100
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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "demo.nc")
        with open(path, "w", encoding="gbk", errors="replace") as fh:
            fh.write(NC_TEXT)

        result = parse_nc(path)
        print("刀具：", result.tool_name, result.tool_diameter)
        print("段数：", len(result.segments))

        cb = analyze_nc_counterbores(result)
        print("沉头孔平台：", json.dumps(cb.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

带要求值的用法（平台铣削与通孔钻削分属两份 NC 时，用「深度 + 直径」签名识别）：

```python
cb = analyze_nc_counterbores(result, required_depth=4.0, required_diameter=20.0)
```

## 输出说明

### `parse_nc(path)` → `NCParseResult`

| 字段 | 含义 |
|---|---|
| `tool_name` / `tool_diameter` / `tool_radius` | 刀具头，来自 `(Tool=<名> D=<直径> R=<圆角>)` |
| `segments` | 段列表，类型见下 |
| `errors` | 单文件解析异常（**不抛**，留给调用方记录） |
| `elapsed_ms` | 解析耗时 |

`NCSegment.kind`：

| 值 | 含义 |
|---|---|
| `ARC_FLAT` | 平底圆弧：`G2/G3 X.. Y.. I.. J..`，圆弧行不带 Z |
| `ARC_HELICAL` | 螺旋下钻：`G2/G3 X.. Y.. Z.. I.. J..`，圆弧行带 Z |
| `DRILL` | 固定循环 `G81/G82/G83`（Z 在参数里） |
| `LINEAR` | 直线插补 `G0/G1` |

段字段：`x/y/z/i/j/r/f`、`radius`（圆弧半径 = √(I²+J²)）、`center_x/center_y`
（圆心绝对坐标 = 起点 + I/J）、`has_z`（本行是否显式写 Z）、`line_no`。
`z` 是该段执行时的**实际模态 Z**——本行没写 Z 就沿用上一段的 Z。

### `analyze_nc_counterbores(parse_result)` → `CounterboreResult`

| 字段 | 含义 |
|---|---|
| `clusters` | 平台候选簇列表 `BoreCluster` |
| `inconsistent` | 同一 NC 内多簇平台 Z 不一致（相差超过 `depth_tolerance_mm`） |
| `platform_depths` | 去重后的平台 Z 列表 |
| `has_counterbore` | 是否发现平台 |
| `error` | 上游解析错误汇总（错误时不识别） |

`BoreCluster` 关键字段：`x/y`（圆心均值）、`platform_z`（平台底面 Z，平底圆弧最深 Z）、
`arc_radius`（平台圆弧半径）、`effective_hole_diameter`（有效孔径 = 刀具直径 + 2×圆弧半径）、
`matched_by`（命中判据）：

| `matched_by` | 判据 |
|---|---|
| `same_cluster_drill` | A：同簇存在更深的螺旋/下钻（平台下面还有通孔） |
| `depth_diameter_signature` | B：`|Z| ≈ required_depth` 且 有效孔径 ≈ `required_diameter`（需传入要求值） |
| `arc_flat_sweep` | C：定深平底弧扫掠形态（螺旋进刀 + 平底弧扫掠 + 整圆收尾 + 同 XY 更深 + 上大下小），包络孔径 ≈ `required_diameter` |

判据 A/B 先跑，**两者都落空时**才跑判据 C（兜底语义，不改变 A/B 结论）；
判据 B/C 需要 `required_depth` / `required_diameter`（可缺）。

## 运行测试

```bash
pytest -q
```

如需对本机真实 NC 样本做回归，设置环境变量 `NCPARSE_SAMPLE_DIR` 指向样本目录
（目录内存放 `.nc` 文件）；未设置时相关用例自动跳过。

## 已知限制

- **真实产线样本未随仓库发布**。测试使用脱敏合成 NC 样本覆盖全部判据路径；
  真实样本回归需自行提供（见上节环境变量）。
- **不做切削仿真**：只做文本级解析与几何识别，不验证刀轨的加工可行性。
- 按「智模」类后处理方言假设文件结构：首行 `%`、单文件单刀、G54 绝对坐标、
  圆弧 I/J 为相对起点圆心偏移。其他后处理方言未适配。
- 文件按 **GBK** 编码读取（`errors="replace"`），UTF-8 NC 需自行转码。
- 判据面向沉头孔底面平台的量测，不保证覆盖所有腔体/清角形态；
  判据 C 只负责量出平台 Z，容差判定（做深/做浅口径）在上层规则完成。

## 许可证

[MIT](LICENSE)
