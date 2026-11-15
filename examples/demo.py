"""最小示例：解析一份 NC 文件并识别沉头孔底面平台。

运行（需先 `pip install -e .`，或从仓库根目录用 `python -m examples.demo`）：

    python examples/demo.py
"""

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
