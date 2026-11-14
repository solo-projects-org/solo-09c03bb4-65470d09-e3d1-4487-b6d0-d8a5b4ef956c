"""生成食双星形态判别的虚构样例(sample_eclipsing_binary.csv)。

虚构目标:一颗轨道周期 6.6 天的分离食双星,主、次食深度明显不同
(主食 0.55、次食 0.22 相对流量单位)。巡测的 Lomb–Scargle 往往先检出
半周期 3.3 天(交替食谷在 P 折相上重叠为两个疑似食谷),因此页面上
填写的最高峰周期 P = 3.3 天,形态判别应在 2P = 6.6 天折相上把两谷
分开并判为「双食结构」。

样例刻意包含:
* 多个不同 night_id 的夜晚,其中 20 个夜晚的窗口对准交替发生的食
  事件(每类食谷约 10 个夜晚),窗口宽于食谷,使同一夜晚既有谷底点
  也有邻近基线点;另有 12 个随机夜晚主要覆盖食外基线;
* 每夜独立的零点偏移(幅度大于测光噪声),验证同夜基线校正不会把
  夜间零点差误当成深度证据;
* 逐点 flux_err(全为正),深度检验与拟合 χ² 均按误差加权。

固定随机种子,结果可复现。
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "sample_eclipsing_binary.csv"

rng = np.random.default_rng(20261114)

T_ORB = 6.6            # 真实轨道周期(天)
P_PEAK = T_ORB / 2.0   # 周期筛查常见的最高峰周期(半周期)
DEPTH1 = 0.55          # 主食深度
DEPTH2 = 0.22          # 次食深度
ECL_SIGMA = 0.010      # 食谷在轨道相位上的宽度(1σ),对应约 0.066 天
NIGHT_WINDOW = 0.50    # 单晚观测窗口(天),宽于食谷,谷底两侧留基线
BASELINE = 10.0
NOISE = 0.012

night_starts: list[float] = []
night_jitter: list[float] = []

# 20 个夜晚刻意对准交替发生的食事件(带小抖动,保证多夜覆盖两类食谷)
for k in range(20):
    epoch = k * P_PEAK                    # 每隔半周期一次食,主/次交替
    jitter = rng.uniform(-0.03, 0.03)
    # 窗口中心对准食事件:start = epoch - 半窗 + 小抖动
    night_starts.append(epoch - NIGHT_WINDOW / 2.0 + jitter)
    night_jitter.append(jitter)

# 12 个随机夜晚(主要覆盖食外基线,模拟真实巡测的不规则夜分布)
for _ in range(12):
    night_starts.append(rng.uniform(0.0, 175.0))
    night_jitter.append(0.0)

order = np.argsort(night_starts)
night_starts = [night_starts[i] for i in order]

rows: list[tuple[str, float, float, float]] = []
for ni, start in enumerate(night_starts):
    n_pts = int(rng.integers(9, 14))
    # 单晚密集采样:0.50 天内 9~13 个点
    local = np.sort(rng.uniform(0.0, NIGHT_WINDOW, size=n_pts))
    # 同一晚共享零点偏移(幅度大于噪声,模拟夜间零点差)
    zero = rng.normal(0.0, 0.08)
    for tt in start + local:
        phase_orb = (tt / T_ORB) % 1.0
        d1 = abs(((phase_orb - 0.0 + 0.5) % 1.0) - 0.5)
        d2 = abs(((phase_orb - 0.5 + 0.5) % 1.0) - 0.5)
        dip = (DEPTH1 * np.exp(-0.5 * (d1 / ECL_SIGMA) ** 2)
               + DEPTH2 * np.exp(-0.5 * (d2 / ECL_SIGMA) ** 2))
        err = float(rng.uniform(0.010, 0.016))
        flux = BASELINE + zero - dip + rng.normal(0.0, NOISE)
        rows.append((f"night_{ni:02d}", tt, flux, err))

with OUT.open("w", encoding="utf-8") as fh:
    fh.write("night_id,time_day,flux,flux_err\n")
    for nid, tt, fl, ee in rows:
        fh.write(f"{nid},{tt:.6f},{fl:.6f},{ee:.5f}\n")

print(f"已写出 {OUT}({len(rows)} 行,{len(night_starts)} 个夜晚,"
      f"轨道周期 {T_ORB} 天,建议填写 P = {P_PEAK} 天)")
