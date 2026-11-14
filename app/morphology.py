"""食双星形态判别 —— 纯 numpy/matplotlib 实现,无外部状态。

判定目标
========
给定同一颗星含 ``night_id、time_day、flux、flux_err`` 的测光 CSV 与周期
筛查得到的最高峰周期 P,判断相位图中的两个疑似食谷是否构成**双食结构**
(即真实轨道周期可能为 2P、两次食深度稳定不同的食双星候选)。

判定原则(与业务规则一一对应)
------------------------------
1. 分别按 P 与 2P 折相;谷底与同夜基线的深度一律做**同夜基线校正**,
   夜间零点差只进基线、不进深度,因此单晚密集采样或夜间零点差本身
   不能充当证据;
2. 两类食谷(主/次食)各须有 **>=2 个不同夜晚**同时覆盖谷底与邻近
   基线,否则判「待复核」并列出逐夜覆盖缺口;
3. 仅当校正同夜基线后的深度差**超过测量误差**(加权 z 检验)、且
   2P 相对 P 的拟合改善在**不同夜晚可重复**(留一夜交叉验证,支持
   2P 的夜晚不少于 2 个且严格多于矛盾夜晚)时,才判「双食结构」;
4. 覆盖充分但无稳定差异(深度差不显著,或拟合改善不可重复)时判
   「未见双食结构」;逐夜结论矛盾(支持 P 的夜晚与支持 2P 的夜晚
   相当且矛盾夜晚占比过高)判「待复核」。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# 判定阈值(集中定义,便于审计)
# --------------------------------------------------------------------------- #
MIN_NIGHTS_PER_ECLIPSE = 2     # 每类食谷要求的最少覆盖夜晚数
MIN_POINTS_REGION = 2          # 谷底/基线各自要求的最少点数
DEPTH_Z_THRESHOLD = 2.0        # 深度差超过测量误差的 z 阈值
FIT_REL_IMPROVEMENT = 0.10      # 留一夜 χ² 改善 >=10% 才算该夜支持 2P
FIT_STRONG_CONTRARY = 0.10      # P 反优 >=10% 记为矛盾夜晚
N_BINS_2P = 80                  # 2P 折相模板的相箱数
N_BINS_P = 40                   # P 折相模板的相箱数(P 一周期内相箱数与 2P 等密度)
MIN_ECLIPSE_SEPARATION = 0.10   # 两个食谷中心的最小 2P 相位间隔
MAX_HALF_WIDTH = 0.20           # 食窗半宽上限(2P 相位单位)
MIN_HALF_WIDTH = 0.008          # 食窗半宽下限(约一个细箱宽度)

VERDICT_DOUBLE = "double_eclipse"        # 双食结构
VERDICT_NONE = "no_double_eclipse"       # 未见双食结构
VERDICT_REVIEW = "needs_review"          # 待复核

VERDICT_TEXT = {
    VERDICT_DOUBLE: "双食结构(建议按食双星复核)",
    VERDICT_NONE: "未见双食结构",
    VERDICT_REVIEW: "待复核",
}


@dataclass
class NightRecord:
    """单个夜晚对两类食谷的覆盖与深度测量结果。"""

    night_id: str
    n_points: int
    # 每个食谷一项;不可用时对应字段为 None
    bottom_n: list[int | None]
    baseline_n: list[int | None]
    depth: list[float | None]          # 同夜基线校正后的深度(正值=向下凹陷)
    depth_err: list[float | None]
    # 留一夜交叉验证的拟合表现(该夜作为测试夜)
    chi2_2p: float | None = None
    chi2_p: float | None = None
    fit_improvement: float | None = None   # (χ²_P - χ²_2P)/χ²_P
    fit_verdict: str = "not_evaluable"     # supports_2p / supports_p / neutral / not_evaluable

    def to_dict(self) -> dict:
        return {
            "night_id": self.night_id,
            "n_points": self.n_points,
            "bottom_n": self.bottom_n,
            "baseline_n": self.baseline_n,
            "depth": [None if v is None else float(v) for v in self.depth],
            "depth_err": [None if v is None else float(v) for v in self.depth_err],
            "chi2_2p": self.chi2_2p,
            "chi2_p": self.chi2_p,
            "fit_improvement": self.fit_improvement,
            "fit_verdict": self.fit_verdict,
        }


@dataclass
class MorphResult:
    verdict: str
    reasons: list[str]
    n_rows: int
    n_nights: int
    period: float
    phase_p: np.ndarray
    phase_2p: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    night_idx: np.ndarray
    night_names: list[str]
    eclipse_centers: np.ndarray          # 两个食谷的 2P 相位中心,长度 2
    half_width: float                    # 食窗半宽(2P 相位单位)
    depths: list[float | None]           # 两类食谷的跨夜加权深度
    depth_errs: list[float | None]
    depth_nights: list[int]              # 参与各食谷深度汇总的夜晚数
    depth_diff: float | None
    depth_diff_err: float | None
    depth_z: float | None
    support_nights: list[str]
    contrary_nights: list[str]
    neutral_nights: list[str]
    global_chi2_2p: float
    global_chi2_p: float
    nights: list[NightRecord] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

def fold_phase(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    return ((t - t0) / period) % 1.0


def _weighted_median_uncertainty(err: np.ndarray) -> float:
    """中位 flux 的不确定度:1.2533·σ/√n(σ 由逐点测量误差合成)。"""
    if err.size == 0:
        return math.nan
    sigma = math.sqrt(1.0 / np.sum(1.0 / np.maximum(err, 1e-12) ** 2))
    return 1.2533 * sigma / math.sqrt(err.size)


def _in_eclipse(phase_2p: np.ndarray, center: float, half: float) -> np.ndarray:
    """环形相位上判断点是否落在 [center-half, center+half] 内。"""
    d = np.abs(((phase_2p - center + 0.5) % 1.0) - 0.5)
    return d <= half


# --------------------------------------------------------------------------- #
# 食谷定位(在 2P 折相上做,两个食谷本就分处一周期的不同相位)
# --------------------------------------------------------------------------- #

def locate_eclipses(phase_2p: np.ndarray, flux: np.ndarray,
                    flux_err: np.ndarray) -> tuple[np.ndarray, float]:
    """在 2P 相位上定位两个最深的凹陷中心,返回 (中心[2], 食窗半宽)。

    采用粗细两级网格:粗网格(60 箱)每箱点数充足,用于稳健定位谷底;
    细网格(120 箱)分辨窄食谷,用于按凹陷阈值估计食窗半宽。
    """
    n_coarse = 60
    edges_c = np.linspace(0.0, 1.0, n_coarse + 1)
    idx_c = np.clip(np.digitize(phase_2p, edges_c) - 1, 0, n_coarse - 1)
    med_c = np.full(n_coarse, np.nan)
    cnt_c = np.zeros(n_coarse, dtype=int)
    for b in range(n_coarse):
        m = idx_c == b
        cnt_c[b] = int(m.sum())
        if m.any():
            med_c[b] = np.median(flux[m])

    # 全局基线:有数据相箱的上四分位(食谷向下,正常测光位于高位)
    filled = med_c[np.isfinite(med_c)]
    baseline_level = np.percentile(filled, 75) if filled.size else np.median(flux)

    # 第一个食谷:中位 flux 最低的粗箱,要求至少 3 个点以免单噪声点成谷
    valid = np.isfinite(med_c) & (cnt_c >= 3)
    if not valid.any():  # 兜底:退化为全部相箱
        valid = np.isfinite(med_c)
    b1 = int(np.argmin(np.where(valid, med_c, np.inf)))
    center1 = (edges_c[b1] + edges_c[b1 + 1]) / 2.0

    # 第二个食谷:与 b1 相隔 >=5 个粗箱、深度最明显的次低箱
    ring_dist = np.minimum(np.abs(np.arange(n_coarse) - b1),
                           n_coarse - np.abs(np.arange(n_coarse) - b1))
    valid2 = valid & (ring_dist >= 5)
    if not valid2.any():
        valid2 = ring_dist >= 5
    b2 = int(np.argmin(np.where(valid2, med_c, np.inf)))
    center2 = (edges_c[b2] + edges_c[b2 + 1]) / 2.0

    # ---- 细网格:在两个粗谷底附近估计食窗半宽 ----
    # 先在粗谷底 ±2 个粗箱范围内,用细网格重新确定真正的最深箱中心,
    # 消除粗/细箱中心错位对宽度扩展的影响。
    n_fine = 120
    edges_f = np.linspace(0.0, 1.0, n_fine + 1)
    idx_f = np.clip(np.digitize(phase_2p, edges_f) - 1, 0, n_fine - 1)
    med_f = np.full(n_fine, np.nan)
    for b in range(n_fine):
        m = idx_f == b
        if m.any():
            med_f[b] = np.median(flux[m])
    fine_centers = (np.arange(n_fine) + 0.5) / n_fine

    refined: list[tuple[int, float]] = []  # (最深细箱, 精化后的中心)
    for bc in (b1, b2):
        coarse_center = (edges_c[bc] + edges_c[bc + 1]) / 2.0
        ring = np.abs(((fine_centers - coarse_center + 0.5) % 1.0) - 0.5)
        local = np.where(ring <= 2.0 / n_coarse)[0]
        local = local[np.isfinite(med_f[local])]
        cf = int(local[np.argmin(med_f[local])])
        refined.append((cf, float(fine_centers[cf])))

    # 思路:对每个食谷,从最深细箱出发沿环形相位两侧扩展,只要相邻细箱
    # 的凹陷仍 >= 该食谷深度的 50% 就并入,遇到第一个不达标箱或空箱即
    # 停止(不跨箱延伸)。50% 阈值对应高斯食形约 ±1.18σ 的核心食相,
    # 对逐箱噪声稳健;食窗只圈核心食相,深度比较聚焦在谷底而非谷肩。
    # 两个食谷共用同一个对称食窗半宽,保证两类食谷按同一口径比较深度,
    # 避免稀疏采样的不对称给等深食谷制造虚假深度差。
    per_eclipse_reach = []
    for bc, (cf, _) in zip((b1, b2), refined):
        eclipse_depth = max(baseline_level - med_c[bc], 0.0)
        dip_thresh = 0.50 * eclipse_depth

        def reach_toward(sign: int) -> int:
            """沿 sign(±1)方向统计连续达标细箱数;空箱即停止,不跨箱。"""
            reach = 0
            for step in range(1, n_fine // 2):
                bb = (cf + sign * step) % n_fine
                if not np.isfinite(med_f[bb]):
                    break
                if baseline_level - med_f[bb] < dip_thresh:
                    break
                reach = step
            return reach

        # 该食谷在两个方向上的连续延伸;取较小者作为它能稳健支撑的半宽
        per_eclipse_reach.append(min(reach_toward(+1), reach_toward(-1)))
    # 两个食谷取一致半宽(较保守的一者),再加半箱到箱缘
    reach = min(per_eclipse_reach)
    half = float(np.clip((reach + 0.5) / n_fine,
                         MIN_HALF_WIDTH, MAX_HALF_WIDTH))

    # 深谷在前(主食),浅谷在后(次食);中心采用细网格精化值
    centers = np.array([c for _, c in refined], dtype=float)
    order = np.argsort([med_c[b1], med_c[b2]])
    centers = centers[order]
    return centers, half


# --------------------------------------------------------------------------- #
# 逐夜深度(同夜基线校正)
# --------------------------------------------------------------------------- #

def per_night_depths(phase_2p: np.ndarray, flux: np.ndarray,
                     flux_err: np.ndarray, night_idx: np.ndarray,
                     n_nights: int, centers: np.ndarray,
                     half: float) -> list[NightRecord]:
    in1 = _in_eclipse(phase_2p, centers[0], half)
    in2 = _in_eclipse(phase_2p, centers[1], half)
    in_any = in1 | in2

    records: list[NightRecord] = []
    for ni in range(n_nights):
        sel = night_idx == ni
        rec = NightRecord(
            night_id=str(ni),
            n_points=int(sel.sum()),
            bottom_n=[None, None],
            baseline_n=[None, None],
            depth=[None, None],
            depth_err=[None, None],
        )
        base_sel = sel & ~in_any
        base_flux = flux[base_sel]
        base_err = flux_err[base_sel]
        for k, mask in enumerate((in1, in2)):
            bot_sel = sel & mask
            bot_flux = flux[bot_sel]
            bot_err = flux_err[bot_sel]
            rec.bottom_n[k] = int(bot_flux.size)
            rec.baseline_n[k] = int(base_flux.size)
            if bot_flux.size >= MIN_POINTS_REGION and base_flux.size >= MIN_POINTS_REGION:
                bot_med = float(np.median(bot_flux))
                base_med = float(np.median(base_flux))
                # 深度 = 同夜基线 - 谷底(向下凹陷为正);同夜相减自动扣除夜零点
                rec.depth[k] = base_med - bot_med
                sig = math.hypot(_weighted_median_uncertainty(bot_err),
                                 _weighted_median_uncertainty(base_err))
                rec.depth_err[k] = sig if math.isfinite(sig) and sig > 0 else None
        records.append(rec)
    return records


def aggregate_depths(records: list[NightRecord]) -> tuple[
        list[float | None], list[float | None], list[int],
        float | None, float | None, float | None]:
    """跨夜逆方差加权汇总两类食谷深度,并做深度差 z 检验。

    逐夜深度的形式误差只含测量误差;若各夜结果存在超出形式误差的真实
    离散(约化 χ²>1,来自窗口采样、定标残差等),按 sqrt(χ²_red) 膨胀
    不确定度,避免把夜间系统差误判为两类食谷的稳定深度差。
    """
    depths: list[float | None] = [None, None]
    errs: list[float | None] = [None, None]
    n_nights_used = [0, 0]
    for k in range(2):
        vals = np.array([r.depth[k] for r in records if r.depth[k] is not None
                         and r.depth_err[k] is not None], dtype=float)
        sigs = np.array([r.depth_err[k] for r in records if r.depth[k] is not None
                         and r.depth_err[k] is not None], dtype=float)
        n_nights_used[k] = int(vals.size)
        if vals.size:
            w = 1.0 / sigs ** 2
            mean = float(np.sum(w * vals) / np.sum(w))
            base_var = 1.0 / np.sum(w)
            if vals.size >= 2:
                chi2_red = float(np.sum(w * (vals - mean) ** 2)
                                 / (vals.size - 1))
                # 超散布膨胀(只放大、不缩小)
                base_var *= max(1.0, chi2_red)
            depths[k] = mean
            errs[k] = float(math.sqrt(base_var))
    diff = diff_err = z = None
    if depths[0] is not None and depths[1] is not None:
        diff = depths[0] - depths[1]
        diff_err = math.hypot(errs[0], errs[1])
        z = diff / diff_err if diff_err > 0 else math.inf
    return depths, errs, n_nights_used, diff, diff_err, z


# --------------------------------------------------------------------------- #
# 留一夜(Leave-One-Night-Out)拟合对比:2P 模板 vs P 模板
# --------------------------------------------------------------------------- #

def _binned_template(phase_train: np.ndarray, flux_train: np.ndarray,
                     n_bins: int) -> np.ndarray:
    """训练夜相位分箱中位模板;空箱回退全局中位,保证任意测试点可预测。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(phase_train, edges) - 1, 0, n_bins - 1)
    template = np.full(n_bins, np.median(flux_train))
    for b in range(n_bins):
        m = idx == b
        if m.any():
            template[b] = np.median(flux_train[m])
    return template


def lono_fit_compare(phase_p: np.ndarray, phase_2p: np.ndarray,
                     flux: np.ndarray, flux_err: np.ndarray,
                     night_idx: np.ndarray, n_nights: int,
                     records: list[NightRecord]) -> tuple[
                        list[NightRecord], float, float, list[str], list[str], list[str]]:
    """每个夜晚轮流作为测试夜,其余夜建模板,比较 P / 2P 折相的 χ²。"""
    total_chi2_2p = 0.0
    total_chi2_p = 0.0
    support: list[str] = []
    contrary: list[str] = []
    neutral: list[str] = []

    for ni in range(n_nights):
        test = night_idx == ni
        train = ~test
        if train.sum() < N_BINS_P // 2 or test.sum() < 3:
            continue  # 训练/测试样本不足以做留一夜验证
        tpl_2p = _binned_template(phase_2p[train], flux[train], N_BINS_2P)
        tpl_p = _binned_template(phase_p[train], flux[train], N_BINS_P)

        idx_2p = np.clip(np.digitize(phase_2p[test],
                                     np.linspace(0, 1, N_BINS_2P + 1)) - 1,
                         0, N_BINS_2P - 1)
        idx_p = np.clip(np.digitize(phase_p[test],
                                    np.linspace(0, 1, N_BINS_P + 1)) - 1,
                        0, N_BINS_P - 1)
        resid_2p = flux[test] - tpl_2p[idx_2p]
        resid_p = flux[test] - tpl_p[idx_p]
        e2 = np.maximum(flux_err[test], 1e-9) ** 2
        chi2_2p = float(np.sum(resid_2p ** 2 / e2))
        chi2_p = float(np.sum(resid_p ** 2 / e2))

        rec = records[ni]
        rec.chi2_2p = chi2_2p
        rec.chi2_p = chi2_p
        rec.fit_improvement = (chi2_p - chi2_2p) / chi2_p if chi2_p > 0 else 0.0
        if rec.fit_improvement >= FIT_REL_IMPROVEMENT:
            rec.fit_verdict = "supports_2p"
            support.append(rec.night_id)
        elif rec.fit_improvement <= -FIT_STRONG_CONTRARY:
            rec.fit_verdict = "supports_p"
            contrary.append(rec.night_id)
        else:
            rec.fit_verdict = "neutral"
            neutral.append(rec.night_id)
        total_chi2_2p += chi2_2p
        total_chi2_p += chi2_p
    return records, total_chi2_2p, total_chi2_p, support, contrary, neutral


# --------------------------------------------------------------------------- #
# 主分析入口
# --------------------------------------------------------------------------- #

def analyze(t: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
            night_raw: np.ndarray, period: float) -> MorphResult:
    """执行完整形态判别。所有输入数组已通过 main 层校验。"""
    # night_id 统一转字符串类别,保持首次出现顺序
    night_names: list[str] = []
    name_to_idx: dict[str, int] = {}
    night_idx = np.empty(t.size, dtype=int)
    for i, raw in enumerate(night_raw):
        key = str(raw)
        if key not in name_to_idx:
            name_to_idx[key] = len(night_names)
            night_names.append(key)
        night_idx[i] = name_to_idx[key]
    n_nights = len(night_names)

    t0 = float(t.min())
    phase_p = fold_phase(t, period, t0)
    phase_2p = fold_phase(t, 2.0 * period, t0)

    centers, half = locate_eclipses(phase_2p, flux, flux_err)
    records = per_night_depths(phase_2p, flux, flux_err, night_idx,
                               n_nights, centers, half)
    (depths, depth_errs, depth_nights,
     depth_diff, depth_diff_err, depth_z) = aggregate_depths(records)
    (records, global_chi2_2p, global_chi2_p,
     support, contrary, neutral) = lono_fit_compare(
        phase_p, phase_2p, flux, flux_err, night_idx, n_nights, records)
    for rec, name in zip(records, night_names):
        rec.night_id = name

    reasons: list[str] = []

    # ---- 覆盖检查:两类食谷各须 >=2 个不同夜晚同时覆盖谷底与邻近基线 ----
    coverage_ok = True
    for k, label in enumerate(("主食谷", "次食谷")):
        covered = [r.night_id for r in records
                   if (r.bottom_n[k] or 0) >= MIN_POINTS_REGION
                   and (r.baseline_n[k] or 0) >= MIN_POINTS_REGION]
        if len(covered) < MIN_NIGHTS_PER_ECLIPSE:
            coverage_ok = False
            reasons.append(
                f"{label}仅 {len(covered)} 个夜晚同时覆盖谷底与邻近基线"
                f"(要求 ≥{MIN_NIGHTS_PER_ECLIPSE});覆盖夜晚:"
                f"{', '.join(covered) if covered else '无'}")
        else:
            reasons.append(f"{label}覆盖夜晚({len(covered)} 个):"
                           f"{', '.join(covered)}")

    sep = float(np.abs(((centers[0] - centers[1] + 0.5) % 1.0) - 0.5))
    if sep < MIN_ECLIPSE_SEPARATION:
        reasons.append(f"两个凹陷的 2P 相位间隔仅 {sep:.3f}"
                       f"(< {MIN_ECLIPSE_SEPARATION}),实为同一凹陷的重复,")
        coverage_ok = False

    # ---- 深度差显著性 ----
    depth_significant = (depth_z is not None
                         and abs(depth_z) >= DEPTH_Z_THRESHOLD)
    if depth_z is None:
        reasons.append("两类食谷深度无法同时跨夜汇总,不能比较深度差。")
    else:
        reasons.append(
            f"同夜基线校正后的加权深度:主食 {fmt(depths[0])} ± {fmt(depth_errs[0])},"
            f"次食 {fmt(depths[1])} ± {fmt(depth_errs[1])};"
            f"深度差 {fmt(depth_diff)} ± {fmt(depth_diff_err)},"
            f"z = {depth_z:.2f}(阈值 |z| ≥ {DEPTH_Z_THRESHOLD})。")

    # ---- 2P 拟合改善的跨夜可重复性 ----
    n_eval = len(support) + len(contrary) + len(neutral)
    repeatable = (len(support) >= MIN_NIGHTS_PER_ECLIPSE
                  and len(support) > len(contrary))
    if n_eval == 0:
        reasons.append("留一夜交叉验证无法执行(夜晚数或每夜点数不足),"
                       "无法评估 2P 相对 P 的拟合改善。")
    else:
        reasons.append(
            f"留一夜交叉验证:{n_eval} 个夜晚可评估;支持 2P 折相的夜晚 "
            f"{len(support)} 个({', '.join(support) if support else '无'}),"
            f"矛盾(支持 P)夜晚 {len(contrary)} 个"
            f"({', '.join(contrary) if contrary else '无'}),"
            f"中性 {len(neutral)} 个;全局 χ²:2P={global_chi2_2p:.1f},"
            f"P={global_chi2_p:.1f}。")

    # 矛盾仅指:深度差本身显著,但 2P 的拟合改善在不同夜晚重复不出来
    # (支持 P 的夜晚与支持 2P 的夜晚相当且各 >=2);等深情形下留一夜
    # 检验天然偏向 P,属于「无稳定改善」而非夜间矛盾
    contradictory = (depth_significant and n_eval > 0
                     and len(contrary) >= MIN_NIGHTS_PER_ECLIPSE
                     and len(contrary) >= len(support))

    # ---- 终判 ----
    if not coverage_ok:
        verdict = VERDICT_REVIEW
        reasons.append("结论:覆盖不足,标「待复核」;单晚密集采样或夜间零点差"
                       "不能充当双食证据。")
    elif contradictory:
        verdict = VERDICT_REVIEW
        reasons.append("结论:深度差虽显著,但不同夜晚的拟合结论矛盾、"
                       "2P 改善不可重复,标「待复核」。")
    elif depth_significant and repeatable:
        verdict = VERDICT_DOUBLE
        reasons.append("结论:深度差超过测量误差且 2P 拟合改善跨夜可重复,"
                       "标「双食结构」,建议按食双星(真实周期约 2P)复核。")
    else:
        verdict = VERDICT_NONE
        if not depth_significant:
            reasons.append("深度差未超过测量误差;")
        if not repeatable:
            reasons.append("2P 相对 P 的拟合改善未在 ≥2 个不同夜晚重复"
                           "(或矛盾夜晚更多);")
        reasons.append("结论:覆盖充分但无稳定差异,标「未见双食结构」。")

    return MorphResult(
        verdict=verdict,
        reasons=reasons,
        n_rows=int(t.size),
        n_nights=n_nights,
        period=float(period),
        phase_p=phase_p,
        phase_2p=phase_2p,
        flux=flux,
        flux_err=flux_err,
        night_idx=night_idx,
        night_names=night_names,
        eclipse_centers=centers,
        half_width=half,
        depths=depths,
        depth_errs=depth_errs,
        depth_nights=depth_nights,
        depth_diff=depth_diff,
        depth_diff_err=depth_diff_err,
        depth_z=depth_z,
        support_nights=support,
        contrary_nights=contrary,
        neutral_nights=neutral,
        global_chi2_2p=float(global_chi2_2p),
        global_chi2_p=float(global_chi2_p),
        nights=records,
    )


def fmt(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.4g}"
