"""形态判别结果绘图:P/2P 折相对比图与逐夜深度覆盖图。"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .morphology import MorphResult


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_fold_comparison(res: MorphResult) -> bytes:
    """左:按 P 折相(两个疑似食谷叠在同一相位);右:按 2P 折相(两谷分离)。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(res.n_nights)]

    for ni in range(res.n_nights):
        m = res.night_idx == ni
        c = colors[ni]
        ph1 = res.phase_p[m]
        ax1.errorbar(np.concatenate([ph1, ph1 + 1.0]),
                     np.concatenate([res.flux[m], res.flux[m]]),
                     yerr=np.concatenate([res.flux_err[m], res.flux_err[m]]),
                     fmt="o", ms=3.2, alpha=0.8, color=c, ecolor=c,
                     elinewidth=0.6, capsize=0)
        ax2.errorbar(res.phase_2p[m], res.flux[m], yerr=res.flux_err[m],
                     fmt="o", ms=3.2, alpha=0.8, color=c, ecolor=c,
                     elinewidth=0.6, capsize=0)

    ax1.set_xlim(0, 2)
    ax2.set_xlim(0, 1)
    ax1.set_xlabel(f"Phase @ P = {res.period:.6g} d (two cycles)")
    ax2.set_xlabel(f"Phase @ 2P = {2 * res.period:.6g} d (one cycle)")
    ax1.set_ylabel("Flux")
    ax1.set_title("Folded at P: two dips overlap")
    ax2.set_title("Folded at 2P: primary / secondary separated")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)

    # 2P 面板上标出两个食窗与中心
    for k, (center, depth) in enumerate(zip(res.eclipse_centers, res.depths)):
        ax2.axvspan(center - res.half_width, center + res.half_width,
                    color="#d62728", alpha=0.12)
        ax2.axvline(center, color="#d62728", lw=1.0, ls="--")
        ax2.text(center, ax2.get_ylim()[0], f" E{k + 1}",
                 color="#a02018", fontsize=9, va="bottom", ha="center")

    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[i % 20],
                          label=res.night_names[i])
               for i in range(min(res.n_nights, 10))]
    ax2.legend(handles=handles, fontsize=7, loc="best", framealpha=0.85,
               title=("nights (first 10)" if res.n_nights > 10 else "nights"))
    fig.tight_layout()
    return fig_to_png(fig)


def make_night_chart(res: MorphResult) -> bytes:
    """逐夜深度(带误差棒)与留一夜拟合结论;覆盖缺口留空。"""
    n = len(res.nights)
    x = np.arange(n, dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.5, 0.42 * n + 2.0), 4.6))

    e1_d = np.array([r.depth[0] if r.depth[0] is not None else np.nan
                     for r in res.nights])
    e2_d = np.array([r.depth[1] if r.depth[1] is not None else np.nan
                     for r in res.nights])
    e1_s = np.array([r.depth_err[0] if r.depth_err[0] is not None else 0.0
                     for r in res.nights])
    e2_s = np.array([r.depth_err[1] if r.depth_err[1] is not None else 0.0
                     for r in res.nights])

    ax.bar(x - width / 2, e1_d, width=width, yerr=e1_s, capsize=3,
           color="#1f5fa8", alpha=0.85, label="E1 depth (same-night baseline)",
           error_kw={"elinewidth": 0.8})
    ax.bar(x + width / 2, e2_d, width=width, yerr=e2_s, capsize=3,
           color="#e08a1e", alpha=0.85, label="E2 depth (same-night baseline)",
           error_kw={"elinewidth": 0.8})

    # 留一夜拟合结论标记(放在绘图区内顶部,避免与标题重叠)
    mark = {"supports_2p": ("v2P", "#1e8449"),
            "supports_p": ("xP", "#c0392b"),
            "neutral": ("-", "#777"),
            "not_evaluable": ("", "#777")}
    y_lo, y_hi = ax.get_ylim()
    y_mark = y_hi - 0.06 * (y_hi - y_lo)
    for i, r in enumerate(res.nights):
        txt, col = mark[r.fit_verdict]
        if txt:
            ax.text(x[i], y_mark, txt, color=col, fontsize=8,
                    ha="center", va="top")
    ax.set_ylim(top=y_hi + 0.12 * (y_hi - y_lo))

    ax.axhline(0.0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.night_id for r in res.nights], rotation=45,
                       ha="right", fontsize=8)
    ax.set_ylabel("Depth (baseline - bottom)")
    ax.set_title("Per-night eclipse depths and leave-one-night-out fit "
                 "(v2P=favors 2P, xP=contradiction)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig_to_png(fig)
