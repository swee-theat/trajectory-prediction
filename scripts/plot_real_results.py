#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用第一阶段真实实验数据生成 5 张图表（论文/报告用）。

数据来源（均为本项目真实运行的产出）：
    - outputs/results/ablation_results.json         训练/验证曲线
    - outputs/results/table3_ablation.csv           消融实验 minADE
    - outputs/results/table5_param_scan.csv         参数规模-精度扫描
    - outputs/results/real_baseline/per_category_metrics.csv  分场景指标

输出：outputs/figures/fig1_train_loss.png ... fig5_category.png

用法: python scripts/plot_real_results.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 中文字体 ──────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 调色板（light mode，来自 dataviz 已验证默认调色板）──
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
       "#008300", "#4a3aa7", "#e34948"]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

OUT = "outputs/figures"


def _style_axes(ax):
    """统一坐标轴样式：浅色网格、下/左轴线、无多余边框。"""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def load_ablation_curves():
    with open("outputs/results/ablation_results.json", encoding="utf-8") as f:
        return json.load(f)


def fig1_train_loss():
    """图1：训练 Loss 收敛曲线（5 个代表性配置）。"""
    data = load_ablation_curves()
    picks = ["K_5多模态", "无上下文6维", "去不确定性", "2层MLP", "K_1单模态"]
    colors = [CAT[0], CAT[1], CAT[2], CAT[3], CAT[4]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, c in zip(picks, colors):
        for d in data:
            if d["name"] == name and "train_losses" in d:
                epochs = np.arange(1, len(d["train_losses"]) + 1)
                ax.plot(epochs, d["train_losses"], color=c, linewidth=2,
                        marker="o", markersize=4, markerfacecolor=c,
                        markeredgecolor="white", markeredgewidth=0.8, label=name)
                break
    _style_axes(ax)
    ax.set_xlabel("Epoch", fontsize=10, color=SEC)
    ax.set_ylabel("训练 Loss", fontsize=10, color=SEC)
    ax.set_title("训练 Loss 收敛曲线（5 个配置）", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_train_loss.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  ✅ {p}")


def fig2_val_minade():
    """图2：验证 minADE 收敛曲线（5 个代表性配置）。"""
    data = load_ablation_curves()
    picks = ["K_5多模态", "无上下文6维", "去不确定性", "2层MLP", "K_1单模态"]
    colors = [CAT[0], CAT[1], CAT[2], CAT[3], CAT[4]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, c in zip(picks, colors):
        for d in data:
            if d["name"] == name and "val_ades" in d:
                epochs = np.arange(1, len(d["val_ades"]) + 1)
                ax.plot(epochs, d["val_ades"], color=c, linewidth=2,
                        marker="o", markersize=4, markerfacecolor=c,
                        markeredgecolor="white", markeredgewidth=0.8, label=name)
                break
    _style_axes(ax)
    ax.set_xlabel("Epoch", fontsize=10, color=SEC)
    ax.set_ylabel("验证 minADE (m)", fontsize=10, color=SEC)
    ax.set_title("验证 minADE 收敛曲线（5 个配置）", fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_val_minade.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  ✅ {p}")


def fig3_ablation():
    """图3：消融实验 minADE 对比（横向条形，按 minADE 升序）。"""
    df = pd.read_csv("outputs/results/table3_ablation.csv", encoding="utf-8-sig")
    df = df.sort_values("minADE", ascending=True)
    names = df["实验条件"].tolist()
    vals = df["minADE"].astype(float).tolist()
    # 基线用蓝色高亮，其余用浅蓝；K_1 是 out-of-distribution 最差，仍如实展示
    colors = [BLUE if n == "完整模型(基线)" else "#9ec5f4" for n in names]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=colors, height=0.62)
    # 直接标注数值
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.4f}", va="center", ha="left",
                fontsize=8, color=INK)
    _style_axes(ax)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=INK)
    ax.set_xlabel("minADE (m)", fontsize=10, color=SEC)
    ax.set_title("消融实验 minADE 对比（越低越好）", fontsize=12, color=INK, pad=12)
    ax.set_xlim(0, max(vals) * 1.16)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_ablation.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  ✅ {p}")


def fig4_param_tradeoff():
    """图4：参数规模-精度权衡（hidden_dim 扫描，散点 + 参数量对数轴）。"""
    df = pd.read_csv("outputs/results/table5_param_scan.csv", encoding="utf-8-sig")
    params = df["参数量"].astype(str).str.replace(",", "").astype(float).tolist()
    ade = df["minADE(估)"].astype(float).tolist()
    hd = df["hidden_dim"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(params, ade, s=55, color=BLUE, edgecolor="white",
               linewidth=0.8, zorder=3)
    for p, a, h in zip(params, ade, hd):
        ax.annotate(f"hd={h}", (p, a), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=SEC)
    # 参数红线 288K
    ax.axvline(288000, color=ORANGE, linestyle="--", linewidth=1.4)
    ax.text(288000, min(ade), "  288K 红线", color=ORANGE, fontsize=8,
            va="bottom", ha="left", rotation=0)
    ax.set_xscale("log")
    _style_axes(ax)
    ax.set_xlabel("参数量（对数轴）", fontsize=10, color=SEC)
    ax.set_ylabel("minADE (m)", fontsize=10, color=SEC)
    ax.set_title("参数规模-精度权衡（hidden_dim 扫描）", fontsize=12, color=INK, pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_param_tradeoff.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  ✅ {p}")


def fig5_category():
    """图5：分场景 minADE（真实基线，5 类场景）。"""
    df = pd.read_csv("outputs/results/real_baseline/per_category_metrics.csv",
                     encoding="utf-8-sig")
    df = df.dropna(subset=["min_ade"])
    names = df.iloc[:, 0].tolist()
    vals = df["min_ade"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(names))
    ax.bar(x, vals, color=BLUE, width=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK)
    _style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, color=INK)
    ax.set_xlabel("场景类型", fontsize=10, color=SEC)
    ax.set_ylabel("minADE (m)", fontsize=10, color=SEC)
    ax.set_title("分场景 minADE（真实基线，42.7 万样本）", fontsize=12, color=INK, pad=12)
    ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_category.png")
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  ✅ {p}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("生成真实实验数据图表：")
    fig1_train_loss()
    fig2_val_minade()
    fig3_ablation()
    fig4_param_tradeoff()
    fig5_category()
    print("\n完成：5 张图已输出到 outputs/figures/")
