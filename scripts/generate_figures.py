"""综合图表生成：fig3(池化对比), fig5(Pareto), fig6(消融柱状图)。"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 加载数据 ──
with open("outputs/results/ablation_results.json", encoding="utf-8") as f:
    data = json.load(f)

# 基线 minADE
baseline = next(r for r in data if "基线" in r["name"])
BASELINE_ADE = baseline["best_val_min_ade"]
print(f"基线 minADE = {BASELINE_ADE:.4f}")

# ─────────────────────────────────────────────
# 图3: 池化方式对比
# ─────────────────────────────────────────────
def fig3_pooling():
    pooling = [r for r in data if "池化" in r["name"]]
    # 添加注意力池化（基线）
    pooling.insert(0, {"name": "注意力池化", "best_val_min_ade": BASELINE_ADE, "params": 145331})

    names = [r["name"].replace("池化", "") for r in pooling]
    ades = [r["best_val_min_ade"] for r in pooling]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    bars = ax.bar(names, ades, color=colors, width=0.45, edgecolor="white", linewidth=0.8)
    ax.axhline(y=BASELINE_ADE, color="red", linestyle="--", alpha=0.5, linewidth=1.2,
               label=f"注意力池化基线 ({BASELINE_ADE:.4f})")

    for bar, val in zip(bars, ades):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("minADE (m)", fontsize=12)
    ax.set_title("池化方式对预测精度的影响", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.72, max(ades) * 1.04)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_pooling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3_pooling.png 已保存")

# ─────────────────────────────────────────────
# 图5: 参数量-ADE Pareto 前沿
# ─────────────────────────────────────────────
def fig5_pareto():
    fig, ax = plt.subplots(figsize=(10, 6))

    # 本文模型（消融数据）
    our_names = []
    our_params = []
    our_ades = []
    for r in data:
        name = r["name"]
        our_names.append(name)
        our_params.append(r["params"])
        our_ades.append(r["best_val_min_ade"])

    ax.scatter(our_params, our_ades, c="#FF5722", s=120, zorder=5,
               edgecolors="white", linewidth=0.8, label="本文模型变体")
    for i, name in enumerate(our_names):
        offset = 10 if i % 2 == 0 else -15
        ax.annotate(name, (our_params[i], our_ades[i]),
                   textcoords="offset points", xytext=(8, offset),
                   fontsize=7, alpha=0.85)

    # SOTA 文献数据
    sota = {
        "CDDM (231K)": (231_000, 1.15),
        "VectorNet (~1M)": (1_000_000, 0.85),
        "HiVT (~800K)": (800_000, 0.70),
        "LaneGCN (~1.5M)": (1_500_000, 0.65),
    }
    s_x, s_y, s_names = [], [], []
    for name, (p, a) in sota.items():
        s_x.append(p)
        s_y.append(a)
        s_names.append(name)

    ax.scatter(s_x, s_y, c="#2196F3", s=120, marker="s", zorder=4,
               edgecolors="white", linewidth=0.8, label="SOTA方法")
    for x, y, name in zip(s_x, s_y, s_names):
        ax.annotate(name, (x, y), textcoords="offset points",
                   xytext=(10, 8), fontsize=8, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("参数量 (log scale)", fontsize=12)
    ax.set_ylabel("minADE (m)", fontsize=12)
    ax.set_title("参数效率 Pareto 前沿", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(8e4, 2e6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig5_pareto.png 已保存")

# ─────────────────────────────────────────────
# 图6: 消融实验柱状图
# ─────────────────────────────────────────────
def fig6_ablation():
    # 排除K=1（值太大破坏图表），K=5单独高亮
    main = [r for r in data if r["name"] != "K_1单模态"]
    names = [r["name"] for r in main]
    ades = [r["best_val_min_ade"] for r in main]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(names))
    colors = []
    for name in names:
        if "基线" in name:
            colors.append("#4CAF50")  # 基线绿色
        elif "K_5" in name:
            colors.append("#E91E63")  # 最优粉色
        elif "2层" in name or "K_1" in name:
            colors.append("#FF9800")  # 较差橙色
        else:
            colors.append("#2196F3")  # 普通蓝色

    bars = ax.bar(x, ades, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    # 基线水平线
    ax.axhline(y=BASELINE_ADE, color="#4CAF50", linestyle="--", alpha=0.6, linewidth=1.5,
               label=f"基线 minADE = {BASELINE_ADE:.4f}")

    # 数值标签
    for bar, val in zip(bars, ades):
        diff = val - BASELINE_ADE
        sign = "+" if diff > 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f} ({sign}{diff:.4f})", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("minADE (m)", fontsize=12)
    ax.set_title("消融实验: 各组件对预测精度的影响", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.60, max(ades) * 1.08)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig6_ablation.png 已保存")

    # 补充: K=1 单独图（值太大）
    k1 = next(r for r in data if "K_1" in r["name"])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    all_ades = [r["best_val_min_ade"] for r in data]
    ax2.bar(["K=3基线", "K=1单模态", "K=5多模态"],
            [BASELINE_ADE, k1["best_val_min_ade"],
             next(r["best_val_min_ade"] for r in data if "K_5" in r["name"])],
            color=["#4CAF50", "#F44336", "#E91E63"], width=0.5, edgecolor="white")
    ax2.set_ylabel("minADE (m)")
    ax2.set_title("模态数 K 的消融对比")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, val in enumerate([BASELINE_ADE, k1["best_val_min_ade"],
                              next(r["best_val_min_ade"] for r in data if "K_5" in r["name"])]):
        ax2.text(i, val + 0.02, f"{val:.4f}", ha="center", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "fig6_k_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  fig6_k_ablation.png 已保存")

# ── 执行 ──
if __name__ == "__main__":
    print("生成消融图表...\n")
    fig3_pooling()
    fig5_pareto()
    fig6_ablation()
    print(f"\n全部图表已保存到 {OUTPUT_DIR}/")
