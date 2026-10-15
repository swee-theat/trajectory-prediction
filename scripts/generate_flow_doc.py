#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「整体流程讲解 + 指导老师问答」.docx 文档。

覆盖训练线、Web Demo 使用线、ROS 实车部署线，以及面向指导老师的 8 个问答。
用法:
    python scripts/generate_flow_doc.py
输出:
    轻量化多模态轨迹预测系统_流程讲解与答辩问答.docx
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import datetime

doc = Document()

style = doc.styles["Normal"]
style.font.name = "微软雅黑"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.3


def add_code(text: str):
    """添加等宽字体代码/数据流块。"""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.left_indent = Pt(12)
    lines = text.strip("\n").split("\n")
    for i, line in enumerate(lines):
        run = p.add_run(("\n" if i > 0 else "") + line)
        run.font.name = "Consolas"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
        run.font.size = Pt(9.5)
    return p


def add_bullet(text: str):
    doc.add_paragraph(text, style="List Bullet")


# ═══ 封面 ═══
title = doc.add_heading("轻量化多模态轨迹预测系统", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("整体流程讲解 · 面向指导老师的问答")
r.font.size = Pt(16)
r.font.color.rgb = RGBColor(0x4F, 0x46, 0xE5)
m = doc.add_paragraph()
m.alignment = WD_ALIGN_PARAGRAPH.CENTER
m.add_run(f"文档版本: 1.0 | 日期: {datetime.date.today().strftime('%Y-%m-%d')}").font.size = Pt(10)
doc.add_page_break()

# ═══ 一、系统全景 ═══
doc.add_heading("一、系统全景", level=1)
doc.add_paragraph(
    "本系统的核心任务一句话概括：给模型看车过去 2 秒的轨迹（20 个点），"
    "让它猜出未来 3 秒会怎么走（30 个点），并且不止猜一条，而是同时猜 5 条不同的走法（多模态），"
    "还告诉每条走法它自己有多没把握（不确定性）。"
)
doc.add_heading("1.1 三个使用入口", level=2)
t = doc.add_table(rows=4, cols=3, style="Light Grid Accent 1")
for i, h in enumerate(["入口", "触发方式", "关键文件"]):
    t.rows[0].cells[i].text = h
for i, (a, b, c) in enumerate([
    ("训练", "命令行 python scripts/train_with_uncertainty.py", "train_with_uncertainty.py"),
    ("网页 Demo", "浏览器打开 http://localhost:5000", "demo/app.py + demo/templates/index.html"),
    ("实车部署", "小车上运行 ROS 节点", "deploy/ros_node/trajectory_predictor.py"),
]):
    t.rows[i + 1].cells[0].text = a
    t.rows[i + 1].cells[1].text = b
    t.rows[i + 1].cells[2].text = c

doc.add_heading("1.2 核心数据流（所有入口共用）", level=2)
add_code(
    "20个(x,y)点  →  11维特征[20,11]  →  编码成1个向量[128]  →  展开成\n"
    "5条轨迹[5,30,2] + 5个概率 + 5条方差"
)

doc.add_page_break()

# ═══ 二、训练线 ═══
doc.add_heading("二、训练线：从原始数据到能用的模型", level=1)

doc.add_heading("第 1 步：原始数据长什么样", level=2)
doc.add_paragraph(
    "数据来自 Argoverse（自动驾驶开源数据集），原始是 CSV 文件，"
    "每行是一个「某智能体在某时刻的坐标」，0.1 秒一帧（10Hz）。"
)
add_code(
    "timestamp, track_id, object_type, x, y\n"
    "0.0,  1, AGENT, 102.5, 88.3\n"
    "0.1,  1, AGENT, 102.7, 88.4   <- 0.1 秒一帧(10Hz)"
)

doc.add_heading("第 2 步：预处理——把原始坐标变成 11 维特征", level=2)
doc.add_paragraph("文件：data/preprocessing.py，入口函数 process_dataset() → extract_agent_track()。这一步是整个系统的地基，干三件事：")
add_bullet("切窗口：切成长轨迹「前 20 帧 = 观测（输入），后 30 帧 = 未来（真值）」")
add_bullet("居中：坐标整体平移，让最后一帧观测点变成原点 (0,0)，模型学到的是「相对位移」")
add_bullet("算 11 维特征（下表）")
t = doc.add_table(rows=12, cols=3, style="Light Grid Accent 1")
for i, h in enumerate(["索引", "特征", "物理含义"]):
    t.rows[0].cells[i].text = h
for i, (a, b, c) in enumerate([
    ("0-1", "x, y", "车现在在哪（相对原点）"),
    ("2-3", "vx, vy", "车往哪个方向、多快（相邻帧差分/0.1s）"),
    ("4", "t/T", "现在是第几帧（0→1）"),
    ("5", "1", "常数项，给线性层一个偏置"),
    ("6-8", "dist/100 等", "离最近的其他车多远（社交特征，小车单机时填 0）"),
    ("9", "heading", "航向角，车头朝哪（arctan2(vy,vx)）"),
    ("10", "curvature", "曲率，转弯多急（航向角变化率）"),
]):
    t.rows[i + 1].cells[0].text = a
    t.rows[i + 1].cells[1].text = b
    t.rows[i + 1].cells[2].text = c
doc.add_paragraph("最后存成压缩文件 data/processed/train.npz，形状 [样本数, 20, 11]。")

doc.add_heading("第 3 步：加载数据成 batch", level=2)
doc.add_paragraph(
    "data/dataset.py 的 __getitem__ 从 .npz 取一条样本转成张量；"
    "data/dataloader.py 的 create_dataloader 把 64 条拼成一批，形状 [20,11] → [64,20,11]。"
)

doc.add_heading("第 4 步：训练主入口——两阶段训练", level=2)
doc.add_paragraph("文件：scripts/train_with_uncertainty.py，把训练拆成两阶段：")
add_bullet("Stage A（正常训练）：lambda_unc=0，只训练「轨迹预测 + 模态选择」")
add_bullet("Stage B（不确定性微调）：lambda_unc=0.05，冻结编码器和预测器，只微调不确定性头")
doc.add_paragraph("freeze_except_uncertainty() 把除 uncertainty 开头的参数全部 requires_grad=False，实现「冻结」。")

doc.add_heading("第 5 步：模型数据流", level=2)
doc.add_paragraph("文件：models/trajectory_model.py（总装），拆三层。")
doc.add_paragraph("① 编码器 models/trajectory_encoder.py —— 把 20 帧压成 1 个向量：")
add_code(
    "[64, 20, 11] 特征\n"
    "  → feature_proj 线性层        [64, 20, 128]   (11维→128维)\n"
    "  → 3 层 MLPBlock              [64, 20, 128]   (每帧独立处理，不混时间)\n"
    "  → temporal_mixer 时序卷积     [64, 20, 128]   (让每帧看到前后帧)\n"
    "  → 注意力池化                  [64, 128]       (20帧加权求和→1个向量)"
)
doc.add_paragraph("注意力池化（attention_pooling.py）学一个 query 向量，让模型自己决定 20 帧里哪几帧最重要，参数仅 128 个。")
doc.add_paragraph("② 预测器 models/multimodal_predictor.py —— 从 1 个向量变出 5 条轨迹：")
add_code(
    "[64, 128] 向量\n"
    "  ├→ 模态选择器 mode_selector   [64, 5]        (5个概率，softmax后和为1)\n"
    "  └→ 5 个独立轨迹分支           [64, 5, 30, 2] (5条轨迹，每条30个点)"
)
doc.add_paragraph("③ 不确定性 models/uncertainty_estimator.py —— 每条轨迹配一个方差，Softplus 保证方差永远为正，独立头让 5 条轨迹各有各的方差。")

doc.add_heading("第 6 步：损失函数——教模型怎么算「错多少」", level=2)
doc.add_paragraph("文件：train/losses.py。总损失：")
add_code("总损失 = 0.7×轨迹损失 + 0.3×模态损失 + 0.05×不确定性损失")
doc.add_paragraph(
    "轨迹损失用 WTA（赢者通吃）：5 条轨迹里只挑离真值最近的那条惩罚，"
    "其余 4 条不管，逼 5 条轨迹各奔东西。不确定性损失用高斯负对数似然 NLL，逼模型「说实话」。"
)

doc.add_heading("第 7 步：训练循环 + 保存", level=2)
doc.add_paragraph(
    "文件：train/trainer.py。每个 epoch：train_epoch()（前向→损失→反向→更新，数据增强在此）"
    "→ validate()（验证集算 minADE，不更新参数）→ 用监控指标判断是否保存 best_model。"
)
doc.add_paragraph(
    "最终产物 outputs/checkpoints/unc_stage_b/best_model.pt，一个装着全部权重的 .pt 文件。"
)

doc.add_page_break()

# ═══ 三、使用线 Web Demo ═══
doc.add_heading("三、使用线（Web Demo）：从点击到看到 5 条轨迹", level=1)

doc.add_heading("第 1 步：启动后端", level=2)
doc.add_paragraph("python demo/app.py 启动 Flask，app.py 启动时先把训练好的模型加载进内存并 model.eval()，之后每次预测不重新加载。")

doc.add_heading("第 2 步：用户画轨迹", level=2)
doc.add_paragraph("用户在 Canvas 点 20 个点，或点「弯道示例」按钮 → loadExample() 从 /api/v1/examples 拿一段圆弧坐标填进画布。")

doc.add_heading("第 3 步：点「开始预测」发请求", level=2)
doc.add_paragraph("前端 runPrediction() 把 20 个点打包 JSON，POST 到 /api/v1/predict。")

doc.add_heading("第 4 步：后端算特征 + 推理", level=2)
doc.add_paragraph("app.py predict() → run_inference()，与训练预处理严格一致：居中 → 算 11 维特征 → model(input_tensor) → 还原绝对坐标。")

doc.add_heading("第 5 步：数据在模型里流一遍", level=2)
add_code(
    "[1, 20, 11] 特征(1个样本)\n"
    "  → 编码器     [1, 128]\n"
    "  → 预测器     [1, 5, 30, 2] 轨迹 + [1, 5] 概率\n"
    "  → 不确定性   [1, 5, 30, 2] 方差"
)

doc.add_heading("第 6 步：后端打包 JSON 返回", level=2)
doc.add_paragraph("app.py 把轨迹、概率、方差、统计信息塞进 JSON 返回给前端。")

doc.add_heading("第 7 步：前端画图——最终页面", level=2)
doc.add_paragraph("index.html draw() 用 Canvas 画出最终结果：")
add_bullet("蓝色实线 = 观测轨迹（20 个点）")
add_bullet("5 条彩色虚线 = 5 条预测轨迹，按概率排序，终点标数字 1-5")
add_bullet("半透明椭圆 = 不确定性（方差越大椭圆越大，表示越没把握）")
add_bullet("右侧面板 = 模态概率条 + 各模态详情表")

doc.add_page_break()

# ═══ 四、使用线 ROS 实车 ═══
doc.add_heading("四、使用线（ROS 实车）：小车实时预测", level=1)
doc.add_paragraph("文件：deploy/ros_node/trajectory_predictor.py，与 Web Demo 思路一致，输入换成小车里程计 /odom。")
for s in [
    "1. 订阅 /odom（20Hz 位姿）→ _odom_callback()",
    "2. 降采样到 10Hz（训练数据是 10Hz，odom 20Hz 不降采样则速度和曲率被高估 2 倍）",
    "3. 攒满 20 帧 → _extract_features() 算 11 维特征（与训练完全一致）",
    "4. ONNX 推理 → 5 条轨迹 + 概率",
    "5. _publish_markers() 转成 MarkerArray 发到 RViz，车上屏幕显示 5 条彩色预测线",
]:
    doc.add_paragraph(s)

doc.add_page_break()

# ═══ 五、指导老师问答 ═══
doc.add_heading("五、站在指导老师角度：可能的问题 & 回答", level=1)

qa = [
    (
        "Q1. 为什么预测 5 条轨迹，而不是只给 1 条「最准」的？",
        "因为车辆未来是「多意图」的——到路口可能直行、左转、右转。单条轨迹只能猜一个，猜错就全错。"
        "多模态（K=5）+ 模态概率让模型同时给出 5 种走法，mode_selector 给出每条概率。"
        "训练用 WTA 损失只惩罚最接近真值的那条，逼 5 条轨迹各奔东西覆盖不同可能。"
        "评价用 minADE（5 条里挑最好的算误差）。",
    ),
    (
        "Q2. 11 维特征是怎么设计的？有什么依据？",
        "11 维 = 位置(2) + 速度(2) + 归一化时间(1) + 偏置(1) + 社交距离(3) + 航向角(1) + 曲率(1)。"
        "前 9 维是基础运动学量，后两维是改进核心：早期 9 维只能从 (vx,vy) 间接猜方向，导致「弯道走直线」。"
        "显式加入航向角和曲率后模型有了明确方向和转弯信息。坐标先以最后观测帧为中心归一化，消除绝对位置影响。",
    ),
    (
        "Q3. 为什么要两阶段训练？不确定性头为什么用独立头而非共享头？",
        "两阶段各司其职：Stage A 专注把轨迹和模态学对（lambda_unc=0）；"
        "Stage B 冻结主网络只微调不确定性头（lambda_unc=0.05），避免为优化方差而牺牲轨迹精度。"
        "独立头（share_across_modes=False）的意义：共享头会让 5 条轨迹方差完全相同，无法区分哪条更确定；"
        "独立头让直行（确定，方差小）和掉头（没把握，方差大）有各自置信度，不确定性才真正有意义。",
    ),
    (
        "Q4. 模型只有约 244K 参数，凭什么说「轻量化」？",
        "对比基于 Transformer 或图网络的轨迹预测模型（动辄数百万到数千万参数），244K 小了两个数量级。"
        "轻量化体现在三处：① MLP 逐帧编码，复杂度 O(N×d²) 而非注意力机制的 O(N²×d)；"
        "② 时序卷积用 depthwise，只加约 384 参数就获得时序交互；③ 注意力池化只学 1 个 query 向量（约 128 参数）。"
        "目标是在 RTX 3060（6GB）和实体小车 NUC 上都能跑。",
    ),
    (
        "Q5. 从训练到部署，怎么保证结果不「跑偏」？",
        "靠三处一致性对齐：① 特征一致——compute_features() 和 _extract_features() 与训练预处理的居中、差分、dt=0.1、"
        "曲率裁剪完全一致；② 频率一致——训练数据 10Hz，ROS 里 odom 20Hz，专门降采样，否则 20 帧只覆盖 1 秒；"
        "③ 部署格式一致——训练是 PyTorch，部署导出 ONNX 用 onnxruntime 推理，输入输出名对齐。",
    ),
    (
        "Q6. 数据增强里为什么要「同步更新航向角」？",
        "增强会随机旋转轨迹坐标（模拟不同朝向样本）。旋转了坐标，速度分量 (vx,vy) 也要跟着转，"
        "但航向角特征（索引 9）若不变就会和旋转后的速度方向矛盾——模型看到「速度朝东但航向角朝北」的脏数据。"
        "所以让航向角也加上旋转角。曲率是航向角的差分，旋转不改变它，无需更新。"
        "旋转范围也从 ±180° 收紧到 ±30°，避免出现「倒着开」的不合理样本。",
    ),
    (
        "Q7. 用什么指标评价预测好不好？",
        "核心指标在 eval/metrics.py：ADE（平均位移误差，整条轨迹逐点误差平均）、FDE（终点误差）、"
        "minADE/minFDE（5 条里挑最好算误差，体现多模态 oracle 能力）、MR（未命中率）、终点命中率。"
        "多模态场景下 minADE 最关键，反映「只要覆盖到正确走法，误差能多小」。",
    ),
    (
        "Q8. 这套系统的创新点/工作量体现在哪？",
        "四点：① 轻量多模态——极简 MLP+注意力+WTA 实现多模态预测，244K 参数跑出可用 minADE；"
        "② 物理可解释特征工程——11 维特征各有明确运动学/社交含义，航向角+曲率显著改善弯道表现；"
        "③ 不确定性量化——两阶段训练+独立头输出每条轨迹方差，可画置信椭圆；"
        "④ 完整落地链路——从训练到 Web Demo 到 ROS 实车端到端跑通，并解决频率、特征、格式三类一致性坑。",
    ),
]

for q, a in qa:
    doc.add_heading(q, level=2)
    p = doc.add_paragraph()
    run = p.add_run("答：")
    run.bold = True
    p.add_run(a)

output_path = "轻量化多模态轨迹预测系统_流程讲解与答辩问答.docx"
doc.save(output_path)
print(f"文档已保存: {output_path}")
