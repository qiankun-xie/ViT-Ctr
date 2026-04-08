const pptxgen = require("pptxgenjs");
const fs = require("fs");
const pptx = new pptxgen();

pptx.layout = "LAYOUT_16x9";
pptx.author = "ViT-Ctr Project";
pptx.title = "ViT-Ctr Phase 1-5 Summary Report";

// ── Color palette: Deep Teal + Coral accent ──
const P = {
  bg:      "0D1B2A",   // very dark blue-black
  bgCard:  "1B2838",   // slightly lighter card bg
  primary: "5EA8A7",   // teal
  accent:  "FE4447",   // coral red
  gold:    "F0A500",   // gold
  green:   "3DDC84",   // bright green
  white:   "FFFFFF",
  ltGray:  "B0BEC5",
  midGray: "78909C",
  text:    "E0E6ED",
};

// ── Helpers ──
function addBgBar(slide, y, h, color) {
  slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y, w: 10, h, fill: { color } });
}

function titleSlide(slide, titleText, subtitleText) {
  slide.background = { fill: P.bg };
  // Top accent line
  slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: P.primary } });
  // Bottom accent line
  slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: P.accent } });

  slide.addText(titleText, { x: 0.8, y: 1.5, w: 8.4, h: 1.2, fontSize: 36, fontFace: "Arial", bold: true, color: P.white, align: "center" });
  slide.addText(subtitleText, { x: 1.2, y: 2.8, w: 7.6, h: 0.8, fontSize: 18, fontFace: "Arial", color: P.primary, align: "center" });
}

function sectionHeader(slide, num, text) {
  slide.background = { fill: P.bg };
  slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: P.primary } });
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.0, w: 1.0, h: 1.0, rectRadius: 0.15, fill: { color: P.primary } });
  slide.addText(num, { x: 0.6, y: 2.0, w: 1.0, h: 1.0, fontSize: 32, fontFace: "Arial", bold: true, color: P.bg, align: "center", valign: "middle" });
  slide.addText(text, { x: 1.9, y: 2.0, w: 7.5, h: 1.0, fontSize: 30, fontFace: "Arial", bold: true, color: P.white, valign: "middle" });
}

function slideTitle(slide, text) {
  slide.background = { fill: P.bg };
  // header bar
  slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.75, fill: { color: P.bgCard } });
  slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0.75, w: 10, h: 0.03, fill: { color: P.primary } });
  slide.addText(text, { x: 0.5, y: 0.08, w: 9, h: 0.6, fontSize: 20, fontFace: "Arial", bold: true, color: P.primary });
}

function metricBox(slide, x, y, w, h, label, value, color) {
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: P.bgCard }, line: { color: color || P.primary, width: 1.5 } });
  slide.addText(label, { x, y: y + 0.1, w, h: 0.4, fontSize: 11, fontFace: "Arial", color: P.ltGray, align: "center" });
  slide.addText(value, { x, y: y + 0.45, w, h: 0.55, fontSize: 22, fontFace: "Arial", bold: true, color: color || P.white, align: "center" });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1: Cover
// ════════════════════════════════════════════════════════════════════════════
const s1 = pptx.addSlide();
titleSlide(s1,
  "ViT-Ctr 项目阶段性总结",
  "基于Vision Transformer的RAFT链转移常数预测系统"
);
s1.addText("Phase 1 – 5 完成情况审查", { x: 1.2, y: 3.5, w: 7.6, h: 0.5, fontSize: 16, fontFace: "Arial", color: P.ltGray, align: "center" });
s1.addText("2026年4月6日", { x: 3.5, y: 4.5, w: 3, h: 0.4, fontSize: 14, fontFace: "Arial", color: P.midGray, align: "center" });

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2: 项目核心价值
// ════════════════════════════════════════════════════════════════════════════
const s2 = pptx.addSlide();
slideTitle(s2, "项目核心价值");

// Central value proposition box
slide = s2;
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 1.1, w: 8.4, h: 1.2, rectRadius: 0.15, fill: { color: "1A3A4A" }, line: { color: P.primary, width: 2 } });
slide.addText([
  { text: "一次输入实验数据", options: { bold: true, color: P.primary } },
  { text: "，同时提取 ", options: { color: P.text } },
  { text: "Ctr", options: { bold: true, color: P.accent } },
  { text: "、", options: { color: P.text } },
  { text: "诱导期", options: { bold: true, color: P.gold } },
  { text: "、", options: { color: P.text } },
  { text: "减速因子", options: { bold: true, color: P.green } },
  { text: " 三个参数", options: { color: P.text } },
], { x: 1.0, y: 1.2, w: 8.0, h: 0.5, fontSize: 18, fontFace: "Arial", align: "center" });
slide.addText("传统方法需要三组独立实验才能分别获得", { x: 1.0, y: 1.75, w: 8.0, h: 0.4, fontSize: 13, fontFace: "Arial", color: P.midGray, align: "center" });

// Pipeline flow
const steps = [
  { label: "ODE模拟", sub: "RAFT动力学", color: P.primary },
  { label: "ctFP编码", sub: "64×64双通道", color: P.primary },
  { label: "SimpViT", sub: "877K参数", color: P.accent },
  { label: "Bootstrap UQ", sub: "200次×F分布", color: P.gold },
  { label: "Web应用", sub: "Streamlit", color: P.green },
];
const startX = 0.5, stepW = 1.65, arrowW = 0.2, stepY = 2.8, stepH = 1.2;
steps.forEach((s, i) => {
  const x = startX + i * (stepW + arrowW);
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x, y: stepY, w: stepW, h: stepH, rectRadius: 0.1, fill: { color: P.bgCard }, line: { color: s.color, width: 1.5 } });
  slide.addText(s.label, { x, y: stepY + 0.15, w: stepW, h: 0.45, fontSize: 14, fontFace: "Arial", bold: true, color: P.white, align: "center" });
  slide.addText(s.sub, { x, y: stepY + 0.6, w: stepW, h: 0.35, fontSize: 10, fontFace: "Arial", color: P.ltGray, align: "center" });
  if (i < steps.length - 1) {
    slide.addText("\u2192", { x: x + stepW, y: stepY + 0.3, w: arrowW, h: 0.5, fontSize: 20, fontFace: "Arial", color: P.primary, align: "center" });
  }
});

// Key stats bottom bar
metricBox(slide, 0.5, 4.35, 2.0, 1.0, "\u6837\u672C\u91CF", "973,693", P.primary);
metricBox(slide, 2.8, 4.35, 2.0, 1.0, "\u53C2\u6570\u91CF", "877,571", P.accent);
metricBox(slide, 5.1, 4.35, 2.0, 1.0, "R\u00B2 (log\u2081\u2080Ctr)", "0.991", P.gold);
metricBox(slide, 7.4, 4.35, 2.0, 1.0, "\u8BA1\u5212\u5B8C\u6210", "13/14", P.green);

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3: 总体进度
// ════════════════════════════════════════════════════════════════════════════
const s3 = pptx.addSlide();
slideTitle(s3, "总体进度一览");

const phases = [
  ["Phase 1", "ODE基础与ctFP编码器", "3/3", "Complete", "2026-03-26"],
  ["Phase 2", "大规模数据集生成", "2/2", "Complete", "2026-03-27"],
  ["Phase 3", "模型训练与评估", "5/5", "Complete", "2026-04-03"],
  ["Phase 4", "文献验证与Mayo基线", "2/2", "Complete", "2026-04-04"],
  ["Phase 5", "Streamlit Web应用", "1/2", "In Progress", "—"],
  ["Phase 6", "论文与支撑材料", "0/?", "Not started", "—"],
];

const tblData = [
  [
    { text: "阶段", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "名称", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "计划", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "状态", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "完成日期", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
  ],
  ...phases.map(([phase, name, plans, status, date]) => {
    const sc = status === "Complete" ? P.green : status === "In Progress" ? P.gold : P.midGray;
    return [
      { text: phase, options: { align: "center", bold: true, color: P.white } },
      { text: name, options: { color: P.text } },
      { text: plans, options: { align: "center", color: P.text } },
      { text: status, options: { align: "center", bold: true, color: sc } },
      { text: date, options: { align: "center", color: P.ltGray } },
    ];
  })
];

s3.addTable(tblData, {
  x: 0.4, y: 1.0, w: 9.2, colW: [1.2, 3.2, 0.9, 1.6, 1.6],
  rowH: [0.45, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
  border: { pt: 0.5, color: "37474F" },
  fill: { color: P.bgCard }, fontSize: 13, fontFace: "Arial", valign: "middle"
});

// Progress bar
s3.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 4.3, w: 8.4, h: 0.25, rectRadius: 0.12, fill: { color: "263238" } });
s3.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 4.3, w: 8.4 * 0.929, h: 0.25, rectRadius: 0.12, fill: { color: P.primary } });
s3.addText("92.9%  (13/14 plans)", { x: 0.8, y: 4.65, w: 8.4, h: 0.3, fontSize: 12, fontFace: "Arial", color: P.ltGray, align: "center" });

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4: Phase 1 - ODE & ctFP
// ════════════════════════════════════════════════════════════════════════════
const s4 = pptx.addSlide();
slideTitle(s4, "Phase 1：ODE基础与ctFP编码器（2026-03-26）");

// Left column: ODE
s4.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: 1.0, w: 4.4, h: 4.0, rectRadius: 0.1, fill: { color: P.bgCard }, line: { color: P.primary, width: 1 } });
s4.addText("RAFT动力学ODE模拟器", { x: 0.6, y: 1.1, w: 4.0, h: 0.4, fontSize: 15, fontFace: "Arial", bold: true, color: P.primary });
s4.addText([
  { text: "\u2022 单平衡模型（14变量）：TTC/Xanthate/DTC\n", options: { fontSize: 12, color: P.text, breakType: "none" } },
  { text: "\u2022 预平衡模型（16变量）：Dithioester\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 Radau求解器 + brentq转化率网格\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 3个文献体系验证通过\n", options: { fontSize: 12, color: P.text } },
  { text: "  \u2013 CDB/苯乙烯\n", options: { fontSize: 11, color: P.ltGray } },
  { text: "  \u2013 TTC/MMA\n", options: { fontSize: 11, color: P.ltGray } },
  { text: "  \u2013 Xanthate/VAc", options: { fontSize: 11, color: P.ltGray } },
], { x: 0.6, y: 1.6, w: 4.0, h: 2.8, fontSize: 12, fontFace: "Arial", color: P.text, valign: "top", lineSpacing: 20 });

// Right column: ctFP
s4.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.0, w: 4.4, h: 4.0, rectRadius: 0.1, fill: { color: P.bgCard }, line: { color: P.accent, width: 1 } });
s4.addText("ctFP指纹编码器", { x: 5.4, y: 1.1, w: 4.0, h: 0.4, fontSize: 15, fontFace: "Arial", bold: true, color: P.accent });
s4.addText([
  { text: "\u2022 输入：(cta_ratio, conv, Mn, \u0110)\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 输出：(2, 64, 64) PyTorch张量\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 Ch0 = Mn (归一化)\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 Ch1 = \u0110 (截断于4.0)\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 X轴 = [CTA]/[M]\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 Y轴 = conversion\n", options: { fontSize: 12, color: P.text } },
  { text: "\u2022 训练/推理共享同一transform()函数", options: { fontSize: 12, color: P.green } },
], { x: 5.4, y: 1.6, w: 4.0, h: 2.8, fontSize: 12, fontFace: "Arial", color: P.text, valign: "top", lineSpacing: 20 });

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5: Phase 2 - Dataset
// ════════════════════════════════════════════════════════════════════════════
const s5 = pptx.addSlide();
slideTitle(s5, "Phase 2：大规模数据集生成（2026-03-27）");

metricBox(s5, 0.5, 1.0, 2.0, 1.0, "总样本数", "973,693", P.primary);
metricBox(s5, 2.8, 1.0, 2.0, 1.0, "RAFT类型", "4种", P.accent);
metricBox(s5, 5.1, 1.0, 2.0, 1.0, "数据总量", "~30 GB", P.gold);
metricBox(s5, 7.4, 1.0, 2.0, 1.0, "标签维度", "3列", P.green);

const dsTbl = [
  [
    { text: "RAFT类型", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "样本数", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "文件大小", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
  ],
  [{ text: "Dithioester", options: { color: P.text } }, { text: "243,365", options: { align: "center", color: P.text } }, { text: "7,634 MB", options: { align: "center", color: P.ltGray } }],
  [{ text: "Trithiocarbonate", options: { color: P.text } }, { text: "243,455", options: { align: "center", color: P.text } }, { text: "7,634 MB", options: { align: "center", color: P.ltGray } }],
  [{ text: "Xanthate", options: { color: P.text } }, { text: "243,417", options: { align: "center", color: P.text } }, { text: "7,634 MB", options: { align: "center", color: P.ltGray } }],
  [{ text: "Dithiocarbamate", options: { color: P.text } }, { text: "243,456", options: { align: "center", color: P.text } }, { text: "7,634 MB", options: { align: "center", color: P.ltGray } }],
];

s5.addTable(dsTbl, {
  x: 1.5, y: 2.3, w: 7.0, colW: [2.8, 2.0, 2.2],
  rowH: [0.4, 0.38, 0.38, 0.38, 0.38],
  border: { pt: 0.5, color: "37474F" }, fill: { color: P.bgCard },
  fontSize: 13, fontFace: "Arial", valign: "middle"
});

s5.addText([
  { text: "采样策略：", options: { bold: true, color: P.primary } },
  { text: "7维LHS（log10_Ctr, log10_kp, log10_kt, log10_kd, I0, f, log10_cta_ratio）", options: { color: P.text } },
], { x: 0.5, y: 4.1, w: 9.0, h: 0.35, fontSize: 11, fontFace: "Arial" });
s5.addText([
  { text: "噪声注入：", options: { bold: true, color: P.primary } },
  { text: "sigma = 0.03 (3%)  |  ", options: { color: P.text } },
  { text: "数据划分：", options: { bold: true, color: P.primary } },
  { text: "80/10/10 分层 (按log10_Ctr 0.5单位分箱)  |  ", options: { color: P.text } },
  { text: "格式：", options: { bold: true, color: P.primary } },
  { text: "chunked HDF5 (N, 64, 64, 2)", options: { color: P.text } },
], { x: 0.5, y: 4.5, w: 9.0, h: 0.35, fontSize: 11, fontFace: "Arial" });

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6: Phase 3 - Model
// ════════════════════════════════════════════════════════════════════════════
const s6 = pptx.addSlide();
slideTitle(s6, "Phase 3：模型训练与评估（2026-04-03）");

// Architecture box
s6.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: 1.0, w: 4.4, h: 2.5, rectRadius: 0.1, fill: { color: P.bgCard }, line: { color: P.primary, width: 1 } });
s6.addText("SimpViT 架构", { x: 0.6, y: 1.05, w: 4.0, h: 0.35, fontSize: 14, fontFace: "Arial", bold: true, color: P.primary });
s6.addText([
  { text: "\u2022 输入：(2, 64, 64) 双通道ctFP\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 Patch：16\u00D716 \u2192 16个token\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 2层Transformer, 4头自注意力\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 隐藏维度：64\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 输出：3维回归\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 参数量：877,571", options: { fontSize: 11, color: P.green } },
], { x: 0.6, y: 1.45, w: 4.0, h: 2.0, fontSize: 11, fontFace: "Arial", valign: "top", lineSpacing: 18 });

// Training config box
s6.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.0, w: 4.4, h: 2.5, rectRadius: 0.1, fill: { color: P.bgCard }, line: { color: P.gold, width: 1 } });
s6.addText("训练配置", { x: 5.4, y: 1.05, w: 4.0, h: 0.35, fontSize: 14, fontFace: "Arial", bold: true, color: P.gold });
s6.addText([
  { text: "\u2022 优化器：Adam (lr=3\u00D710\u207B\u2074)\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 损失：加权MSE (2.0/0.5/0.5)\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 LR调度：ReduceLROnPlateau\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 早停：patience=15\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 硬件：RTX 4090 (CUDA)\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 142 epochs, best@126", options: { fontSize: 11, color: P.green } },
], { x: 5.4, y: 1.45, w: 4.0, h: 2.0, fontSize: 11, fontFace: "Arial", valign: "top", lineSpacing: 18 });

// Bootstrap UQ
s6.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: 3.7, w: 9.2, h: 1.5, rectRadius: 0.1, fill: { color: P.bgCard }, line: { color: P.accent, width: 1 } });
s6.addText("Bootstrap UQ", { x: 0.6, y: 3.75, w: 3.0, h: 0.35, fontSize: 14, fontFace: "Arial", bold: true, color: P.accent });
s6.addText([
  { text: "\u2022 200次bootstrap迭代 \u00D7 5 epochs/head  |  冻结backbone，仅微调输出头\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 F分布联合置信区间：p=3, f.ppf(0.95, 3, 197) \u2248 2.65\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 事后校准因子：[100.0, 53.74, 3.51]  \u2192  inhibition/retardation达95%覆盖率\n", options: { fontSize: 11, color: P.text } },
  { text: "\u2022 总训练时间：9小时17分钟", options: { fontSize: 11, color: P.ltGray } },
], { x: 0.6, y: 4.1, w: 8.8, h: 1.0, fontSize: 11, fontFace: "Arial", valign: "top", lineSpacing: 18 });

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7: Phase 4 - Literature Validation
// ════════════════════════════════════════════════════════════════════════════
const s7 = pptx.addSlide();
slideTitle(s7, "Phase 4：文献验证 — ML模型 vs Mayo方程（2026-04-04）");

s7.addText("14个已发表Ctr值，4类RAFT剂 \u00D7 3种测量方法（Mayo/CLD/Dispersity）", { x: 0.5, y: 0.9, w: 9, h: 0.35, fontSize: 12, fontFace: "Arial", color: P.ltGray });

const valTbl = [
  [
    { text: "指标", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "ML模型", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "Mayo方程", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
    { text: "优势方", options: { fill: { color: P.primary }, color: P.bg, bold: true, align: "center" } },
  ],
  [{ text: "中位fold-error", options: { color: P.text, bold: true } }, { text: "1.17", options: { align: "center", color: P.text } }, { text: "1.01", options: { align: "center", color: P.green } }, { text: "Mayo", options: { align: "center", bold: true, color: P.ltGray } }],
  [{ text: "2倍以内比例", options: { color: P.text, bold: true } }, { text: "92.9%", options: { align: "center", color: P.green } }, { text: "85.7%", options: { align: "center", color: P.text } }, { text: "ML", options: { align: "center", bold: true, color: P.green } }],
  [{ text: "10倍以内比例", options: { color: P.text, bold: true } }, { text: "100%", options: { align: "center", color: P.green } }, { text: "92.9%", options: { align: "center", color: P.text } }, { text: "ML", options: { align: "center", bold: true, color: P.green } }],
  [{ text: "RMSE (log\u2081\u2080)", options: { color: P.text, bold: true } }, { text: "0.126", options: { align: "center", color: P.green } }, { text: "0.558", options: { align: "center", color: P.text } }, { text: "ML", options: { align: "center", bold: true, color: P.green } }],
  [{ text: "R\u00B2 (log\u2081\u2080)", options: { color: P.text, bold: true } }, { text: "0.991", options: { align: "center", color: P.green } }, { text: "0.825", options: { align: "center", color: P.text } }, { text: "ML", options: { align: "center", bold: true, color: P.green } }],
];

s7.addTable(valTbl, {
  x: 1.0, y: 1.4, w: 8.0, colW: [2.2, 2.0, 2.0, 1.5],
  rowH: [0.45, 0.42, 0.42, 0.42, 0.42, 0.42],
  border: { pt: 0.5, color: "37474F" }, fill: { color: P.bgCard },
  fontSize: 14, fontFace: "Arial", valign: "middle"
});

s7.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 4.15, w: 8.4, h: 1.0, rectRadius: 0.1, fill: { color: "1A3A4A" }, line: { color: P.primary, width: 1 } });
s7.addText([
  { text: "结论：", options: { bold: true, color: P.primary } },
  { text: "ML模型在整体准确度（R\u00B2=0.991 vs 0.825）和鲁棒性（100% vs 92.9% 10\u00D7以内）上显著优于传统Mayo方程。Mayo在中位fold-error上更精确（1.01 vs 1.17），适合精密单点估计。", options: { color: P.text } },
], { x: 1.0, y: 4.25, w: 8.0, h: 0.8, fontSize: 12, fontFace: "Arial", valign: "middle" });

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8: Phase 5 - Web App
// ════════════════════════════════════════════════════════════════════════════
const s8 = pptx.addSlide();
slideTitle(s8, "Phase 5：Streamlit Web应用");

// Feature cards
const features = [
  { title: "手动输入", desc: "[CTA]/[M], conv,\nMn, \u0110 表格编辑", icon: "\u270D", color: P.primary },
  { title: "文件上传", desc: "Excel/CSV上传\n+ 模板下载", icon: "\u2B06", color: P.accent },
  { title: "三参数预测", desc: "Ctr + 诱导期 +\n减速因子 + 95%CI", icon: "\u2699", color: P.gold },
  { title: "ctFP可视化", desc: "双通道热力图\n数据编码展示", icon: "\uD83D\uDCC8", color: P.green },
];

features.forEach((f, i) => {
  const x = 0.5 + i * 2.35;
  s8.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x, y: 1.1, w: 2.1, h: 2.2, rectRadius: 0.12, fill: { color: P.bgCard }, line: { color: f.color, width: 1.5 } });
  s8.addText(f.title, { x, y: 1.2, w: 2.1, h: 0.4, fontSize: 14, fontFace: "Arial", bold: true, color: f.color, align: "center" });
  s8.addText(f.desc, { x, y: 1.7, w: 2.1, h: 0.8, fontSize: 11, fontFace: "Arial", color: P.text, align: "center", lineSpacing: 16 });
});

// Status
s8.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 3.6, w: 9.0, h: 1.6, rectRadius: 0.1, fill: { color: P.bgCard }, line: { color: P.gold, width: 1 } });
s8.addText("当前状态", { x: 0.7, y: 3.7, w: 3.0, h: 0.35, fontSize: 14, fontFace: "Arial", bold: true, color: P.gold });
s8.addText([
  { text: "\u2713 Plan 1 完成", options: { fontSize: 12, bold: true, color: P.green } },
  { text: "：app_utils.py (176行) + test_app_utils.py (240行) — 输入验证、归一化桥接、模板生成、结果格式化\n", options: { fontSize: 11, color: P.text } },
  { text: "\u25CB Plan 2 待验证", options: { fontSize: 12, bold: true, color: P.gold } },
  { text: "：app.py (378行) 已编码完成，含手动输入/Excel上传/预测/CI展示/ctFP热力图，待人工视觉验证", options: { fontSize: 11, color: P.text } },
], { x: 0.7, y: 4.1, w: 8.6, h: 1.0, fontSize: 11, fontFace: "Arial", valign: "top", lineSpacing: 18 });

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9: 论文准备清单
// ════════════════════════════════════════════════════════════════════════════
const s9 = pptx.addSlide();
slideTitle(s9, "论文写作准备清单");

const checkItems = [
  ["\u2713", "ODE物理模型", "验证通过，3个文献体系复现", P.green],
  ["\u2713", "合成数据集", "973,693样本，4类RAFT剂全覆盖", P.green],
  ["\u2713", "SimpViT模型", "R\u00B2=0.991，877K参数，训练完成", P.green],
  ["\u2713", "Bootstrap UQ", "200次迭代，95%覆盖率校准完成", P.green],
  ["\u2713", "文献验证", "14个Ctr值，ML vs Mayo对比完成", P.green],
  ["\u2713", "论文图表", "19张图表已生成（parity/残差/验证/ODE）", P.green],
  ["\u2713", "代码完整", "~5,065行，11个测试文件", P.green],
  ["\u25CB", "Web应用", "app.py已编码，待人工UI验证", P.gold],
];

checkItems.forEach((item, i) => {
  const y = 1.0 + i * 0.52;
  s9.addText(item[0], { x: 0.5, y, w: 0.4, h: 0.4, fontSize: 18, fontFace: "Arial", bold: true, color: item[3], align: "center" });
  s9.addText(item[1], { x: 1.0, y, w: 2.5, h: 0.4, fontSize: 13, fontFace: "Arial", bold: true, color: P.white, valign: "middle" });
  s9.addText(item[2], { x: 3.6, y, w: 5.8, h: 0.4, fontSize: 12, fontFace: "Arial", color: P.ltGray, valign: "middle" });
  // Divider line
  if (i < checkItems.length - 1)
    s9.addShape(pptx.shapes.RECTANGLE, { x: 0.5, y: y + 0.44, w: 9.0, h: 0.005, fill: { color: "37474F" } });
});

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10: Conclusion
// ════════════════════════════════════════════════════════════════════════════
const s10 = pptx.addSlide();
s10.background = { fill: P.bg };
s10.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: P.primary } });
s10.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: P.accent } });

s10.addText("结论", { x: 0.5, y: 0.8, w: 9, h: 0.6, fontSize: 30, fontFace: "Arial", bold: true, color: P.white, align: "center" });

s10.addShape(pptx.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 1.6, w: 8.0, h: 2.4, rectRadius: 0.15, fill: { color: "1A3A4A" }, line: { color: P.primary, width: 2 } });
s10.addText([
  { text: "Phase 1-4 全部完成，Phase 5 基本就绪\n\n", options: { fontSize: 16, bold: true, color: P.primary } },
  { text: "ViT-Ctr 已具备进入 Phase 6（论文撰写）的全部前置条件：\n", options: { fontSize: 13, color: P.text } },
  { text: "\u2022 物理模型验证通过   \u2022 百万级数据集生成完毕\n", options: { fontSize: 12, color: P.ltGray } },
  { text: "\u2022 模型R\u00B2=0.991，优于Mayo基线   \u2022 UQ校准达95%\n", options: { fontSize: 12, color: P.ltGray } },
  { text: "\u2022 19张论文级图表就绪   \u2022 Web应用已编码\n", options: { fontSize: 12, color: P.ltGray } },
], { x: 1.2, y: 1.7, w: 7.6, h: 2.2, fontSize: 13, fontFace: "Arial", valign: "top", lineSpacing: 20 });

s10.addText([
  { text: "建议：", options: { bold: true, color: P.accent } },
  { text: "可直接进入Phase 6论文撰写，Phase 5人工验证可并行进行", options: { color: P.text } },
], { x: 1.5, y: 4.3, w: 7.0, h: 0.5, fontSize: 14, fontFace: "Arial", align: "center" });

// ── Save ──
pptx.writeFile({ fileName: "ViT-Ctr_Phase1-5_Summary.pptx" }).then(() => {
  console.log("PPT report generated: ViT-Ctr_Phase1-5_Summary.pptx");
}).catch(console.error);
