"""Generate a Chinese design-presentation .pptx for the joint TextGrad optimization.

Output: test/textgrad_validation/TextGrad联合优化方案.pptx
Requires: python-pptx (verified 1.0.2)
"""
from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
PRIMARY = RGBColor(0x1F, 0x4E, 0x79)   # deep blue
ACCENT = RGBColor(0x2E, 0x75, 0xB6)    # light blue
GOOD = RGBColor(0x2E, 0x8B, 0x57)      # green
DANGER = RGBColor(0xC0, 0x00, 0x00)    # red
DARK = RGBColor(0x33, 0x33, 0x33)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xEF, 0xF3, 0xF8)     # light blue fill
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
EXBG = RGBColor(0xFF, 0xF4, 0xE0)      # example: light amber fill
EXLINE = RGBColor(0xE8, 0xA3, 0x3D)    # example: amber border
FONT = "Microsoft YaHei"

SW, SH = 13.333, 7.5


def _set_font(run, size=18, bold=False, color=DARK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", name)


def _fill_para(tf, lines, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT, bullet="", space=6):
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        run = p.add_run()
        run.text = (bullet + line) if bullet else line
        _set_font(run, size, bold, color)


def add_text(slide, l, t, w, h, lines, size=18, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, bullet="", space=6):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    _fill_para(tf, lines if isinstance(lines, list) else [lines], size, bold, color, align, bullet, space)
    return box


def add_box(slide, l, t, w, h, lines, fill=ACCENT, text_color=WHITE, size=15, bold=True,
            line_color=None, align=PP_ALIGN.CENTER, bullet=""):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line_color is not None:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    _fill_para(tf, lines if isinstance(lines, list) else [lines], size, bold, text_color, align, bullet, space=4)
    return shape


def add_arrow(slide, l, t, w, h, color=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def header(slide, title, subtitle=None, page=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = PRIMARY
    bar.line.fill.background()
    add_text(slide, 0.6, 0.35, 12.1, 0.7, title, size=27, bold=True, color=PRIMARY)
    if subtitle:
        add_text(slide, 0.62, 1.02, 12.1, 0.4, subtitle, size=14, color=GRAY)
    if page:
        add_text(slide, 12.35, 7.02, 0.85, 0.35, str(page), size=12, color=GRAY, align=PP_ALIGN.RIGHT)


def why_slide(prs, page, title, subtitle, reasons, example_title, example_lines):
    """A 'why' page: three reason cards + one highlighted example card."""
    slide = new_slide(prs)
    header(slide, title, subtitle, page)
    x = 0.6
    for head, body in reasons:
        add_box(slide, x, 1.7, 3.95, 3.2, [head, "", body], fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
        x += 4.05
    tag = add_box(slide, 0.6, 5.25, 1.55, 0.55, "例子", fill=EXLINE, size=15)
    add_box(slide, 2.15, 5.25, 10.55, 1.9,
            [example_title, "", *example_lines],
            fill=EXBG, text_color=DARK, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=EXLINE)
    return slide


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def summary(prs, page):
    """One-page overview: problem -> approach -> round flow -> four design decisions."""
    slide = new_slide(prs)
    header(slide, "一页概括", "问题 → 方案 → 一轮流程 → 四个设计决策", page)
    add_box(slide, 0.6, 1.45, 12.1, 0.75,
            ["问题：schema description 静态 → 字段边界错误反复      方案：与抽取提示词一起被梯度优化，每轮一个联合候选"],
            fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
    add_text(slide, 0.6, 2.3, 7.2, 0.4, "一轮流程（round-00X/joint）", size=16, bold=True, color=PRIMARY)
    steps = [
        "两个独立 TextGrad 变量：schema 补丁提示词 / 抽取提示词",
        "基于同一已接受反馈各自 backward → 两个独立优化器各自 step",
        "更新后 schema 提示词生成补丁 → description-only 校验 → 候选 schema",
        "候选 schema + 新抽取提示词 → 3 篇训练 PDF 各评分一次",
        "有效 → 原子接受；非法 / 缺分 → 两个变量与 schema 一起回滚",
    ]
    y = 2.78
    for i, s in enumerate(steps, 1):
        add_box(slide, 0.6, y, 0.5, 0.5, str(i), fill=ACCENT, size=15)
        add_box(slide, 1.25, y, 6.55, 0.5, s, fill=WHITE, text_color=DARK, size=12, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)
        y += 0.62
    add_box(slide, 0.6, 6.0, 7.2, 0.95,
            ["结论只看盲测配对（优化 − 基线）；训练分数仅作诊断。"],
            fill=LIGHT, text_color=DARK, size=14, bold=False)
    add_text(slide, 8.15, 2.3, 4.6, 0.4, "四个设计决策", size=16, bold=True, color=PRIMARY)
    cards = [
        ("联合", "一轮一个分数，增量可归因"),
        ("无门槛", "补丁合法 + 3 篇有分即接受"),
        ("description-only", "结构指纹不变，只改字段语义"),
        ("单阶段", "每篇一次 PDF→JSON，无 two-stage"),
    ]
    y = 2.78
    for title, body in cards:
        add_box(slide, 8.15, y, 4.6, 1.15, [title, "", body], fill=WHITE, text_color=PRIMARY, size=14, bold=False, line_color=ACCENT)
        y += 1.27


def cover(prs):
    slide = new_slide(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(2.6))
    band.fill.solid()
    band.fill.fore_color.rgb = PRIMARY
    band.line.fill.background()
    add_text(slide, 0.9, 0.85, 11.5, 1.1,
             "TextGrad 联合优化方案", size=44, bold=True, color=WHITE)
    add_text(slide, 0.9, 1.95, 11.5, 0.6,
             "化学文献 PDF→JSON 结构化抽取 · schema description 与抽取提示词的联合优化",
             size=18, color=RGBColor(0xD9, 0xE2, 0xF3))
    add_text(slide, 0.9, 3.05, 11.5, 0.45,
             "单阶段抽取 · 双提示词联合 TextGrad 优化 · 无门槛接受 · description-only 安全补丁",
             size=15, color=PRIMARY, bold=True)
    add_box(slide, 0.9, 4.0, 4.6, 1.35,
            ["每轮一个联合候选", "round-00X/joint"], fill=LIGHT, text_color=PRIMARY, size=16)
    add_box(slide, 5.8, 4.0, 4.6, 1.35,
            ["有效即接受", "补丁非法 / 缺分才回滚"], fill=LIGHT, text_color=GOOD, size=16)
    add_text(slide, 0.9, 6.35, 11.5, 0.4,
             "TextGrad 金标验证实验  ·  2026-08-04", size=14, color=GRAY)


def overview(prs, page):
    slide = new_slide(prs)
    header(slide, "概览：为什么需要联合优化", "核心问题是 schema 的 description 描述无法从训练证据中吸收字段边界知识", page)
    add_box(slide, 0.6, 1.7, 3.85, 2.1,
            ["目标", "LLM 从化学论文 PDF 直接抽取结构化 JSON（聚合物 / 反应 / 性质）"],
            fill=LIGHT, text_color=PRIMARY, size=15)
    add_box(slide, 4.74, 1.7, 3.85, 2.1,
            ["现状", "抽取提示词由 TextGrad 优化；但 schema 的 description 是静态全局文本"],
            fill=LIGHT, text_color=PRIMARY, size=15)
    add_box(slide, 8.88, 1.7, 3.85, 2.1,
            ["痛点", "字段边界不清 → 命名 / 结构歧义错误反复出现"],
            fill=LIGHT, text_color=DANGER, size=15)
    add_text(slide, 0.6, 4.15, 12.1, 0.45,
             "本方案：同时优化“schema-description 补丁提示词”与“抽取提示词”，每轮产生一个联合候选", size=17, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 4.9, 12.1, 1.6,
            ["联合优化 = 同一已接受反馈 → 两个独立 TextGrad loss → 各自 step → 候选 schema + 新抽取提示词只评分一次",
             "变量之间绝不合拼：schema 提示词与抽取提示词是两个独立变量，各自保持任务语义与安全约束"],
            fill=WHITE, text_color=DARK, size=15, bold=False, line_color=ACCENT)
    add_text(slide, 0.6, 6.8, 12.1, 0.4,
             "抽取仍为单阶段：每篇文档一次 PDF→JSON 调用，额外模型操作仅训练时的补丁生成", size=13, color=GRAY)


def background(prs, page):
    slide = new_slide(prs)
    header(slide, "背景：静态 description 的缺陷", "这些是在真实训练抽取中反复出现的错误", page)
    add_text(slide, 0.6, 1.6, 12.1, 0.45, "反复出现的三类错误（真实抽取片段）", size=18, bold=True, color=PRIMARY)
    rows = [
        ("自由文本样本身份标识",
         "身份标识 = “Hydroxylated aromatic polyamic acid IVa derived from … (I) and pyromellitic dianhydride (IIa)”\n应只写论文命名的样品代号，却写成含单体与反应路线的一段描述"),
        ("样本形态混入外观/尺寸",
         "样本形态 = “precipitated polymer; pale yellow transparent film”\n颜色、透明度本不属于形态字段"),
        ("工艺中间体被误提升为独立聚合物",
         "同一产物在“工艺流程”出现 5 个近似重复条目（98% / 95% / 96% / 99%），中间体与产物层级不分"),
    ]
    y = 2.15
    for title, desc in rows:
        add_box(slide, 0.6, y, 3.7, 1.35, title, fill=PRIMARY, size=14)
        add_box(slide, 4.5, y, 8.2, 1.35, desc.split("\n"), fill=LIGHT, text_color=DARK, size=13, bold=False, align=PP_ALIGN.LEFT)
        y += 1.55
    add_box(slide, 0.6, 6.0, 12.1, 1.0,
            ["根因：description 是“写给抽取模型看”的字段语义指南，但它是固定文本；TextGrad 没有渠道把训练中暴露的字段边界问题写回 schema。"],
            fill=WHITE, text_color=DANGER, size=14, bold=False, line_color=DANGER)


def core(prs, page):
    slide = new_slide(prs)
    header(slide, "核心思想：两个可优化变量", "它们是两个独立 TextGrad 变量，每轮各自更新、一起评估", page)
    add_box(slide, 0.7, 1.75, 5.9, 1.5,
            ["变量 1 · schema-description 补丁提示词",
             "提议 description-only 的 JSON 补丁，受不可变安全协议约束"],
            fill=ACCENT, size=15)
    add_box(slide, 6.8, 1.75, 5.9, 1.5,
            ["变量 2 · 抽取提示词",
             "PDF→JSON 单次调用的抽取指令，保持 JSON-only 输出"],
            fill=PRIMARY, size=15)
    add_text(slide, 0.7, 3.5, 12.0, 0.4, "一轮的联合流程", size=17, bold=True, color=PRIMARY)
    steps = [
        "同一已接受状态的 3 篇 judge 反馈 → 各自构造 per-document TextLoss",
        "schema 提示词 loss 与抽取提示词 loss 都 backward（梯度独立）",
        "一个优化器同时 step 两个变量",
        "更新后的 schema 提示词生成补丁 → description-only 校验 → 物化候选 schema",
        "候选 schema + 新抽取提示词 → 3 篇训练 PDF 各重抽取 + 评分一次",
        "有效 → 原子接受；补丁非法 / 缺分 → 两个变量与 schema 一起回滚",
    ]
    y = 4.0
    for i, step in enumerate(steps, 1):
        add_box(slide, 0.7, y, 0.55, 0.55, str(i), fill=ACCENT, size=15)
        add_text(slide, 1.45, y + 0.04, 11.3, 0.55, step, size=14, color=DARK)
        y += 0.66


def why_joint(prs, page):
    slide = new_slide(prs)
    header(slide, "为什么“联合”而不是“交替”？", "一轮一个分数，才谈得上归因；两个分数各说各话只会互相掩盖", page)
    add_box(slide, 0.6, 1.7, 3.95, 3.2,
            ["可归因性", "", "交替流程每轮两个分数（schema 阶段、抽取阶段）。\n分数涨了，说不清是谁带来的；\n联合流程只有一个分数，归因于二者共同提议的状态。"],
            fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
    add_box(slide, 4.65, 1.7, 3.95, 3.2,
            ["训练预算", "", "交替流程一轮要重抽取 + 评分两次；\n联合流程一轮只有一次。\n同样轮数下 API 预算更省，也更接近一次真实抽取的语义。"],
            fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
    add_box(slide, 8.7, 1.7, 3.95, 3.2,
            ["语义一致性", "", "schema 与抽取提示词一起推进，\n不会出现“schema 已改、抽取还是旧行为”的错位中间态。"],
            fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
    add_box(slide, 0.6, 5.25, 1.55, 0.55, "例子", fill=EXLINE, size=15)
    add_box(slide, 2.15, 5.25, 10.55, 1.9,
            ["交替流程里一个正确的补丁被冤枉拒绝",
             "第 3 轮把“身份标识”的 description 改对了，但交替流程用【旧抽取提示词】先评 schema 阶段：抽取还是旧行为，分数没涨 → 补丁被拒。",
             "联合流程让候选 schema 与新抽取提示词一起评估，正确补丁的效果才会真正显现。"],
            fill=EXBG, text_color=DARK, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=EXLINE)


def compare(prs, page):
    slide = new_slide(prs)
    header(slide, "旧交替流程 vs 新联合流程", "联合候选让一轮只有一个分数，变化来源可归因", page)
    add_text(slide, 0.6, 1.55, 5.9, 0.45, "旧：交替两个阶段", size=18, bold=True, color=GRAY)
    old = [
        "① schema 阶段：旧抽取提示词 + 候选 schema → 评分",
        "② extraction 阶段：新抽取提示词 + 已接受 schema → 评分",
        "一轮两个分数，两个变化来源无法区分",
        "schema 提升与抽取提升互相掩盖归因",
    ]
    add_box(slide, 0.6, 2.1, 5.9, 3.4, old, fill=LIGHT, text_color=DARK, size=14, bold=False, align=PP_ALIGN.LEFT)
    add_text(slide, 6.8, 1.55, 5.9, 0.45, "新：联合一个候选", size=18, bold=True, color=PRIMARY)
    new = [
        "① 两个变量基于同一反馈各自 step",
        "② 候选 schema + 新抽取提示词只评估一次",
        "一轮只有一个分数（round-00X/joint）",
        "分数归因于“二者共同提议的状态”，决策原子化",
    ]
    add_box(slide, 6.8, 2.1, 5.9, 3.4, new, fill=WHITE, text_color=PRIMARY, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)
    add_box(slide, 0.6, 5.85, 12.1, 1.0,
            ["旧流程每轮两次重抽取+评分；新流程每轮一次 → 同样轮数下训练 API 预算更省、语义更一致"],
            fill=PRIMARY, size=15)


def flow(prs, page):
    slide = new_slide(prs)
    header(slide, "联合优化：一轮的完整流程", "每轮产出 round-00X/joint 唯一评分，原子接受或回滚", page)
    add_box(slide, 3.4, 1.5, 6.5, 0.7, "已接受训练状态（3 篇 judge 反馈）", fill=LIGHT, text_color=PRIMARY, size=15)
    add_arrow(slide, 6.5, 2.2, 0.35, 0.3)
    add_box(slide, 0.7, 2.55, 5.9, 1.15,
            ["阶段 1 · 梯度", "schema 提示词 loss + 抽取提示词 loss\n（per-document batch，各自 backward）"],
            fill=ACCENT, size=14)
    add_box(slide, 6.8, 2.55, 5.9, 1.15,
            ["阶段 2 · 各自 step", "两个独立优化器\nschema 与抽取各用各的约束"],
            fill=ACCENT, size=14)
    add_arrow(slide, 6.4, 3.0, 0.35, 0.3)
    add_box(slide, 0.7, 4.15, 5.9, 1.15,
            ["阶段 3 · 补丁", "更新后 schema 提示词生成补丁\n→ description-only 校验 → 候选 schema"],
            fill=PRIMARY, size=14)
    add_box(slide, 6.8, 4.15, 5.9, 1.15,
            ["阶段 4 · 联合评估", "候选 schema + 新抽取提示词\n→ 3 篇训练 PDF 各评分一次"],
            fill=PRIMARY, size=14)
    add_arrow(slide, 6.4, 4.6, 0.35, 0.3)
    add_box(slide, 0.7, 5.75, 7.4, 1.0,
            ["判定", "补丁合法 且 3 篇均有有效评分 → 原子接受"], fill=GOOD, size=15)
    add_box(slide, 8.4, 5.75, 4.3, 1.0,
            ["否则 → 两个变量 + schema 一起回滚"], fill=DANGER, size=14)


def why_threshold(prs, page):
    slide = new_slide(prs)
    header(slide, "为什么去掉 min_accept_delta 门槛？", "门槛会丢弃本该推进的更新；接受与否只看“是否有效”", page)
    add_box(slide, 0.6, 1.7, 3.95, 3.2,
            ["梯度本身已是指向改进的方向", "", "loss 来自 judge 反馈（GENERAL_RULE），已是“哪里要改”的指令；\ndelta 是一次二次筛选，容易把正确更新当噪声扔掉。"],
            fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
    add_box(slide, 4.65, 1.7, 3.95, 3.2,
            ["训练分数只是诊断", "", "真正结论来自盲测配对（优化 − 基线）；\n用训练均分 delta 卡接受，等于在噪声上做硬决策，\n也容易过拟合三篇训练。"],
            fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
    add_box(slide, 8.7, 1.7, 3.95, 3.2,
            ["回滚仍然存在", "", "补丁非法 / 任一文档缺分 → 回滚。\n“无门槛”不是乱接受，而是只按有效性判定。"],
            fill=LIGHT, text_color=PRIMARY, size=14, bold=False)
    add_box(slide, 0.6, 5.25, 1.55, 0.55, "例子", fill=EXLINE, size=15)
    add_box(slide, 2.15, 5.25, 10.55, 1.9,
            ["正确补丁因“恰好没涨分”被误杀",
             "第 2 轮把“样本形态”description 改对（只记物态、不写颜色），但 3 篇里恰好有一篇没报该字段 → 均分持平，delta=0 < 2 → 旧设计拒绝。",
             "无门槛接受则直接推进，让梯度连续累积；盲测才见分晓。"],
            fill=EXBG, text_color=DARK, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=EXLINE)


def acceptance(prs, page):
    slide = new_slide(prs)
    header(slide, "无门槛接受（去掉了 min_accept_delta）", "接受与否只看“是否有效”，不再比较均分提升幅度", page)
    add_text(slide, 0.6, 1.6, 12.1, 0.45, "接受条件", size=18, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 2.15, 12.1, 1.0,
            ["补丁合法（通过 description-only 校验） 且  3 篇训练文档均有有效 judge 评分"],
            fill=GOOD, size=17)
    add_text(slide, 0.6, 3.4, 12.1, 0.45, "回滚条件（仅这两类）", size=18, bold=True, color=DANGER)
    add_box(slide, 0.6, 3.95, 5.9, 1.0, "补丁非法（proposal_failed）", fill=DANGER, size=15)
    add_box(slide, 6.8, 3.95, 5.9, 1.0, "任一文档评分失败（缺分）", fill=DANGER, size=15)
    add_box(slide, 0.6, 5.35, 12.1, 1.35,
            ["为什么不再设提升门槛：TextGrad 每轮已基于“上一已接受反馈”产生梯度；",
             "加 delta 门槛会把本应推进的更新当作噪声丢弃，也削弱梯度反馈的连续性。"],
            fill=LIGHT, text_color=DARK, size=14, bold=False)


def safety(prs, page):
    slide = new_slide(prs)
    header(slide, "description-only 补丁安全契约", "补丁只能触碰 description 字符串，结构语义保持不变", page)
    add_text(slide, 0.6, 1.5, 5.9, 0.45, "允许", size=17, bold=True, color=GOOD)
    allow = [
        "修改已存在的 description 字符串",
        "补丁输出为严格 JSON 补丁文档",
        "描述面向通用抽取指引",
    ]
    add_box(slide, 0.6, 1.95, 5.9, 1.8, allow, fill=WHITE, text_color=GOOD, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=GOOD)
    add_text(slide, 6.8, 1.5, 5.9, 0.45, "禁止", size=17, bold=True, color=DANGER)
    forbid = [
        "改动字段名 / 类型 / required / items / properties / additionalProperties",
        "写入文档特定事实：标识符、DOI、URL、数值与单位、JSONPath、示例、实验条件",
        "非严格 JSON（拒绝 NaN / Infinity / 重复键 / 散文 / Markdown）",
    ]
    add_box(slide, 6.8, 1.95, 5.9, 1.8, forbid, fill=WHITE, text_color=DANGER, size=13, bold=False, align=PP_ALIGN.LEFT, line_color=DANGER)
    add_text(slide, 0.6, 4.05, 12.1, 0.45, "为什么只改 description（真实例子）", size=16, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 4.6, 2.2, 1.5, ["身份标识", "改前：样本身份标识\n改后：只写论文明确命名的样品代号（如 IVa、Va），不要写成含单体与反应路线的描述"],
            fill=LIGHT, text_color=DARK, size=12, bold=False, align=PP_ALIGN.LEFT)
    add_arrow(slide, 2.95, 5.0, 0.5, 0.4, color=GOOD)
    add_box(slide, 3.6, 4.6, 4.4, 1.5, ["抽取结果：长句 → 规范代号", "“Hydroxylated aromatic polyamic acid IVa derived from …” → “IVa”"],
            fill=WHITE, text_color=GOOD, size=12, bold=False, align=PP_ALIGN.LEFT, line_color=GOOD)
    add_box(slide, 8.2, 4.6, 2.0, 1.5, ["样本形态", "改前：样本的形态\n改后：只记录物态（粉末 / 薄膜 / 纤维 / 油），不写颜色、透明度、外观"],
            fill=LIGHT, text_color=DARK, size=12, bold=False, align=PP_ALIGN.LEFT)
    add_arrow(slide, 10.3, 5.0, 0.5, 0.4, color=GOOD)
    add_box(slide, 10.9, 4.6, 1.9, 1.5, ["抽取结果：去外观", "“precipitated polymer; pale yellow transparent film” → 只留物态"],
            fill=WHITE, text_color=GOOD, size=12, bold=False, align=PP_ALIGN.LEFT, line_color=GOOD)
    add_box(slide, 0.6, 6.35, 12.1, 0.75,
            ["改结构会改变 JSON 形状、破坏盲测可比性；description 只改变“模型怎么理解字段”，是安全的调优面。"],
            fill=LIGHT, text_color=PRIMARY, size=13)


def fingerprint(prs, page):
    slide = new_slide(prs)
    header(slide, "结构指纹与缓存隔离", "description 变化只影响提示词语义，不改变数据结构，并自动失效相关缓存", page)
    add_text(slide, 0.6, 1.6, 12.1, 0.45, "结构指纹（安全不变式）", size=17, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 2.1, 12.1, 1.0,
            ["对 schema DSL 递归剥离所有 description 后取 SHA-256；原始 vs 补丁后必须一致"],
            fill=LIGHT, text_color=DARK, size=15)
    add_text(slide, 0.6, 3.35, 12.1, 0.45, "缓存隔离", size=17, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 3.85, 5.9, 1.5,
            ["候选 schema 的完整 DSL hash 进入抽取缓存键与裁判缓存键",
             "description 一变 → hash 变 → 缓存自动失效"],
            fill=WHITE, text_color=PRIMARY, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)
    add_box(slide, 6.8, 3.85, 5.9, 1.5,
            ["候选 schema 的 description 会真正传给抽取模型",
             "确保补丁效果反映在抽取结果里，而不是停留在文件层面"],
            fill=WHITE, text_color=PRIMARY, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)
    add_box(slide, 0.6, 5.7, 12.1, 1.0,
            ["过去 bug：候选 schema 没送达抽取模型、缓存键不含 DSL hash → 全轮命中缓存；现已修复并加回归测试。"],
            fill=WHITE, text_color=DANGER, size=14, bold=False, line_color=DANGER)


def engineering(prs, page):
    slide = new_slide(prs)
    header(slide, "工程机制：缓存开关与裁判重试", "可控的成本与健壮的评测链路", page)
    add_text(slide, 0.6, 1.55, 12.1, 0.45, "缓存开关", size=17, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 2.05, 12.1, 1.3,
            ["cache.enabled: false → 强制每次重新调用 extractor / judge / audit",
             "同时绕开 run-local 已存结果与 results/_cache 共享缓存的读写",
             "该字段纳入配置指纹：切换缓存开关必须使用新 run-id"],
            fill=WHITE, text_color=DARK, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)
    add_text(slide, 0.6, 3.6, 12.1, 0.45, "裁判重试", size=17, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 4.1, 12.1, 1.3,
            ["judge 结果解析失败时最多重试 max_retries 次",
             "日志记录 recovery：judge recovered on retry N, score=…",
             "修复了“空 optimization_feedback 一次失败就废掉整个候选”的边界情况"],
            fill=WHITE, text_color=DARK, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)
    add_box(slide, 0.6, 5.7, 12.1, 1.0,
            ["schema-patch 模型复用 judge 角色，不新增模型密钥；请求格式与缓存键保持版本化。"],
            fill=LIGHT, text_color=PRIMARY, size=14)


def experiment(prs, page):
    slide = new_slide(prs)
    header(slide, "实验设置", "金标验证：训练分数仅作诊断，结论以盲测配对为准", page)
    metrics = [
        ("20 篇文档", "确定性 3/17 分割", "训练 3 · 盲测 17"),
        ("max_iterations", "联合轮数", "同 run-id 可上调/下调"),
        ("主裁判", "prediction vs gold", "看不到 PDF（PDF 守卫）"),
        ("金标审计", "advisory", "PDF 核查 gold，不改分"),
    ]
    x = 0.6
    for title, desc, extra in metrics:
        add_box(slide, x, 1.7, 2.95, 1.6, [title, "", desc, extra], fill=LIGHT, text_color=PRIMARY, size=14)
        x += 3.05
    add_box(slide, 0.6, 3.6, 12.1, 1.15,
            ["盲测配对结果 = optimized − baseline 的逐文档 delta",
             "胜 / 平 / 负 与均值增量汇总；基线臂用基线 schema，优化臂用最终选定 schema + 抽取提示词"],
            fill=WHITE, text_color=DARK, size=15, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)
    add_text(slide, 0.6, 5.05, 12.1, 0.45, "不可变输入（变更需新 run-id）", size=16, bold=True, color=PRIMARY)
    add_box(slide, 0.6, 5.55, 12.1, 1.15,
            ["模式 / schema / 提示词 / 模型 / 分割算法 / 请求格式 / 补丁策略版本 / 缓存开关",
             "只有 max_iterations 可在同 run-id 续跑时变更"],
            fill=LIGHT, text_color=DARK, size=14, bold=False)


def progress(prs, page):
    slide = new_slide(prs)
    header(slide, "当前进展与下一步", "", page)
    add_text(slide, 0.6, 1.5, 12.1, 0.45, "已完成", size=17, bold=True, color=GOOD)
    done = [
        "联合优化器（joint-round-v2）：两变量各自 step，每轮一个候选",
        "无门槛接受：有效即接受，仅非法/缺分回滚",
        "缓存开关 cache.enabled，judge 重试机制",
        "回归测试全绿：100 passed",
    ]
    add_box(slide, 0.6, 2.0, 12.1, 1.8, done, fill=WHITE, text_color=GOOD, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=GOOD)
    add_text(slide, 0.6, 4.05, 12.1, 0.45, "下一步", size=17, bold=True, color=PRIMARY)
    nxt = [
        "用新 run-id 跑真实实验（指纹已变更，旧 run-id 不能续跑）",
        "分析盲测配对 delta 与胜/平/负；核对 description 补丁是否符合预期、结构指纹恒定",
        "若推广：多 seed 稳定性、跨领域验证、与外部基线对比",
    ]
    add_box(slide, 0.6, 4.55, 12.1, 1.8, nxt, fill=WHITE, text_color=PRIMARY, size=14, bold=False, align=PP_ALIGN.LEFT, line_color=ACCENT)


def closing(prs, page):
    slide = new_slide(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(SH))
    band.fill.solid()
    band.fill.fore_color.rgb = PRIMARY
    band.line.fill.background()
    add_text(slide, 1.2, 2.2, 10.9, 1.0,
             "联合优化：让抽取提示词与 schema 语义一起被梯度改进", size=30, bold=True, color=WHITE)
    add_text(slide, 1.2, 3.3, 10.9, 0.6,
             "每轮一个候选 · 无门槛接受 · description-only 安全补丁 · 单阶段抽取", size=18, color=RGBColor(0xD9, 0xE2, 0xF3))
    add_text(slide, 1.2, 6.2, 10.9, 0.4, "谢谢", size=16, color=WHITE)


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    cover(prs)
    summary(prs, 2)
    overview(prs, 3)
    background(prs, 4)
    core(prs, 5)
    why_joint(prs, 6)
    compare(prs, 7)
    flow(prs, 8)
    why_threshold(prs, 9)
    acceptance(prs, 10)
    safety(prs, 11)
    fingerprint(prs, 12)
    engineering(prs, 13)
    experiment(prs, 14)
    progress(prs, 15)
    closing(prs, 16)
    out = "test/textgrad_validation/TextGrad联合优化方案.pptx"
    prs.save(out)
    print("saved:", out, "| slides:", len(prs.slides._sldIdLst))


if __name__ == "__main__":
    main()
