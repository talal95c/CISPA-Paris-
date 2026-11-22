"""Teletubbies slides: intro, then one part per task (problem -> idea -> result -> code). Run charts.py first."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

DARK, GREY, LIGHT, WHITE = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x5A, 0x63, 0x73), RGBColor(0xF3, 0xF5, 0xF9), RGBColor(255, 255, 255)
BLUE, PINK, GREEN = RGBColor(0x2F, 0x6F, 0xD6), RGBColor(0xE0, 0x6C, 0x9F), RGBColor(0x2E, 0xA6, 0x6B)
P = Presentation(); P.slide_width, P.slide_height = Inches(13.333), Inches(7.5)


def text(s, x, y, w, h, t, size=20, color=DARK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    for i, line in enumerate(t.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, 'Calibri'
    return tf


def box(s, x, y, w, h, fill, t='', size=18, color=WHITE, bold=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.fill.background(); b.shadow.inherit = False
    if t:
        tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, line in enumerate(t.split('\n')):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = line
            r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold if i == 0 else False, color, 'Calibri'
    return b


def arrow(s, x, y, w=0.45, color=GREY):
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(0.4))
    a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background()


def page(title, part=None, color=DARK):
    s = P.slides.add_slide(P.slide_layouts[6])
    box(s, 0, 0, 13.333, 0.12, color, shape=MSO_SHAPE.RECTANGLE)
    if part:
        box(s, 11.0, 0.35, 1.9, 0.45, color, part, size=14)
    text(s, 0.6, 0.3, 10.3, 0.9, title, size=32, bold=True)
    return s


def bullets(s, x, y, w, items, size=20, color=DARK, dot=DARK):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(4)).text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        r = p.add_run(); r.text = '●  '; r.font.size, r.font.color.rgb = Pt(size - 6), dot
        r = p.add_run(); r.text = it; r.font.size, r.font.color.rgb, r.font.name = Pt(size), color, 'Calibri'


def section(n, name, color, question, score):
    s = P.slides.add_slide(P.slide_layouts[6])
    box(s, 0, 0, 13.333, 7.5, color, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.9, 1.4, 11, 0.8, f'PART {n} / 3', size=24, color=WHITE, bold=True)
    text(s, 0.9, 2.1, 11.5, 1.4, name, size=54, color=WHITE, bold=True)
    text(s, 0.9, 3.6, 11.5, 1.2, question, size=26, color=WHITE)
    box(s, 0.9, 5.2, 4.6, 1.1, WHITE, f'Best public score: {score}', size=24, color=color)


def code(s, path, color):
    box(s, 0.6, 6.65, 12.1, 0.55, LIGHT, '', shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 0.8, 6.68, 11.8, 0.5, '</> Code:  ' + path, size=15, color=color, bold=True, anchor=MSO_ANCHOR.MIDDLE)


def pic(s, f, x, y, w=None, h=None):
    s.shapes.add_picture(f'img/{f}', Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


# ---------- Intro ----------
s = P.slides.add_slide(P.slide_layouts[6])
box(s, 0, 0, 13.333, 7.5, DARK, shape=MSO_SHAPE.RECTANGLE)
text(s, 0.9, 1.0, 11.5, 1.2, 'Teletubbies', size=66, color=WHITE, bold=True)
text(s, 0.9, 2.2, 11.5, 0.8, 'CISPA × Fraunhofer AISEC Hackathon — Paris 2026', size=26, color=WHITE)
for i, (c, n, sc) in enumerate([(BLUE, '1  Watermark\nLocalization', '0.714 mIoU'), (PINK, '2  Model Equality\nTesting', '1.00 MCC'),
                                (GREEN, '3  Unlearning', '0.112')]):
    box(s, 0.9 + i * 4.0, 3.9, 3.6, 2.2, c, f'{n}\n\n{sc}', size=24)
text(s, 0.9, 6.5, 11.5, 0.6, 'github.com/talal95c/CISPA-Paris-   ·   public leaderboard scores', size=16, color=RGBColor(0xC8, 0xCE, 0xDA))

s = page('Our approach, the same for every task')
for i, (t, d) in enumerate([('Understand', 'what exactly\nthe score measures'), ('Exploit structure', 'rectangles, open weights,\nthe "gold" model'),
                            ('Verify', 'local checks before\nevery submission'), ('One change', 'per submission,\nexpected score written first')]):
    box(s, 0.6 + i * 3.15, 2.3, 2.7, 2.6, [BLUE, PINK, GREEN, DARK][i], f'{t}\n\n{d}', size=21)
    if i < 3: arrow(s, 3.33 + i * 3.15, 3.4)
text(s, 0.6, 5.5, 12, 1, 'Each part: the problem → our idea → the result → where the code is.', size=22, color=GREY, align=PP_ALIGN.CENTER)

# ---------- Part 1 ----------
section(1, 'Watermark Localization', BLUE, 'Which pixels of an edited image still carry the watermark?', '0.714 mIoU')

s = page('The problem', 'Part 1', BLUE)
pic(s, 't1_examples.png', 0.4, 1.25, h=5.2)
bullets(s, 8.4, 1.5, 4.6, ['Images watermarked with PixelSeal, StegaStamp or TrustMark',
                            'Then edited: outpainting, object removal / addition / transplant',
                            'Output: a binary mask (1 = watermarked)',
                            'Score: mIoU of the two classes, averaged over images'], size=19, dot=BLUE)
text(s, 0.4, 6.5, 8, 0.5, 'Real training examples; pink = ground-truth watermarked region.', size=13, color=GREY)

s = page('Our idea: segment, then verify', 'Part 1', BLUE)
steps = [('1. Segment', 'neural net predicts\na rough mask\n(trained on JURECA)'), ('2. Fit rectangles', 'watermarks live in\nrectangles\n(≤16:1 or ≤2:1)'),
         ('3. Decode inside', 'keep a rectangle only\nif it decodes the\nknown message'), ('4. Polish', 'fill PixelSeal holes,\nsnap rectangles\nto the pixel')]
for i, (t, d) in enumerate(steps):
    box(s, 0.6 + i * 3.15, 1.6, 2.7, 2.5, BLUE if i != 2 else DARK, f'{t}\n\n{d}', size=19)
    if i < 3: arrow(s, 3.33 + i * 3.15, 2.65)
box(s, 0.6, 4.6, 12.1, 1.6, LIGHT, '')
text(s, 0.9, 4.7, 11.6, 1.5, 'Why it works: the post-hoc watermarks are trained to decode one message from the whole image, not to localise.\n'
     'The message per image is given, so decoding becomes a yes/no test: "is this rectangle really watermarked?"', size=19, color=DARK)

s = page('Result: every step adds mIoU', 'Part 1', BLUE)
pic(s, 't1_scores.png', 0.4, 1.2, h=5.3)
bullets(s, 8.2, 1.6, 4.8, ['0.693 → 0.714 public mIoU', '60 TrustMark rectangles confirmed by decoding', '49 rectangles re-aligned to the pixel',
                            'Best: submission 4038'], size=20, dot=BLUE)
code(s, 'Task 1 - Watermark Localization/  (README + solution)', BLUE)

# ---------- Part 2 ----------
section(2, 'Model Equality Testing', PINK, '33 LLM endpoints: which ones serve the same weights?', '1.00 MCC')

s = page('The problem', 'Part 2', PINK)
box(s, 0.6, 1.5, 5.8, 0.7, PINK, 'Same model (same weights)', size=20)
bullets(s, 0.8, 2.4, 5.6, ['different temperature, top-k / top-p', 'different system prompt or persona', 'a text watermark'], size=20, dot=PINK)
box(s, 6.9, 1.5, 5.8, 0.7, DARK, 'Different model', size=20)
bullets(s, 7.1, 2.4, 5.6, ['another checkpoint', 'fp8 quantization', 'a LoRA adapter'], size=20)
box(s, 0.6, 4.6, 12.1, 1.6, LIGHT, '')
text(s, 0.9, 4.7, 11.6, 1.5, 'The trap: decoding changes make the text look different, while LoRA / fp8 make it look the same.\n'
     'Score: MCC over the 528 endpoint pairs (90 % same model, 10 % same family), limited query budget.', size=19)

s = page('Our idea: go white-box', 'Part 2', PINK)
flow = [('Ask', 'short & long prompts\nto every endpoint'), ('Identify', 'the open checkpoint:\nQwen2.5-3B / 1.5B,\nLlama-3.2-1B'),
        ('Re-score', 'each answer token\nby token under the\nopen weights'), ('Fingerprint', 'temperature, fp8,\nLoRA, watermark,\nprompt, top-k/p')]
for i, (t, d) in enumerate(flow):
    box(s, 0.6 + i * 3.15, 1.5, 2.7, 2.4, PINK if i != 2 else DARK, f'{t}\n\n{d}', size=19)
    if i < 3: arrow(s, 3.33 + i * 3.15, 2.5)
rows = [('Temperature', 'apparent T fitted from token log-probs'), ('fp8', 'likelihood ratio base vs. simulated fp8 weights'),
        ('LoRA', 'stop-token hazard, top-1 excess'), ('top-k / top-p', 'no token of rank ≥ 50 ever sampled')]
for i, (a, b) in enumerate(rows):
    box(s, 0.6, 4.3 + i * 0.55, 2.6, 0.45, LIGHT, a, size=16, color=PINK)
    text(s, 3.4, 4.3 + i * 0.55, 9, 0.45, b, size=17, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 6.55, 12, 0.5, 'Every detector validated first on the 9 reference endpoints whose variants are known.', size=15, color=GREY)

s = page('Leaderboard as an experiment', 'Part 2', PINK)
pic(s, 't2_scores.png', 0.3, 1.3, w=7.5)
bullets(s, 8.0, 1.5, 5.0, ['Black-box text distances: stuck at 0.10', 'White-box fingerprints: 1.00',
                            'One endpoint changed per submission', 'A drop rejects a candidate (ep-26, ep-25)',
                            'Ties never used as proof: weights decide'], size=19, dot=PINK)

s = page('Final answer: 3 weight groups', 'Part 2', PINK)
fams = [('A · Qwen2.5-3B', '04 · 12 · 14 · 23 · 33'), ('B · Qwen2.5-1.5B', '06 · 07 · 08 · 20 · 30 · 32'), ('C · Llama-3.2-1B', '10 · 13 · 16 · 27 · 29 · 31')]
for i, (t, e) in enumerate(fams):
    box(s, 0.6 + i * 4.1, 1.6, 3.8, 2.6, PINK, f'{t}\n\n{e}', size=24)
text(s, 0.6, 4.5, 12.1, 1.6, 'Each group = original + versions that only change decoding (watermark, persona, T=1.0, system prompt, top-k/p).\n'
     'All other endpoints (LoRA, fp8, …) stay alone.', size=20, align=PP_ALIGN.CENTER)
code(s, 'task2/loop/  (wb_*.py fingerprints, score_wb.py, sub/final_s15_plus12.csv)', PINK)

# ---------- Part 3 ----------
section(3, 'Unlearning', GREEN, 'Make a ViT forget 3 500 images, as if it never saw them.', '0.112')

s = page('The problem', 'Part 3', GREEN)
pic(s, 't3_forget.png', 1.3, 1.2, h=3.85)
bullets(s, 0.6, 5.25, 12.2, ['Target: behave like a model retrained from scratch without these images (the "gold" model)',
                            'Low accuracy alone is not forgetting; keep ≥ 70 % accuracy on held-out data'], size=19, dot=GREEN)

s = page('Our idea: be uncertain, not wrong', 'Part 3', GREEN)
pic(s, 't3_idea.png', 0.6, 1.2, w=12.1)
bullets(s, 0.6, 5.3, 12.2, ['The gold model never saw the images, so it is unsure on them, not confidently wrong',
                            'We lower confidence on the 9 forget classes and lightly reshape 36 related classes'], size=19, dot=GREEN)

s = page('Result: softening beats deleting', 'Part 3', GREEN)
pic(s, 't3_scores.png', 0.4, 1.2, h=5.3)
bullets(s, 8.2, 1.6, 4.8, ['Deleting classes: 0.015', 'Spreading probability: 0.112 (×7.7)',
                            'Every change checked on a held-out split with a local simulator'], size=20, dot=GREEN)
code(s, 'Task 3 - Unlearning/  (README + solution)', GREEN)

# ---------- Wrap-up ----------
s = page('Takeaways')
for i, (c, t, d) in enumerate([(BLUE, 'Watermarks', 'A decoder that only says "yes/no"\nbecomes a localiser'),
                               (PINK, 'Model equality', 'Open weights turn text into\nmeasurable fingerprints'),
                               (GREEN, 'Unlearning', 'Imitate the retrained model:\nuncertain, not wrong')]):
    box(s, 0.6 + i * 4.1, 1.7, 3.8, 3.0, c, f'{t}\n\n{d}', size=22)
text(s, 0.6, 5.3, 12.1, 1, 'Thank you!  ·  github.com/talal95c/CISPA-Paris-', size=26, bold=True, align=PP_ALIGN.CENTER)

P.save('Teletubbies_CISPA_Paris_2026.pptx'); print('ok', len(P.slides))
