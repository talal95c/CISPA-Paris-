"""Builds Teletubbies_CISPA_Paris_2026.pptx (python-pptx)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

P = Presentation(); P.slide_width, P.slide_height = Inches(13.333), Inches(7.5)
DARK, ACC = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0xE0, 0x6C, 0x9F)

def slide(title, bullets=None, table=None, sub=None):
    s = P.slides.add_slide(P.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12), Inches(1)).text_frame
    tb.text = title; r = tb.paragraphs[0].runs[0]; r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = DARK
    if sub:
        p = tb.add_paragraph(); p.text = sub; p.runs[0].font.size = Pt(18); p.runs[0].font.color.rgb = ACC
    y = Inches(1.7)
    if bullets:
        tf = s.shapes.add_textbox(Inches(0.8), y, Inches(11.8), Inches(5)).text_frame; tf.word_wrap = True
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            lvl = b.startswith('  '); p.text = ('– ' if lvl else '• ') + b.strip(); p.level = int(lvl)
            p.runs[0].font.size = Pt(18 if lvl else 22); p.space_after = Pt(8)
        y = Inches(1.7 + 0.55 * len(bullets))
    if table:
        rows, cols = len(table), len(table[0])
        t = s.shapes.add_table(rows, cols, Inches(0.8), y, Inches(11.8), Inches(0.42 * rows)).table
        for i, row in enumerate(table):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = str(v)
                for p in c.text_frame.paragraphs:
                    for r in p.runs: r.font.size = Pt(15); r.font.bold = i == 0
    return s

slide('Teletubbies', sub='CISPA × Fraunhofer AISEC Hackathon — Paris 2026',
      bullets=['Task 1 Watermark Localization: 0.714 mIoU (public)',
               'Task 2 Model Equality Testing: 1.00 (public)',
               'Task 3 Unlearning: 0.112 (public)',
               'Code, notes and this deck: github.com/talal95c/CISPA-Paris-'])

slide('Task 2 — Which endpoints share the same weights?',
      bullets=['33 anonymous LLM endpoints, 9 labelled references, 100 answers max per endpoint',
               'Same model = same checkpoint + quantization + LoRA',
               'Disguises that do NOT change the model: temperature, top-k/top-p, system prompt, persona, watermark',
               'Black-box text distances (TF-IDF, MMD, k-NN): stuck at 0.10 — they confuse temperature with weights'])

slide('Key idea: go white-box',
      bullets=['Endpoints self-identify as Qwen / Llama → found the 4 public checkpoints',
               'Re-score every collected answer token by token under the true weights (teacher forcing)',
               'Each modification leaves a fingerprint, validated first on the 9 references'],
      table=[['Family', 'Checkpoint', '# endpoints'],
             ['A', 'Qwen2.5-3B-Instruct', 10], ['B', 'Qwen2.5-1.5B-Instruct', 12],
             ['C', 'Llama-3.2-1B-Instruct', 11], ['refs', 'SmolLM2-360M-Instruct', 9]])

slide('Fingerprints of each modification', table=[
    ['Modification', 'Signal', 'Found'],
    ['Temperature', 'fitted apparent T (default 0.70)', '33, 30, 31 at T≈1.0'],
    ['fp8', 'LLR base vs simulated E4M3 weights', '15, 22'],
    ['Watermark', 'identical openings when re-asked', '04, 06, 10'],
    ['System prompt', 'recovered prompt + re-scoring', '29, 08'],
    ['top-k / top-p', 'no token of rank ≥ 50; exact T×k×p fit', '07, 27, 14'],
    ['LoRA', 'stop hazard, entropy-split T, top-1 excess', 'kept alone'],
    ['Original', 'perfect fit at T 0.69, zero top-1 excess', '12, 32, 13']])

slide('Leaderboard as an experiment, not a target',
      bullets=['One change per submission, expected score written down before',
               'Adding x to a correct group raises the score only if x truly shares the weights',
               'Drops/ties only exclude; keep a change only if a weight measurement also supports it'],
      table=[['Step', 'Public'], ['Black-box baselines', '0.10'], ['Core disguises', '0.45'],
             ['+ watermarks', '0.60'], ['+ system prompt ep-08', '0.70'], ['+ original ep-13 (C)', '0.85'],
             ['+ original ep-32 (B)', '1.00'], ['+ top-k/p 07, 27, 14 + original ep-12 (A)', '1.00']])

slide('Task 2 — Final answer', bullets=[
    'A: 04 wm · 12 original · 14 top-k/p · 23 persona · 33 T=1.0',
    'B: 06 wm · 07 top-k/p · 08 system prompt · 20 persona · 30 T=1.0 · 32 original',
    'C: 10 wm · 13 original · 16 persona · 27 top-k/p · 29 system prompt · 31 T=1.0',
    'All others alone: fp8 (15, 22) and LoRA-like variants',
    'CPU only (8 cores): white-box scoring fits small models without a GPU'])

slide('Task 1 — Watermark Localization', bullets=[
    'Masks of the watermarked region for 1 020 images (PixelSeal, TrustMark, StegaStamp), mIoU',
    'Segmentation baseline trained on JURECA: 0.693',
    'Confirm locations by decoding the known message: +60 TrustMark rectangles → 0.706',
    'Better PixelSeal hole filling → 0.713; 49 rectangles re-aligned to the pixel → 0.714'])

slide('Task 3 — Unlearning', bullets=[
    'Forget ~3 500 images of a ViT, match a gold retrained model (KL, T=2), keep retain outputs',
    'Hard deletion of forget classes: 0.015 — the gold model is uncertain, not wrong',
    'Moderate confidence drop on the 9 forget classes: 0.095',
    'Plus a light reshaping of 36 related classes, tuned on a held-out split: 0.112'])

slide('Takeaways', bullets=[
    'Open weights turn a black-box test into a likelihood test',
    'Validate every detector on known references before trusting it',
    'Use the public leaderboard for falsification, one hypothesis at a time'])

P.save('Teletubbies_CISPA_Paris_2026.pptx'); print('ok')
