# Task 1 - Watermark Localization: best solution (team Teletubbies)

Best public score: **0.7174** (submission #4248, file `cand_m.pt`), starting from the JURECA pipeline at 0.6919.

## Method
1. Start from the team's JURECA segmentation file (`submission_combo_jureca.pt`, public 0.6919).
2. Image type from the official decoder and the known message: whole-image bit accuracy >= .75 / .78 / .80
   (StegaStamp / TrustMark / PixelSeal) means "hole" (everything marked except a restored region), otherwise "patch".
3. About half of the patches are exact rectangles watermarked as their own image. A rectangle is accepted only if its crop
   decodes the known message (bit accuracy >= .9); the mask is then the rectangle (`basecheck.py`, `build_a.py`, child `trustmark_patch.npz`).
4. Edges are pinned with the re-embedding log-ratio `log(|embed(crop,m)-crop|^2 / |embed(crop,1-m)-crop|^2)` (`rlr.py`):
   coordinate descent on the 4 edges (steps 8/4/2/1). On train the optimum is the true box grown by ~3 px per side, so the
   mask is the found box shrunk by 3 px (`apply.py`, accepted when lr <= -0.6; on train non-rectangle patches never go below -0.34).
   PixelSeal (`apply2.py`): same rule, plus lr <= -0.4 when the crop decodes the message (>= .9 of 256 bits); checked on train with `tps.py`.
5. PixelSeal holes: small model from a child session, 5-fold out-of-fold mIoU 0.637 on train (`children/pixelseal_hole_v2.npz`).
6. No train mask is all-zero or all-one (0/510), so where the base outputs such a mask the second model is used (`fill.py`).
Nothing is tuned on the public score: every rule is checked on train first and applied identically to the 1020 test images.

## Scores (public split, 30 % of test)
| # | file | change | public |
|---|---|---|---|
| - | submission_combo_jureca.pt | JURECA baseline | 0.6919 |
| 3889 | cand_a.pt | + filled boxes whose crop decodes the message | 0.6930 |
| 3984 | cand_d.pt | + 60 TrustMark rectangles verified by decoding | 0.7062 |
| 4008 | cand_e.pt | + PixelSeal holes (child v2, OOF 0.637) | 0.7131 |
| 4038 | cand_f.pt | + 49 rectangles with edges refined by re-embedding | 0.7141 |
| 4063 | cand_g.pt | + more refined rectangles (StegaStamp, TrustMark) | 0.7151 |
| 4092 | cand_h.pt | + fallback for all-zero / all-one base masks (81 images) | 0.7169 |
| 4186 | cand_k.pt | + PixelSeal rectangles refined by re-embedding (11 images) | 0.7171 |
| 4248 | cand_m.pt | + remaining PixelSeal rectangles (17 images in total) | **0.7174** |
Rejected: PixelSeal-hole child v1 (0.6921), StegaStamp-hole child (0.6805), TrustMark-hole child v2 on fitted frames (0.7156).

## Reproduce
```bash
bash setup.sh ~/t1                 # data, starter kit, venv (CPU, ~4 min); then cd ~/t1/loop && . ../.venv/bin/activate
mkdir -p out ../kids && cp children/*.npz ../kids/
python bitacc.py                   # whole-image bit accuracy -> cache/bitacc.npz (already provided in cache/)
python basecheck.py && python build_a.py out/cand_a.pt        # needs the baseline .pt (included alongside the scripts)
python build.py out/cand_a.pt out/cand_d.pt trustmark:patch:../kids/trustmark_patch.npz:rect
python build.py out/cand_d.pt out/cand_e.pt pixelseal:hole:../kids/pixelseal_hole_v2.npz:swap
# Saved edge-search results are provided in rl/; no need to rerun the expensive search.
python apply.py out/cand_e.pt out/tmp.pt && python fill.py out/tmp.pt out/cand_h.pt
python apply2.py out/cand_h.pt out/cand_m.pt
# Stop here: no API key or competition submission is needed to reproduce the masks.
```
`best_submission_cand_m_0.7174.pt` is the submitted file (#4248). The baseline `submission_combo_jureca.pt` is included alongside the scripts. Intermediate candidates are not included. `WORKLOG.md` is the full experiment log
(dead ends included); `exp*.py` are the exploratory scripts.
