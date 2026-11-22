# Task 1 (Watermark Localization) — working notes (same scheme as task2/loop)

Team Teletubbies, public T1 = 0.69 at 07:20 CET (method unknown to Devin). Deadline 22/11 12:00 CET. NO submission without Talal's explicit OK.
Machine: CPU only (8 cores). Setup: `bash task1/loop/setup.sh` (~4 min) -> ~/t1 (release/, starter/, .venv, loop -> repo task1/loop).

## Task facts (PDF + starter kit)
- 512x512 PNG, predict 128x128 grid (cell=1 if >=50% of its 4x4 px watermarked). Score = mean over images of (IoU_0+IoU_1)/2 (empty union -> 1).
- train 510 (170/method) with mask, mask128, method, message, full_watermark; test 1020 (340/method) with method + message.
- Methods: pixelseal (256 bits, decoder input 256), trustmark Q (100 bits, 245, no ECC), stegastamp (100 bits, 400, ONNX).
- Submission: torch.save({"image_names": names, "masks": bool tensor [1020,128,128]}), <= 20 MB. Task id `34-watermark-localisation-image`,
  POST http://34.147.229.55/submit/34-watermark-localisation-image, header X-API-Key, field "file". Public = 30 % of test, private = 70 %.

## Data structure (measured on train, 08:55 CET)
- Exactly half of each method is full_watermark=True, half False (85/85).
- True  ("hole"): whole image watermarked, then a blob/object-shaped part restored to the clean original. Watermarked fraction: q10 .74, median .94, q90 .99.
- False ("patch"): a small rectangle (aspect up to 16:1; 2:1 for trustmark) or an object inside such a rectangle is watermarked; the patch was
  watermarked as its own image. Watermarked fraction: q10 .01, median .04-.05, q90 .16-.30, max .44. ~35-40 % of them are exact rectangles.
- Trivial scores: all-ones ~0.45 on True / ~0.03 on False; all-zeros ~0.03 on True / ~0.47 on False. Type decision then constant mask ~0.46.

## Measurements
- exp0 whole-image bit accuracy vs known message: stegastamp True q10 .90 med .99 (min .69) / False med .51 q90 .61 max .85;
  trustmark True q10 .94 med 1.0 (min .71) / False med .51 q90 .64 max .88 => type classifier = whole-image bit acc (threshold ~.75).
  CPU time: stegastamp .15 s/img, trustmark .09 s/img, pixelseal slow (~1 s?).
- exp1 matched filter (re-embed observed image with the message, correlate residual with high-pass image, local windows): AUC ~.60 stegastamp. Weak.
- exp2 per-position logit agreement maps (features()["per_position_logits"]): trustmark 8x8 whole image AUC ~.61 mean/.69 median; tiled crops
  (256/128 px) ~.5 => trustmark does not decode from sub-crops of its rectangle.

## Ideas not yet tested
- Synthetic data: False-type images are ~95 % clean -> embed ourselves (whole image + restore blobs; or a rectangle) => unlimited labelled data
  and access to the true residual. Train a per-cell classifier / small CNN on decoder features, or study which statistic separates.
- Occlusion / gradient saliency of the message margin (sum of sign(msg)*logit): remove the watermark locally (blur, denoise, trustmark remover) and
  watch the margin; or one backward pass: grad_I(margin) * (denoised - I).
- Patch type: search the rectangle whose crop decodes the message (bit acc ~1 when the crop == the watermarked rectangle), coarse-to-fine.

## 22 Nov, 09:00-09:30 (parent session)
- Team's current best file (public 0.69): `submission_combo_jureca.pt` (GPU pipeline on JURECA, see rapport.pdf; val mIoU 0.709: pixelseal .62, trustmark .671 [holes .594, patches .748], stegastamp .836). Used as BASE; we only override where we have evidence.
- Type classifier (bitacc.py, typeclf.py): whole-image bit accuracy >= .75/.78/.80 (stega/trust/pixel) -> hole, else patch. Train error 1.2-2.4 %. Constant mask by type = 0.4526 train mIoU.
- exp3: patch type has 2 sub-types. Exact rectangles (~half): crop = own watermarked image, decoding the GT box gives the message (trust .94, stega .81, pixel .69); object-shaped masks: bbox decode = chance, whole image slightly above chance (.55-.59) => cut out of a whole-image-watermarked copy.
- exp7: tolerance of crop decoding to box errors. trustmark wide basin (12 % shift -> .90, scale .9/1.1 ok, 1.2 dead); stegastamp very narrow (6 % shift -> chance); pixelseal wide and flat, only 40 % of rectangles decode >= .75.
- refine.py / exp8: coordinate descent on box edges maximising crop agreement (trustmark): verifies the rectangle but does NOT pin the edges (median max edge error 7 px, mIoU of refined box .913 vs .909 for a jittered start). Decoder objectives are flat near the optimum.
- basecheck.py: among predicted-patch test images, the base mask's bbox already decodes >= .9 for 37/168 trustmark, 31/164 pixelseal, 10/163 stegastamp (base empty: 29 / 64 / 3). Expected decodable rectangles ~72 trustmark => ~35 trustmark rectangles missing or badly placed in the base.
- Dead ends measured here: exp4 (re-embedding residual x high-pass image: residual correlates with content in clean regions too), exp6 (pixelseal detection channel and per-position maps at 256/512/768 input: AUC .55-.68), exp9 (trustmark official remover as content canceller: AUC .55-.64, the remover hallucinates a watermark in clean regions).
- exp5: re-embedding energy ratio |embed(X,msg)-X|^2 / |embed(X,random)-X|^2 is lower in marked regions: stegastamp holes region ratio .65 vs 1.02 (10/10 images), cell AUC .76 at 64-px windows; trustmark .85 vs 1.16, AUC .61. Weak per cell.

## 22 Nov, 09:40-09:55
- Talal approved submitting (09:40) and gave the team key in chat (do NOT write it to files; pass it as env K to submit.py). Server http://34.147.229.55, POST /submit/34-watermark-localisation-image, header X-API-Key, field "file". Response gives the PUBLIC score with 4 decimals + previous best + delta. Cooldown 10 min between submissions (429 with seconds to wait). Deadline 12:00 CET. Best T1 teams: 0.82.
- Submission 3889 (09:46): cand_a = base with mask := filled bbox of largest component where that bbox decodes >= .9 (78 images, 28 actually changed) -> 0.6930 public (base 0.6919, +0.0011).
- No T1 code or JURECA train/val predictions in the repo => base vs kid can only be compared through the public score (slice swaps; ~30 % of 1020 = ~306 public images, ~50 per method x type slice => noisy, deltas below ~0.002 are not evidence).
- Images are AI-generated (COCO-caption style), masks are WAM-style: holes = union of random blobs / half planes / object shapes; patches = boxes or object shapes. No JPEG 8x8 grid trace in clean regions (exp11). Rectangle coordinates are not on a grid.
- exp10 (message template matched filter on the full frame) AUC .46-.50; exp12 (pixelseal keep-only tile on gray/blur background, tiles 64/128/192) AUC .50-.66: dead ends.
- Kids, first held-out numbers: stegastamp hole U-Net on synthetic pairs .754-.760 (agreement with base on test .748); pixelseal hole .575 (agreement .569); pixelseal patch .547 OOF (agreement .577; its rect fields are NOT verified rectangles); stegastamp patch .60; trustmark hole: nothing above all-ones yet, suspects the watermark frame is not the 512 image for ~75 % of hole images (band-shaped re-embedding cue).
3917 cand_b (A + pixelseal hole kid .575) 0.6921 (A=0.6930): base pixelseal holes ~ kid

## 22 Nov, 10:00-10:08
- Submissions so far (public, 4 decimals; cooldown 10 min, next slot = last success + 600 s): 3889 cand_a 0.6930 (BEST, = base + filled decoded bboxes); 3917 cand_b (A + pixelseal-hole kid .575) 0.6921 => base pixelseal holes ~ .58; 3943 cand_c (A + stegastamp-hole kid .76) 0.6805 => base stegastamp holes ~ .83, keep base there.
- build.py BASE OUT meth:type:npz[:mode[:th]] does slice swaps (type by whole-image bit accuracy), cmp.py compares a kid npz with the base, submit.py (env K) submits and logs to submissions.log.
- FRAME FINDING (exp13/14/15): cue = log(|embed(X,m)-X|^2 / |embed(X,1-m)-X|^2), global and local (blur 64). stegastamp holes always aligned (global -.2..-1). pixelseal and trustmark holes: only ~25 % aligned to the 512x512 frame (global pixelseal -.4..-.5, trustmark -1..-1.6; there the local map has AUC .85-.93); the others ~0. Padding X by reflection to a TALLER frame (exp15, pixelseal): sharp peaks at aspect 1.10/1.14/1.20/1.286 with a specific offset (e.g. H1.286 centre -0.41; H1.10 offset .375 -0.60), wider frames never respond => the image is a 512x512 window of a taller watermarked frame (aspect ~1.0-1.6, varying offset). Per image search = aspect x offset, 2 embeds per hypothesis (~0.4 s each for pixelseal on 6 threads). Told the 4 pixelseal/trustmark kids at 10:02.

## 10:32 cand_e = cand_d + pixelseal hole child v2 (OOF 0.637) -> public 0.7131 (#4008). NEW BEST / base.
- rlr.py (re-embed lr) train: found box = GT expanded by 3 (sometimes 4-5) px/side when lr <= -0.6; mask = found box shrunk 3.
- 10:42 cand_f = cand_e + 49 lr-refined rects (lr<=-0.6, shrink 3) -> public 0.7141 (#4038). NEW BEST / base.
- 10:52 cand_g = cand_f + more refined rects incl. 8 stega kid rects (lr<=-.4 & acc>=.88) -> public 0.7151 (#4063). NEW BEST / base.
- trainobj check (non-rect patches, start GT bbox+3): final lr in [-0.34,-0.03] (11 imgs) -> lr<=-0.6 rule has no false positives; true rects -0.44..-1.5.
- 11:02 cand_h = cand_g + more rects + fill.py (pixelseal empty patch -> kid v2 mask (59), trustmark all-ones hole -> kid hole v2 (22)) -> public 0.7169 (#4092). NEW BEST / base.
cand_i (TM hole v2 where logratio<-1.4, 101 imgs): public 0.7156 (#4121) < cand_h 0.7169 -> rejected
- 11:33 cand_k = cand_h + apply2.py (pixelseal lr-refined rects: lr<=-0.6, or lr<=-0.4 with crop decode acc>=.9; 11 images changed) -> public 0.7171 (#4186). NEW BEST. start4 (10 pixelseal boxes with acc .69-.79) all end at lr > -0.10 -> not rectangles, rule does not fire.
- tps.py (train pixelseal, start = child decoded rect +3): found box = GT grown by 0-5 px/side (e04c lr -0.59: 0,0,4,1; c267 lr -0.62: 5,5,3,4; f916 lr -0.35: 4,4,3,2) -> shrink 3 valid for pixelseal too (<=2 px residual).
- 11:43 cand_l = cand_h + apply2 (13 pixelseal images changed) -> public 0.7171 (#4217), equal to cand_k.
- 11:53 cand_m = cand_h + apply2 (17 pixelseal images changed) -> public 0.7174 (#4248). FINAL BEST.
