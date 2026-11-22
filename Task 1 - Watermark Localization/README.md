# Task 1 — Watermark Localization (Teletubbies)

Best public mIoU: **0.7141** (submission 4038).

| Step | Public mIoU | Submission |
| --- | --- | --- |
| Segmentation baseline trained on JURECA (`submission_combo_jureca.pt`) | 0.6930 | 3889 |
| + 60 TrustMark rectangles confirmed by decoding the known message | 0.7062 | 3984 |
| + improved PixelSeal hole filling | 0.7131 | 4008 |
| + 49 rectangles re-aligned to the pixel | 0.7141 | 4038 |

Idea: watermarks are embedded in rectangles; a candidate rectangle is kept only if decoding the watermark inside it
returns the known message, which makes the localisation robust instead of relying on the segmentation alone.
Code: [`solution/`](solution/) (pushed from the task-1 session).
