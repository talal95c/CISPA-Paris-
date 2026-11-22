# Task 1 · Watermark Localization

**Best public mIoU: 0.7174** (submission **4248**). [Final submitted masks](solution/best_submission_cand_m_0.7174.pt) · [Code and reproduction guide](solution/NOTES.md) · [Official statement](Watermark%20Localization.pdf)

| Selected milestone | Public mIoU |
| --- | ---: |
| [JURECA segmentation baseline](solution/submission_combo_jureca.pt) | 0.6919 |
| Confirm TrustMark rectangles by decoding the known message | 0.7062 |
| Improve PixelSeal holes | 0.7131 |
| Replace empty or full masks after realigning edges | 0.7169 |
| Refine 17 PixelSeal rectangles in total | **0.7174** |

The segmentation gives a rough outline. A matching decoded message is independent evidence that a crop really contains the watermark. We then [refine its edges by re-embedding](solution/rlr.py) and [apply the rule to the final masks](solution/apply2.py). The [experiment notes](solution/NOTES.md) record intermediate scores and rejected variants.
