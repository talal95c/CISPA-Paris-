# Task 3 — Unlearning (Teletubbies)

Best public score: **0.1122**.

| Variant | Public |
| --- | --- |
| Remove the 9 forget classes outright | 0.0146 |
| Moderate confidence drop on the 9 forget classes (130 images each) | 0.0946 |
| + light reshaping of 36 related classes | 0.1108 |
| Same, tuned on a held-out split | **0.1122** |

Idea: the score compares our outputs with a gold model retrained without the forget set (KL at T=2). That gold model
is *uncertain* on forgotten images rather than confidently wrong, so we spread the probability mass instead of
deleting classes, and validate every change on a held-out split with a local simulator.
Code: [`solution/`](solution/) (pushed from the task-3 session).
