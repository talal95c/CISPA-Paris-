import unittest
from types import SimpleNamespace

import torch
from torch import nn

from whitebox import (
    Completion,
    Decode,
    completion_token_logp,
    fake_e4m3,
    fit_decode,
    paired_llr,
    paired_null_pvalue,
    token_logp,
)


class TableLM(nn.Module):
    """Each input token indexes the logits of the following token."""

    def __init__(self, weights):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights).log(), requires_grad=False)

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(logits=self.weights[input_ids])


class WhiteboxTest(unittest.TestCase):
    def setUp(self):
        self.base = TableLM([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5], [0.1, 0.2, 0.7]])
        self.fp8 = TableLM([[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.1, 0.2, 0.7]])

    def test_only_answer_tokens_are_scored_in_ragged_batch(self):
        result = completion_token_logp(
            self.base, [Completion((2, 0), (1, 2)), Completion((1,), (1,))]
        )
        expected = [torch.tensor([0.2, 0.5]).log(), torch.tensor([0.4]).log()]
        for actual, known in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, known)

    def test_pair_direction_and_prompt_matching(self):
        left, right = [Completion((0,), (1,))], [Completion((0,), (0,))]
        delta = paired_llr(self.base, self.fp8, left, right)
        torch.testing.assert_close(delta, torch.tensor([2.8]).log())
        torch.testing.assert_close(paired_llr(self.base, self.fp8, right, left), -delta)
        with self.assertRaises(ValueError):
            paired_llr(self.base, self.fp8, left, [Completion((1,), (0,))])

    def test_nucleus_and_clip(self):
        logits = torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]]).log()
        actual = token_logp(logits, torch.tensor([1, 2]), Decode(1.0, 0.7))
        torch.testing.assert_close(actual[0], torch.tensor(1 / 3).log())
        self.assertTrue(torch.isneginf(actual[1]))
        margin = paired_llr(
            self.base, self.fp8,
            [Completion((0,), (1,))], [Completion((0,), (2,))],
            Decode(1.0, 0.6), clip=0.125,
        )
        torch.testing.assert_close(margin, torch.tensor([0.125]))

    def test_decode_fit(self):
        cold, warm = Decode(0.5), Decode(1.5)
        data = [Completion((0,), (0,))] * 4
        fitted, evidence = fit_decode(self.base, data, [cold, warm])
        self.assertEqual(fitted, cold)
        self.assertGreater(evidence[cold], evidence[warm])

    def test_fp8_scaling(self):
        values = torch.tensor([[0.0, 0.0], [448.0, 239.1], [0.448, 0.2391]])
        expected = torch.tensor([[0.0, 0.0], [448.0, 240.0], [0.448, 0.240]])
        torch.testing.assert_close(fake_e4m3(values, -1), expected)

    def test_sign_flip_null(self):
        self.assertEqual(paired_null_pvalue([0.0] * 10), 1.0)
        self.assertLess(paired_null_pvalue([1.0] * 10), 0.05)
        self.assertEqual(paired_null_pvalue([1.0] * 10), paired_null_pvalue([-1.0] * 10))


if __name__ == "__main__":
    unittest.main()
