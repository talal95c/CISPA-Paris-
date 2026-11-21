import unittest

import numpy as np
import torch

import sim


class SimulatorTests(unittest.TestCase):
    def test_bank_and_burst(self):
        bank = sim.make_bank()
        self.assertEqual(bank, sim.make_bank())
        self.assertEqual(len({x['code'] for x in bank}), 300)
        self.assertEqual({k: sum(x['kind'] == k for x in bank) for k in range(6)}, dict.fromkeys(range(6), 50))
        self.assertTrue(all(x['answer'] in x['options'] for x in bank))
        self.assertEqual(sim.burst(bank[:22]).count('Options:'), 22)
        self.assertEqual(sim.make_bank(count=0), [])
        with self.assertRaises(ValueError):
            sim.burst([])

    def test_probability_and_watermark(self):
        logits = torch.zeros(100)
        p = sim.probabilities(logits, [2, 3], 7, 'plain')
        np.testing.assert_allclose(p, [.01, .01, .98], atol=1e-6)
        self.assertEqual(len(sim.greenlist(100, 7)), 25)
        self.assertTrue(torch.equal(sim.greenlist(100, 7), sim.greenlist(100, 7)))
        self.assertFalse(torch.equal(sim.greenlist(100, 7), sim.greenlist(100, 8)))
        p = sim.probabilities(logits, list(range(100)), 7, 'watermark')
        self.assertAlmostEqual(p.sum(), 1., places=6)
        self.assertGreater(p[sim.greenlist(100, 7)].mean(), p[:100].mean())

    def test_weights_restore_even_on_error(self):
        model = torch.nn.Module()
        model.q_proj = torch.nn.Linear(8, 8, bias=False)
        original = model.q_proj.weight.detach().clone()
        direction = sim.directions(model)
        with sim.weights(model, 'lora1.5', direction, .05):
            self.assertFalse(torch.equal(original, model.q_proj.weight))
        self.assertTrue(torch.equal(original, model.q_proj.weight))
        with self.assertRaises(RuntimeError):
            with sim.weights(model, 'fp8', direction, .05):
                self.assertFalse(torch.equal(original, model.q_proj.weight))
                raise RuntimeError('test')
        self.assertTrue(torch.equal(original, model.q_proj.weight))
        with self.assertRaises(ValueError):
            sim.directions(torch.nn.Linear(8, 8))

    def test_replay_budget_and_errors(self):
        a = np.tile([1., 0.], (30, 1))
        b = np.tile([0., 1.], (30, 1))
        np.testing.assert_array_equal(sim.replay(a, a, repeats=2), [0, 0, 0])
        np.testing.assert_array_equal(sim.replay(a, b, repeats=2), [1, 1, 1])
        with self.assertRaises(ValueError):
            sim.replay(a[:0], b[:0])
        with self.assertRaises(ValueError):
            sim.replay(a, b[:2])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
