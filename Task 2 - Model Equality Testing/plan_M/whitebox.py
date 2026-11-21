"""Offline likelihood scoring for matched prompt/response token sequences."""

import random
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F
from transformers.pytorch_utils import Conv1D
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def fake_e4m3(values: torch.Tensor, reduce_dim: int) -> torch.Tensor:
    """Round in scaled E4M3FN, then dequantize to the compute dtype."""
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = (values.abs().amax(dim=reduce_dim, keepdim=True) / limit).clamp_min(1e-30)
    quantized = (values / scale).clamp(-limit, limit).to(torch.float8_e4m3fn)
    return quantized.to(values.dtype) * scale


class DynamicFP8Linear(nn.Linear):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(fake_e4m3(inputs, reduce_dim=-1), self.weight, self.bias)


def make_replicas(checkpoint: nn.Module, scheme: str = "w8a16") -> tuple[nn.Module, nn.Module]:
    """Emulate bf16 storage and E4M3FN without native FP8 kernels.

    Matrix arithmetic uses fp32. Projection weights use per-output-channel
    scaling. W8A8 additionally quantizes each activation row at inference.
    Embeddings, the language head, and normalization stay unquantized in FP8.
    """
    if scheme not in ("w8a16", "w8a8"):
        raise ValueError(f"Unknown FP8 scheme: {scheme}")
    baseline = deepcopy(checkpoint).float().eval()
    with torch.no_grad():
        for weight in baseline.parameters():
            weight.copy_(weight.to(torch.bfloat16))
    replica = deepcopy(baseline)
    head = replica.get_output_embeddings()
    for name, layer in list(replica.named_modules()):
        if layer is head or not isinstance(layer, (nn.Linear, Conv1D)):
            continue
        weight = layer.weight.T if isinstance(layer, Conv1D) else layer.weight
        quantized = fake_e4m3(weight.detach(), reduce_dim=-1)
        cls = DynamicFP8Linear if scheme == "w8a8" else nn.Linear
        out_features, in_features = quantized.shape
        new = cls(in_features, out_features, bias=layer.bias is not None, device="meta")
        new.weight = nn.Parameter(quantized, requires_grad=False)
        if layer.bias is not None:
            new.bias = layer.bias
        replica.set_submodule(name, new)
    return baseline, replica


@dataclass(frozen=True)
class Completion:
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


def token_completions(
    tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], answers: Sequence[str]
) -> list[Completion]:
    """Encode the exact scored prompt and visible response separately."""
    if len(prompts) != len(answers):
        raise ValueError("Prompts and answers must have equal length")
    prompt_ids = tokenizer(list(prompts), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(list(answers), add_special_tokens=False)["input_ids"]
    return [Completion(tuple(p), tuple(a)) for p, a in zip(prompt_ids, answer_ids, strict=True)]


@dataclass(frozen=True)
class Decode:
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self):
        if not isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Require a positive finite temperature and top_p in (0, 1]")


def token_logp(logits: torch.Tensor, ids: torch.Tensor, decode: Decode) -> torch.Tensor:
    """Gather target-token log probabilities under the specified decode law."""
    scaled = logits.float() / decode.temperature
    if decode.top_p < 1:
        sorted_logits, sorted_ids = scaled.sort(descending=True, stable=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        # Include the token that crosses the nucleus threshold.
        dropped = F.pad(cumulative[..., :-1] >= decode.top_p, (1, 0), value=False)
        scaled = torch.empty_like(scaled).scatter_(
            -1, sorted_ids, sorted_logits.masked_fill(dropped, -torch.inf)
        )
    chosen = scaled.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return chosen - scaled.logsumexp(-1)


def _completion_logits(
    model: nn.Module,
    completions: Sequence[Completion],
    batch_size: int,
):
    if batch_size < 1 or any(not c.prompt_ids or not c.answer_ids for c in completions):
        raise ValueError("Require positive batch size and nonempty prompts and answers")
    device = next(model.parameters()).device
    for start in range(0, len(completions), batch_size):
        batch = completions[start : start + batch_size]
        lengths = [len(c.prompt_ids) + len(c.answer_ids) - 1 for c in batch]
        inputs = torch.zeros((len(batch), max(lengths)), dtype=torch.long, device=device)
        mask = torch.zeros_like(inputs)
        for row, item in enumerate(batch):
            inputs[row, :lengths[row]] = torch.tensor(
                item.prompt_ids + item.answer_ids[:-1], device=device
            )
            mask[row, :lengths[row]] = 1
        logits = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits
        for row, item in enumerate(batch):
            offset = len(item.prompt_ids) - 1
            labels = torch.tensor(item.answer_ids, device=device)
            yield logits[row, offset:lengths[row]], labels


@torch.inference_mode()
def completion_token_logp(
    model: nn.Module,
    completions: Sequence[Completion],
    decode: Decode = Decode(),
    batch_size: int = 4,
) -> list[torch.Tensor]:
    """Teacher-force only answer tokens. Returned vectors live on CPU."""
    return [
        token_logp(logits, labels, decode).cpu()
        for logits, labels in _completion_logits(model, completions, batch_size)
    ]


@torch.inference_mode()
def fit_decode(
    model: nn.Module,
    calibration: Sequence[Completion],
    candidates: Sequence[Decode],
    batch_size: int = 4,
) -> tuple[Decode, dict[Decode, float]]:
    """Fit one decode law on an independent calibration split.

    A fixed -30 nats floor keeps a rare out-of-nucleus visible token from
    invalidating a whole candidate. Use a shared fitted law for both sides.
    """
    if not calibration or not candidates:
        raise ValueError("Require calibration completions and decode candidates")
    totals = torch.zeros(len(candidates), device=next(model.parameters()).device)
    for logits, labels in _completion_logits(model, calibration, batch_size):
        for index, decode in enumerate(candidates):
            totals[index] += token_logp(logits, labels, decode).clamp_min(-30).sum()
    evidence = dict(zip(candidates, totals.tolist(), strict=True))
    return max(evidence, key=evidence.__getitem__), evidence


def checkpoint_evidence(
    candidates: Iterable[tuple[str, nn.Module, PreTrainedTokenizerBase]],
    prompts: Sequence[str],
    answers: Sequence[str],
    batch_size: int = 4,
) -> list[tuple[str, float]]:
    """Rank local checkpoints by visible-answer log likelihood per UTF-8 byte.

    Use each candidate's tokenizer with the same plain prompt strings. The
    winner's margin requires calibration for the intended checkpoint pool.
    """
    byte_count = sum(len(answer.encode("utf-8")) for answer in answers)
    if not byte_count:
        raise ValueError("Require visible answer text")
    ranking = []
    for name, model, tokenizer in candidates:
        completions = token_completions(tokenizer, prompts, answers)
        logps = completion_token_logp(model, completions, batch_size=batch_size)
        ranking.append((name, sum(float(tokens.sum()) for tokens in logps) / byte_count))
    return sorted(ranking, key=lambda candidate: candidate[1], reverse=True)


def paired_llr(
    baseline: nn.Module,
    replica: nn.Module,
    left: Sequence[Completion],
    right: Sequence[Completion],
    decode: Decode = Decode(),
    clip: float = 3.0,
    batch_size: int = 4,
) -> torch.Tensor:
    """Return per-prompt LLR(left) − LLR(right), in nats.

    A positive sum favors FP8 on the left. Both candidates use the same
    decode law. If both replicas exclude a token from their nucleus, it
    supplies zero evidence. Unequal answer lengths are scored as observed.
    """
    if not left or len(left) != len(right) or any(
        a.prompt_ids != b.prompt_ids for a, b in zip(left, right, strict=True)
    ):
        raise ValueError("Require nonempty candidates with identical ordered prompt IDs")
    if not isfinite(clip) or clip <= 0:
        raise ValueError("The per-token clip must be positive and finite")
    completions = [*left, *right]
    base_logp = completion_token_logp(baseline, completions, decode, batch_size)
    fp8_logp = completion_token_logp(replica, completions, decode, batch_size)
    scores = []
    for base, fp8 in zip(base_logp, fp8_logp, strict=True):
        delta = torch.where(torch.isneginf(base) & torch.isneginf(fp8), 0, fp8 - base)
        scores.append(delta.clamp(-clip, clip).sum())
    return torch.stack(scores[:len(left)]) - torch.stack(scores[len(left):])


def paired_null_pvalue(margins: Sequence[float], draws: int = 4095, seed: int = 17) -> float:
    """Two-sided randomization p-value under within-prompt exchangeability.

    Use for independent prompts and an identical decoding law. This only
    tests a shared score distribution. LoRA or other mismatched conditions
    can reject the null, so this p-value does not prove either side is FP8.
    """
    if not margins or draws < 1 or not all(isfinite(value) for value in margins):
        raise ValueError("Require finite margins and at least one randomization draw")
    observed = abs(sum(margins))
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(draws):
        randomized = sum(value if rng.getrandbits(1) else -value for value in margins)
        exceedances += abs(randomized) >= observed
    return (1 + exceedances) / (1 + draws)
