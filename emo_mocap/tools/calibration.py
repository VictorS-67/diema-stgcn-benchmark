"""Confidence calibration: temperature scaling and calibration error.

A trained classifier is usually a good *ranker* and a bad *probability
estimator*. It gets the order of the classes right far more often than it
gets "I am 90% sure" right — and the longer it trains, the more overconfident
it becomes, because cross-entropy keeps rewarding a larger gap between the
winning logit and the rest long after the winner has stopped changing.

Temperature scaling is the cheapest known fix: divide every logit by one
positive scalar ``T`` before the softmax.

    p_i = exp(z_i / T) / sum_j exp(z_j / T)

``T > 1`` flattens the distribution (less confident), ``T < 1`` sharpens it.
Because dividing every logit by the same positive number cannot change which
one is largest, **accuracy and F1 are provably unchanged** — only the
probabilities move. That is what makes it safe to apply to an already-trained
model: it can improve the calibration numbers and cannot damage the
discrimination ones.

``T`` is not chosen by hand. It is fitted on a held-out split by minimising
cross-entropy, which is a *strictly proper* scoring rule — it is minimised
only by the true probabilities, so it cannot be gamed by a model that hedges.

Why this lives in the library rather than in a script: the training loop logs
a temperature-corrected validation loss every epoch, and
``scripts/checkpoint_diagnostics.py`` scores saved checkpoints the same way.
Two copies of a metric drift apart; this is the one definition.
"""

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["fit_temperature", "expected_calibration_error"]


def fit_temperature(logits, labels, max_iter=100):
    """Fit the temperature that minimises cross-entropy on this split.

    The optimisation is over ``log T`` rather than ``T`` so that the
    temperature stays positive without a constrained solver — any real
    ``log T`` maps to a valid positive ``T``. With a single parameter and a
    convex-in-practice objective, LBFGS converges in a few dozen iterations.

    Fit this on **validation** data only. Fitting on test is leakage, and
    fitting on train is pointless: the model is near-interpolating there, so
    the loss is already tiny and the fitted ``T`` says nothing about how the
    confidences behave on data the model has not seen.

    Returns 1.0 (i.e. "leave the logits alone") if the fit fails to beat the
    unscaled loss, so a degenerate or tiny split cannot make things worse.

    Args:
        logits: (N, C) unnormalised scores. Detached; no grad flows to the model.
        labels: (N,) integer class indices.
        max_iter: LBFGS iteration cap.

    Returns:
        float: the fitted temperature.
    """
    logits = torch.as_tensor(logits).detach().float()
    labels = torch.as_tensor(labels).detach().long()
    if logits.numel() == 0 or logits.shape[0] != labels.shape[0]:
        return 1.0

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    # Validation runs under no_grad/inference_mode; the fit needs autograd.
    with torch.enable_grad():
        opt.step(closure)

    t = float(log_t.exp().item())
    if not np.isfinite(t) or t <= 0:
        return 1.0
    base = F.cross_entropy(logits, labels).item()
    scaled = F.cross_entropy(logits / t, labels).item()
    return t if scaled <= base else 1.0


def expected_calibration_error(probs, labels, n_bins=15):
    """Expected calibration error: mean |confidence - accuracy| across bins.

    Sorts predictions into equal-width bins by the confidence of the winning
    class, then asks, per bin, whether the model was right as often as it
    claimed. A perfectly calibrated model scores 0.

    ECE is a *reporting* metric, not a selection one — it is binned, and it is
    not strictly proper: a model that answers the class prior for every sample
    scores near-perfectly on it while being useless. Empty bins contribute
    nothing rather than counting as perfectly calibrated.

    Args:
        probs: (N, C) probabilities that sum to 1 along the class axis.
        labels: (N,) integer class indices.
        n_bins: number of equal-width confidence bins.

    Returns:
        float: the ECE in [0, 1].
    """
    probs = torch.as_tensor(probs).detach().float()
    labels = torch.as_tensor(labels).detach().long()
    if probs.numel() == 0:
        return 0.0

    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.float().mean().item() * abs(
                correct[sel].mean().item() - conf[sel].mean().item())
    return total
