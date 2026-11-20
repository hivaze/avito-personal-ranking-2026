"""Self-supervised trajectory encoder: corpora, model, training, scoring and the inputs made from its scores.

Order of work: ``corpus`` (CPU) -> ``train`` (GPU) -> ``score`` (GPU) -> ``inputs`` (CPU).
"""
