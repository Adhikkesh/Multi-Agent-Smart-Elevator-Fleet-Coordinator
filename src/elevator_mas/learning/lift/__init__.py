"""LiftZero: the learned Contract Net bidder (Phase 4 imitation; Phases 5-6 build on it).

Modules that need PyTorch (``model``, ``losses``, ``augment``, ``train_bc``, ``export``,
``eval_offline``) are training-time only. The runtime path — ``runtime`` (numpy +
onnxruntime), ``bidder`` and ``card`` — imports neither torch nor onnx, so a plain
``uv sync`` can run the ``liftzero_bc`` strategy.
"""
