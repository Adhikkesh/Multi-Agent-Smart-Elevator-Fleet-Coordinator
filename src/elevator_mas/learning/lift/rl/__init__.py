"""Phase 5 — cooperative PPO for the LiftZero bidder, trained on the real simulator.

Modules: ``reward`` (team reward), ``privileged`` (critic-only state), ``critic``,
``gae`` (SMDP advantages), ``policy`` (masked categorical), ``anchors`` (KL to the BC policy),
``curriculum`` (training-building sampler with a no-test-seed guard), ``collector``
(rollout workers), ``ppo`` (the trainer), ``stats`` (paired statistics), ``eval_loop``.
"""
