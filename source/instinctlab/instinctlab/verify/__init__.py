"""Verification that runs off the training hot path.

:mod:`~instinctlab.verify.scene` is the engine-neutral scene the sim2sim harness and the backend
pin tests build from, so that "the same robot in two simulators" is one description rather than one
per test.

``structure.py`` lived here too: it flattened a compiled config and diffed it against a
hand-written one, which was how the Isaac backend used to be accepted. It went with D3, along with
``check_parity.py`` and ``dump_golden.py``, its only two callers.

Nothing here is imported by a training run.
"""
