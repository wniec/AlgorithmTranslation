"""Evaluating a trained translator, independent of any command line.

* ``heldout``  — one switch per held-out problem: translated vs lossy vs cold
  hand-off (used by ``uv run eval`` and the trainers' checkpoint evaluation);
* ``switcher`` — a random multi-switch schedule over the whole budget, five
  ways of running it (used by ``uv run eval-switch``);
* ``anytime``  — the anytime-performance metrics ``switcher`` reports (budget
  AOCC, ERT, average ranks).
"""
