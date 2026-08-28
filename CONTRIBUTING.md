# Contributing

Bug reports and reproducibility reports are welcome. A useful report includes:

1. the commit or release identifier;
2. operating system and Python version;
3. the exact configuration and command;
4. the deterministic instance identifier or seed; and
5. the observed output and expected behavior.

Changes to scheduling logic should add or update a test in `tests/test_core.py`.
Do not weaken domination, connectivity, disjointness, failure-universe, or
energy-accounting assertions to make a run pass. New comparator code must state
whether it is original author code, a faithful port, a reproduction, or a
reconstruction.

Large result contributions should be generated from a committed configuration
and accompanied by validation output. Please do not submit third-party article
PDFs, private review material, credentials, machine-specific paths, or
unlicensed baseline source code.
