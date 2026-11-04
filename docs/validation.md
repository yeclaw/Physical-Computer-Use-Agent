# Source-candidate validation

Validated locally on 2026-11-04 with Python 3.14.3, macOS 26.6.2 arm64,
setuptools 81.0.0, Pillow 12.2.0 and NumPy 2.4.3. Python 3.11 syntax parsing
also passed; this is not a claim of running the suite on every supported Python.

- Portable suite: **844 tests, 843 passed, one skipped**. The skip checks a
  historical private run's token trace that is deliberately not distributed.
- Independent review: **43 adversarial checks passed**, included in the suite.
  These cover pure preflight rejection, dispatch exceptions/unknown outcomes,
  non-replay, interpreter-error recovery, cleanup failures, provider startup
  logging, arbitrary camera views and historical-evidence access.
- All **113 physical-keyboard submodules** imported with `python -S`, without
  robot SDKs, Pillow or NumPy. Layout/dependency validation returned no errors.
- The full authored mock sequence ran from an independent source directory and
  through the installed console command from an empty working directory. The
  latter used an isolated virtual environment with no optional dependencies and
  no source-checkout fallback.
- A wheel was built and installed with networking disabled and no dependencies.
- The ledger-only smoke passed its 33 checks, eight authored trials, promotion,
  rollback and reopen checks.

All checks used synthetic decisions/devices or read-only source analysis. No
live model, robot, SDK, network service or OS keyboard injection was exercised.
These checks validate software behavior; they do not demonstrate autonomous
learning efficacy, physical safety, cross-device transfer or lower model cost.

`SOURCE_MANIFEST.json` identifies the exact exported files and source origins.
It also lists excluded private/device test boundaries and the narrowly adapted
fixtures; it does not imply that the excluded device implementation was tested.
The implementation validation predates the project metadata update. The
project is now maintained as Physical Computer Use Agent by yeclaw under Apache-2.0;
the author/license/citation metadata is checked separately from these core
behavior tests. GitHub-hosted CI is not claimed by this local validation.

## Final release audit

The updated portable suite completed **855 tests: 854 passed and one historical
private-evidence check skipped**. This includes five external-factory cases and
six synthetic credential-redaction regressions. A separate 67-test fake
Codex-client regression run also passed. Tests made no live model, device or
personal credential requests.

A fresh base-only wheel installation ran the offline repair/save/reuse fixture
from an empty working directory without a source-tree fallback. External factory
examples covered different geometry, zero/one/two declared views, scoped
knowledge and unknown-outcome non-replay. These are software integration checks;
real new hardware and autonomous cross-device transfer remain unverified.

An earlier baseline run had local Unix socket bind errors under its sandbox;
those affected cases passed when local IPC was permitted. The final 855-test run
used that local-IPC context and completed as a single suite. No cloud CI result
or actual authentication success is implied by these local checks.
