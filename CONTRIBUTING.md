# Contributing

Keep new capabilities in their owning component: task/model decisions in
`agent`, execution/observation in `runtime`, persistent knowledge in `skills`,
and independent evidence/provenance in `records`. Applications and environments
use explicit adapters. Register implementation modules in `framework_layout.py`.

The runtime must not depend on Agent or Skills, and Records must not depend on
task adapters. Keep real device drivers/configuration and private knowledge out
of this source tree. Do not treat an execution acknowledgment as task success,
automatically replay unknown execution, or overwrite prior evidence/versions.

Use the offline example in the [README](README.md). To run the portable software
checks from the source root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The optional `vision` extra is needed for image-change tests; the base loop needs
only Python 3.11+. See the [validation record](docs/validation.md) for the recorded
environment, results and scope.

Record the actual environment, commands and limitations for new experiments.
Real hardware and model calls require separately configured adapters/access;
they are never part of the default offline test suite.

The project is maintained by yeclaw and distributed under the Apache License, Version 2.0.
Keep experiment conditions and upstream acknowledgments accurate when editing
documentation. A saved candidate or passing mock is not physical validation.
