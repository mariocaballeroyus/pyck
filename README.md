# pyC^k

`pyC^k` is a Python/C++ library for isogeometric beam/plate/shell formulations. As for now, the library is limited to single-patch, linear-elastic cases.

## Installation

From the repository root, create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

And install the library:

```bash
pip install -e .
```

You can now import the library in a python script or notebook:

```python
import pyck as ck
```

You can run the available notebook examples inside the `notebooks/` directory.
