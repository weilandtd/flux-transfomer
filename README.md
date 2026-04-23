# SigmaFlux
Open-source code AI-model for flux identification.

## Installation

```bash
# Editable install for development
pip install -e ".[dev]"

# Install runtime dependencies only
pip install -r requirements.txt

# Or once published to PyPI
pip install sigmaflux
```

## Quick start

```python
import sigmaflux
from sigmaflux.core import MFAModel, Experiment, Tracer
from sigmaflux.expansion import EMUExpansion
```
