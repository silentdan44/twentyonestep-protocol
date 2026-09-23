# 21-Step Molecular Dynamics Equilibration Protocol (OpenMM)
This repository provides a robust Python implementation of the 21-stage pressure ramping and thermal cycling protocol commonly used for the rigorous equilibration of complex, dense systems, such as polymer melts or glasses.

This implementation is based on the methodology described in:

>[Larsen GS, Lin P, Hart KE, Colina CM (2011). Macromolecules 44:6944–6951.](https://pubs.acs.org/doi/10.1021/ma200345v)

## Installation
```bash
git clone https://github.com/silentdan44/twentyonestep-protocol.git
cd twentyonestep-protocol
uv sync
```

The project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. This creates the project environment and installs the package in
editable mode from `uv.lock`.

To run a script with the project environment:

```bash
uv run python path/to/script.py
```

Alternatively, an existing virtual environment can still be used with:

```bash
pip install -e .
```

### Requirements
```
openmm >= 8.0
numpy
```

## Usage Example
```python
from twentyonestep import TwentyOneStepProtocol


P_MAX = 75000 * unit.bar 

protocol = TwentyOneStepProtocol(
    simulation=my_simulation, 
    max_pressure=P_MAX 
)

protocol.run(barostat_frequency=50)
```
