# Beyond21

**Beyond21** is an open-source Python framework for global modeling of the
**Cosmic Dawn (CD)** and **Epoch of Reionization (EoR)**.

It provides a flexible and modular pipeline to compute the coupled
evolution of early stellar populations, radiation backgrounds, and the
intergalactic medium, enabling predictions for multiple observables
within a single framework. A full run takes ~0.1 s using a single CPU
core on a laptop, enabling broad, high resolution, parameter exploration.

Key outputs include:
-   Global 21-cm signal
-   UV luminosity functions (UVLFs)
-   Ionization history
-   Contributions to the cosmic X-ray background (CXB)

We encourage users to take advantage of the modular structure and short running time of Beyond21 to test new astrophysical models and explore physics within and beyond the Standard Model.

------------------------------------------------------------------------

# Installation

### Requirements

Beyond21 requires **Python ≥ 3.9**.

------------------------------------------------------------------------

### Quick install

``` bash
git clone git@github.com:OmerZviKatz/Beyond21.git
cd Beyond21
pip install ".[all]"
```

This installs Beyond21 together with plotting and interactive notebook
support.

------------------------------------------------------------------------

### Using a Conda environment (optional)

If you use Conda, you can create a dedicated environment using the provided beyond21.yml file:

``` bash
conda env create -f beyond21.yml
conda activate beyond21
```
After cloning the repository, install Beyond21 inside the environment:

``` bash
pip install .
```

### For developers

If you plan to modify the code, install Beyond21 in editable mode:

``` bash
pip install -e ".[all]"
```

Editable mode ensures that changes to the source code are immediately
reflected without reinstalling the package.

------------------------------------------------------------------------

# Tutorial

To get started with Beyond21, we recommend exploring the notebooks in the
`tutorial` directory.

------------------------------------------------------------------------

# Extensions

See the 2cDM branch for an example of a beyond-Standard-Model implementation in Beyond21

------------------------------------------------------------------------

# Citation

If you use **Beyond21** in your research, please cite [Katz (2026)](https://arxiv.org/pdf/2603.04542).

