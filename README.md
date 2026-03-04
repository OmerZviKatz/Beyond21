# Beyond21

**Beyond21** is an open-source Python framework for global modeling of the
**Cosmic Dawn (CD)** and **Epoch of Reionization (EoR)**.

It provides a flexible and modular pipeline to compute the coupled
evolution of early stellar populations, radiation backgrounds, and the
intergalactic medium, enabling predictions for multiple observables
within a single framework.

Key outputs include:

-   Global **21-cm signal**
-   **UV luminosity functions (UVLFs)**
-   **Ionization history**
-   Contribution to the **cosmic X-ray background (CXB)**

The modular design allows straightforward modification of astrophysical
prescriptions and the incorporation of new physics.

------------------------------------------------------------------------

# Installation

## Requirements

Beyond21 requires **Python ≥ 3.9**.

------------------------------------------------------------------------

## Quick install (recommended)

``` bash
git clone https://github.com/USERNAME/beyond21.git
cd beyond21
pip install ".[all]"
```

This installs Beyond21 together with plotting and interactive notebook
support.

------------------------------------------------------------------------

## Creating an environment (optional)

If you work with Conda you can create a dedicated environment using 
the provided beyond21.yml:

``` bash
conda env create -f beyond21.yml
conda activate beyond21
```

------------------------------------------------------------------------

## Install the package

Install Beyond21:

``` bash
pip install .
```

------------------------------------------------------------------------

## Optional dependencies

Beyond21 provides optional dependency groups for plotting and
interactive notebooks.

### Plotting

``` bash
pip install ".[plot]"
```

Installs:

-   `matplotlib`

------------------------------------------------------------------------

### Interactive notebooks

``` bash
pip install ".[interactive]"
```

Installs:

-   `jupyterlab`
-   `ipywidgets`

------------------------------------------------------------------------

### Full installation

``` bash
pip install ".[all]"
```

Installs all optional dependencies.

------------------------------------------------------------------------

## For developers

``` bash
pip install -e ".[all]"
```

Editable mode ensures that changes to the source code are immediately
reflected without reinstalling the package.

## Verify installation

You can verify that the installation works by importing the package:

``` python
import beyond21
print(beyond21.__version__)
```

------------------------------------------------------------------------

# License

This project is released under the **MIT License**.

------------------------------------------------------------------------

# Citation

If you use **Beyond21** in your research, please cite the corresponding
paper (to appear).
