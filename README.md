# Beyond21 - 2cDM branch

This branch implements the **2cDM** model of [Liu et al. (2019)](https://arxiv.org/abs/1908.06986), providing an example of a beyond-Standard-Model extension within Beyond21.


------------------------------------------------------------------------

# Installation

### Requirements

Beyond21 requires **Python ≥ 3.9**.

------------------------------------------------------------------------

### Quick install

``` bash
git clone git@github.com:OmerZviKatz/Beyond21.git -b 2cDM
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

# Citation

If you use **Beyond21** in your research, please cite [Katz (2026)](https://arxiv.org/pdf/2603.04542).
