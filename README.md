# Linear Operator Learning

## Install

To install this package as a dependency, run:

```bash
pip install linear-operator-learning
```

## Development

To develop this project, please setup `uv` by running the following commands:

1. `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. `git clone git@github.com:CSML-IIT-UCL/linear_operator_learning.git & cd linear_operator_learning`
3. `uv sync --dev`
4. `uv run pre-commit install`

### Optional
Set up your IDE to automatically apply the `ruff` styling.
- [VS Code](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
- [PyCharm](https://plugins.jetbrains.com/plugin/20574-ruff)

## Contributing

Please adhere to the following principles while contributing to the project:

1. Adopt a functional style of programming. Avoid abstractions (classes) like they were plague.
2. To add a new feature, create a branch and when done open a Pull Request. It is _not possible_ to approve your own PR.
3. Write tests on the functional level _and not_ on the integration level (which shouldn't matter anyway).
4. The package contains both `numpy` and `torch` based algorithms. Let's keep them separated.
5. The functions shouldn't change the `dtype` or device of the inputs (that is, keep a functional approach)
6. Try to complement your contributions with simple examples to be added in the `examples` folder
