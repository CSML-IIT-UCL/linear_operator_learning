(contributing)=
# Developer guide

To develop this project, please setup the [`uv` project manager](https://astral.sh/uv) by running the following commands:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone git@github.com:CSML-IIT-UCL/linear_operator_learning.git 
cd linear_operator_learning
uv sync --dev
uv run pre-commit install
```

### Optional
Set up your IDE to automatically apply the `ruff` styling.
- [VS Code](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
- [PyCharm](https://plugins.jetbrains.com/plugin/20574-ruff)

## Development principles

Please adhere to the following principles while contributing to the project:

1. Adopt a functional style of programming. Avoid abstractions (classes) at all cost.
2. To add a new feature, create a branch and when done open a Pull Request. You should _**not**_ approve your own PRs.
3. The package contains both `numpy` and `torch` based algorithms. Let's keep them separated.
4. The functions shouldn't change the `dtype` or device of the inputs (that is, keep a functional approach).
5. Try to complement your contributions with simple examples to be added in the `examples` folder. If you need some additional dependency add it to the `examples` dependency group as `uv add --group examples _your_dependency_`.