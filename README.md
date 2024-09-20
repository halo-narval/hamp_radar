# HAMP Radar Processing tools

## developing

This project uses various file formaters and linters (e.g. [ruff](https://github.com/astral-sh/ruff) for Python code) and uses [pre-commit](https://pre-commit.com/) to run them automatically.

Please ensure that you have `pre-commit` configured locally by running:
```
pre-commit install
```

One of the pre-commit hooks ensures you use
[conventional commit messages](https://www.conventionalcommits.org/) which have the form:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
