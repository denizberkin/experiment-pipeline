# Experiment Pipeline

A small, config-driven skeleton for ML experiments. It keeps the experiment flow separate from model code, so data, models, losses, metrics, stages, and tracking can be swapped without rewriting the runner.

The pipeline is framework-neutral. PyTorch, scikit-learn, or plain Python components can use the same config and CLI as long as they implement the matching interface.

## What is here

```text
eval_pipeline/      config, interfaces, registry, runner, tracking
examples/components/ minimal framework-free component set
examples/cifar10/    complete PyTorch example
configs/             example experiment configs
tests/               config, CLI, interface, and tracking checks
```

An experiment wires together `data`, `model`, and the CLI stages it needs: `training`, `validation`, and `test`. Losses and metrics are reusable lists, and one tracker is shared across the run.

## Try it

Python 3.13 or newer is required. The toy example has no third-party runtime dependencies:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
pip install -e .
python -m eval_pipeline validate configs/example.toml --check-imports
python -m eval_pipeline run configs/example.toml --stages training validation test
```

For the PyTorch example, install `pip install -e ".[cifar10]"` and use `configs/cifar10_basic.toml`.

## Test

The core suite uses the standard library and skips optional PyTorch or MLflow checks when those packages are absent:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs this suite on Linux and Windows.

## Configure an experiment

Components are registered in Python, then selected by name in TOML. Paths are resolved relative to the config file.

```toml
[registry]
paths = ["../examples/components/data/example_data.py", "../examples/components/models/example_model.py"]
[data]
name = "toy_data"
[model]
name = "toy_model"
```

The full config can also select stages, losses, metrics, output location, seed, device, and tracking. See [`configs/example.toml`](configs/example.toml) for a short complete experiment.

## Add a component

Subclass the matching interface and give the class a config-friendly name:

```python
from eval_pipeline.interfaces import ModelFactory
from eval_pipeline.registry import register_component
@register_component("my_model", category="model")
class MyModel(ModelFactory):
    def build(self):
        return create_model()
```

Available categories are `data`, `model`, `loss`, `metric`, `training`, `validation`, `test`, `prediction`, and `tracking`. Component-specific options go under its `params` table.

```toml
[model]
name = "my_model"
[model.params]
hidden_size = 128
dropout = 0.2
```

## Run and track

```bash
eval-pipeline validate configs/example.toml --check-imports
eval-pipeline run configs/example.toml --stages training validation test
eval-pipeline run configs/example.toml --stages test
```

Without a `[tracking]` section, the local tracker stores the source and resolved configs, JSONL logs, artifacts, and serialized models under `runs/<experiment-name>/`. It refuses to reuse an existing experiment directory, so give each run a unique experiment name or archive the previous output first. MLflow is available with `pip install -e ".[mlflow]"` and `name = "mlflow"` in `[tracking]`.

Components receive a stage-specific context for shared state and logging:

```python
def train(self, *, data, model, losses, metrics, context):
    context.tracker.log_metric("accuracy", 0.9, stage=context.stage)
    context.state["model"] = model
    return {"status": "trained"}
```

## Use it as a GitHub template

This repository can be marked as a template without changing the code. In GitHub, open **Settings > General**, enable **Template repository**, then create projects through **Use this template**.

A repository created from it gets a new, independent repository with these files, not a fork or this commit history. GitHub copies the default branch unless the creator chooses to include all branches.

For each new project:

- Rename the package metadata in `pyproject.toml`.
- Choose the supported Python version in `pyproject.toml` and `.python-version`.
- Update the MIT license copyright notice if needed.
- Add project components and configs; keep `eval_pipeline/` as the reusable runner.
- Remove the toy or CIFAR-10 examples if they are no longer useful.
- Replace this README with the project's purpose and commands.

GitHub templates do not substitute names inside files. Keep the manual checklist above, or add Copier/Cookiecutter only if automatic renaming becomes worth maintaining.

## License

MIT. See [`LICENSE`](LICENSE).
