# Pipeline

I tried to make this as flexible and config driven as possible.

Summarizing:
The project is organized around separate component categories that gather into one experiment:

```text
eval_pipeline/
  core/          config loading, dynamic imports, registry checks
  interfaces/    framework-neutral base classes
  components/    category-specific base/build modules
```

User code can live anywhere, but the recommended shape is:

```text
components/
  data/
  models/
  losses/
  metrics/
  training/
  validation/
  testing/
```

## Single Experiment Config

Recommended style: register component aliases in your Python files, preload those files once, and refer to components by `name`.

```toml
[experiment]
name = "baseline"
seed = 42
output_dir = "runs"

[tracking]
name = "local"

[registry]
paths = [
  "components/data/challenge_data.py",
  "components/models/unet.py",
  "components/training/torch_supervised.py",
  "components/validation/default.py",
  "components/testing/default.py",
  "components/losses/mse.py",
  "components/metrics/mae.py",
]

[data]
name = "challenge_data"

[model]
name = "unet"

[training]
name = "torch_supervised"

[validation]
name = "default_validator"

[test]
name = "default_tester"

[[losses]]
name = "mse"
weight = 1.0

[[metrics]]
name = "mae"
stages = ["validation", "test"]
```

Component paths are resolved relative to the config file.

Built-in components under `eval_pipeline/components/` can be referenced by name without adding a path. For example, `eval_pipeline/components/metrics/dice.py` can provide `name = "dice"` directly. External project components still need either `[registry].paths` or a local `path`.

You can also skip `[registry]` and keep paths local to each component:

```toml
[model]
name = "unet"
path = "components/models/unet.py"
```

For explicit class loading, use both `path` and `class_name`:

```toml
[model]
name = "unet"
path = "components/models/unet.py"
class_name = "UNetFactory"
```

## Experiment Tracking

Every context includes `context.tracker`. With no `[tracking]` section, the local tracker is used and writes into:

```text
runs/<experiment-name>/
  config.source.toml
  config.resolved.json
  logs/
    metrics.jsonl
    losses.jsonl
  artifacts/
```

The base tracker contract lives at `eval_pipeline.interfaces.tracking.ExperimentTracker`. Concrete trackers live under `eval_pipeline.components.trackers`, currently `LocalExperimentTracker` and `MlflowExperimentTracker`.

Use MLflow by switching the tracking component:

```toml
[tracking]
name = "mlflow"

[tracking.params]
experiment_name = "segmentation"
run_name = "unet_baseline"
tracking_uri = "file:../mlruns"
```

Custom trackers use the same component loading style:

```toml
[tracking]
name = "wandb_tracker"
path = "components/trackers/wandb_tracker.py"

[tracking.params]
project = "my-project"
```

MLflow is optional. Install it with:

```bash
pip install "pipeline[mlflow]"
```

Components can log values and files directly:

```python
context.tracker.log_metric("dice", dice_score, step=epoch, stage=context.stage)
context.tracker.log_loss("mse", loss_value, step=epoch, stage=context.stage)
context.tracker.log_artifact(context.paths.artifacts_dir / "predictions.csv")
context.tracker.log_model(model, name="final_model")
```

For a full train/validate/test run, build one tracker and pass it to each context so all stages share the same MLflow run:

```python
from eval_pipeline.context import build_experiment_paths, build_experiment_tracker, make_training_context

paths = build_experiment_paths(config)
tracker = build_experiment_tracker(config, paths)
context = make_training_context(config, tracker=tracker)
```

## Base Classes

Import base classes from the category folders:

```python
from eval_pipeline.components.data.base import DataModule
from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.components.losses.base import Loss
from eval_pipeline.components.metrics.base import Metric
from eval_pipeline.components.trainers.base import Trainer
from eval_pipeline.components.validators.base import Validator
from eval_pipeline.components.testers.base import Tester
from eval_pipeline.components.trackers.base import ExperimentTracker
```

The pipeline core stays framework-neutral. A PyTorch trainer can own epochs, optimizers, devices, schedulers, and AMP; an sklearn-style trainer can just call `fit`.

## Register Components

Decorate user components with a category-specific alias:

```python
from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component


@register_component("unet", category="model")
class UNetFactory(ModelFactory):
    def build(self):
        ...
```

Trackers register the same way:

```python
from eval_pipeline.components.trackers.base import ExperimentTracker
from eval_pipeline.registry import register_component


@register_component("wandb_tracker", category="tracking")
class WandbTracker(ExperimentTracker):
    def start(self, config, paths):
        ...
```

Static category specs still live in the pipeline so aliases are checked against the right interface. For example, a `category="metric"` component must subclass `Metric`.

## Typed Context

Training, validation, testing, and prediction mechanisms receive typed context objects so editors can autocomplete stable fields:

```python
from typing import Any

from eval_pipeline.components.trainers.base import Trainer
from eval_pipeline.context import TrainingContext
from eval_pipeline.registry import register_component


@register_component("torch_supervised", category="training")
class TorchSupervisedTrainer(Trainer):
    def train(
        self,
        *,
        data: Any,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> dict:
        epochs = self.params.get("epochs", 1)
        experiment_name = context.config.name
        checkpoint_dir = context.paths.artifacts_dir
        context.state["last_epoch"] = epochs
        return {"experiment": experiment_name}
```

The always-defined fields are:

- `context.config`
- `context.paths`
- `context.seed`
- `context.tracker`
- `context.state`
- `context.stage`

Use `params` for values owned by the component, and `context.state` for values shared between mechanisms during a run.

## Validate

```bash
python -m eval_pipeline validate configs/example.toml --check-imports
```

or:

```bash
python main.py validate configs/example.toml --check-imports
```

## CIFAR-10 Components

`components/cifar10/` contains a basic PyTorch CIFAR-10 classifier component set:

- `cifar10_data`
- `cifar10_small_cnn`
- `torch_cross_entropy`
- `classification_accuracy`
- `cifar10_torch_trainer`
- `cifar10_validator`
- `cifar10_tester`

Use it with `configs/cifar10_basic.toml`. Install runtime dependencies first:

```bash
pip install "pipeline[cifar10]"
```
