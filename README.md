# PlantDiseaseTraining

Project dedicated exclusively to training and evaluating a plant disease vision model that will later be consumed by the `PlantDiseaseAI` microservice.

## Architecture Goal

The repository is organized to keep training code isolated from generated artifacts:

- `src/`: reusable Python packages for data, models, training, evaluation, export, and utilities
- `dataset/`: supervised image dataset arranged by split and class
- `checkpoints/`: saved model checkpoints during training
- `outputs/`: metrics, plots, confusion matrices, and evaluation artifacts
- `notebooks/`: optional exploration and research notebooks
- `tests/`: future automated tests for data and pipeline utilities
- root scripts: `train.py`, `evaluate.py`, `export.py`

## Current Stage

This stage only creates the project foundation. No training pipeline is implemented yet.

## Planned Dataset Layout

```text
dataset/
  train/
    Tomato___Healthy/
    Tomato___Early_blight/
    Potato___Healthy/
  validation/
  test/
```

## Dependencies

Install the required libraries with:

```bash
pip install -r requirements.txt
```

## Next Steps

1. Define reusable dataset loading and transform modules.
2. Add training configuration objects.
3. Implement training, evaluation, and export workflows.
4. Add TorchScript export compatible with `PlantDiseaseAI`.
