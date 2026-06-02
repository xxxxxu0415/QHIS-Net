# Model Weights

This directory is used to store the model weights required for training and inference.

## Trained Checkpoint

Currently, we provide the final checkpoint trained on the CIHP training set.

| Model            | Training Dataset | Checkpoint  |
| ---------------- | ---------------- | ----------- |
| M2FP-OHMRM-AISCM | CIHP train set   | `final.pkl` |

## Usage

Please place the checkpoint file in this directory:

```text
weights/final.pkl
```

Then set the weight path in the corresponding config file:

```yaml
MODEL:
  WEIGHTS: "./weights/final.pkl"
```

## Notes

Due to file size limitations, the checkpoint file may be provided separately through cloud storage instead of being directly included in this repository.

The current checkpoint is trained only on the CIHP training set. Results on other datasets should be obtained by retraining or fine-tuning the model with the corresponding dataset.
