# Getting Started with QHIS-Net

This document provides a brief introduction to training and evaluating QHIS-Net for crowded human instance segmentation.

---

## Dataset Preparation

Current experiments are conducted on the CIHP dataset.

Please organize the dataset as follows:

```text
datasets/
└── CIHP
    ├── Training
    ├── Validation
    ├── train_id.txt
    └── val_id.txt
```

Refer to the dataset preparation instructions for more details.

---

## Training

To train QHIS-Net:

```bash
sh train.sh
```

or

```bash
python train_net.py \
    --config-file configs/cihp/qhisnet.yaml
```

The default configuration is designed for multi-GPU training.

---

## Evaluation

To evaluate a trained model:

```bash
sh test.sh
```

or

```bash
python train_net.py \
    --eval-only \
    --config-file configs/cihp/qhisnet.yaml \
    MODEL.WEIGHTS weights/final.pkl
```

---

## Main Components

QHIS-Net contains two key modules:

### OHMRM

Occlusion-aware Human Mask Refinement Module

* Multi-scale feature enhancement
* Difficulty-aware weighting
* Residual mask refinement

### AISCM

Adjacent Instance Separation Constraint Module

* Query-level separation constraint
* Neighboring instance discrimination
* Overlap suppression

---

## Expected Outputs

Training logs and checkpoints will be saved to:

```text
output/
```

including:

```text
output/
├── metrics.json
├── events.out.*
├── model_final.pth
└── inference/
```

---

## Reproducing the Paper Results

To reproduce the results reported in the paper:

```bash
python train_net.py \
    --config-file configs/cihp/qhisnet.yaml
```

After training is completed, evaluate the final checkpoint:

```bash
python train_net.py \
    --eval-only \
    MODEL.WEIGHTS output/model_final.pth
```

---

## Additional Options

For more options:

```bash
python train_net.py -h
```
