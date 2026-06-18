# QHIS-Net: A Query-guided Human Instance Separation Framework for Crowded Human Instance Segmentation

Official implementation of **QHIS-Net**, a query-guided human instance separation framework for crowded human instance segmentation.

QHIS-Net is built upon M2FP and introduces two dedicated modules to address severe occlusion, instance adhesion, and boundary ambiguity in crowded human scenes:

* **OHMRM**: Occlusion-aware Human Mask Refinement Module
* **AISCM**: Adjacent Instance Separation Constraint Module

The proposed framework improves mask quality and instance discrimination under challenging crowded scenarios while preserving the efficiency of the query-based segmentation paradigm.

---


## Main Contributions

### OHMRM

The Occlusion-aware Human Mask Refinement Module refines coarse query masks through:

* Query-conditioned residual refinement
* Difficulty-guided local residual gating
* Weakly gated multi-scale residual compensation
* Boundary auxiliary contour supervision

This module effectively enhances:

* Occluded regions
* Small-scale persons
* Ambiguous boundaries

### AISCM

The Adjacent Instance Separation Constraint Module introduces a query-level separation constraint to reduce overlap between neighboring instances.

The module encourages:

* Better instance discrimination
* Reduced mask adhesion
* Clearer boundary assignment

---

## Datasets

Experiments are conducted on the CIHP dataset.

To evaluate cross-dataset generalization under severe crowding and occlusion, we additionally report results on the OCHuman benchmark without further fine-tuning.

- Training Dataset: CIHP
- Testing Dataset: CIHP
- Generalization Dataset: OCHuman

---

## Installation

Please refer to:

```text
INSTALL.md
```

---


## Training

```bash
python train_net.py \
--config-file configs/cihp/m2fp_R101_bs16_265k.yaml
```

---

## Evaluation

### Standard Evaluation

```bash
python train_net.py \
    --eval-only \
    --config-file configs/cihp/m2fp_R101_bs16_265k.yaml \
    MODEL.WEIGHTS weights/final.pkl
```

### COCO-style Instance Segmentation AP

To evaluate COCO-style mask AP (AP, AP50, AP75, etc.), run:

```bash
python tools/eval_qhisnet_person_coco_ap.py \
    --config configs/cihp/m2fp_R101_bs16_265k.yaml \
    --weights weights/final.pkl \
    --gt-json path/to/annotation.json \
    --image-root path/to/images
```

The evaluation results will be saved in:

```text
outputs/
├── predictions.json
└── metrics.json
```

---

## Project Structure

```text
QHIS-Net
├── configs/                 # Configuration files
├── docs/                    # Framework and module illustrations
│   ├── framework.png
│   ├── ohmrm.png
│   └── aiscm.png
│
├── m2fp/                    # Core implementation
├── tools/                   # Dataset conversion and evaluation tools
├── weights/                 # Model checkpoints
│   └── README.md
│
├── train.sh
├── test.sh
├── train_net.py
│
├── README.md
├── INSTALL.md
├── GETTING_STARTED.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

```

---

Experimental Results

QHIS-Net is evaluated on CIHP for in-domain testing and on OCHuman for cross-dataset generalization without fine-tuning.

The evaluation focuses on crowded and occluded human scenes, where QHIS-Net improves instance separation and mask boundary quality over the M2FP baseline.

The reported metrics include COCO-style mask AP, AP50, AP75, and scale-specific AP.

The framework significantly enhances the segmentation performance of small-scale human instances in crowded scenes while achieving clearer and more accurate boundary assignment for adjacent and adhered human instances.

The evaluation includes CIHP in-domain testing and OCHuman cross-dataset testing.


---

## Acknowledgement

This work is built upon:

* M2FP
* Mask2Former
* Detectron2

We thank the authors for making their code publicly available.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{xu2026qhisnet,
  title={QHIS-Net: A Query-guided Human Instance Separation Framework for Crowded Human Instance Segmentation},
  author={Xu, Kexin and others},
  journal={Under Review},
  year={2026}
}
```
