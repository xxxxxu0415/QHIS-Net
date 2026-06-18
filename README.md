
# QHIS-Net: Query-Guided Refinement and Separation for Crowded Human Instance Segmentation

Official implementation of **QHIS-Net**, a query-guided refinement-and-separation framework for crowded human instance segmentation.

QHIS-Net is designed for human-centric visual computing scenarios such as crowded-scene analysis, video surveillance, virtual human modelling, augmented reality, virtual try-on, and human--computer interaction. The method focuses on challenging crowded scenes where small persons, partial occlusion, close inter-person contact, and boundary adhesion often lead to missed detections, incomplete masks, duplicate predictions, and merged human instances.

The framework is built upon a query-based human instance segmentation pipeline and introduces two complementary components:

- **OHMRM**: Occlusion-aware Human Mask Refinement Module
- **AISCM**: Adjacent Instance Separation Constraint Module

Instead of treating crowded-scene segmentation errors as a single mask-quality problem, QHIS-Net decomposes them into two query-level challenges:

1. **Intra-query mask degradation**: incomplete or fragmented masks within individual human queries.
2. **Inter-query response conflict**: overlapping or ambiguous responses among adjacent human queries.

By jointly addressing these two issues, QHIS-Net improves local mask recovery and adjacent-instance discrimination while preserving the efficiency of the query-based segmentation paradigm.

---

## News

- The code is being organized for public release.
- Configuration files, dataset conversion scripts, inference scripts, and evaluation scripts will be released in this repository.
- Trained model weights will be released when available.

---

## Main Components

### 1. Occlusion-aware Human Mask Refinement Module

The **Occlusion-aware Human Mask Refinement Module (OHMRM)** performs lightweight query-conditioned residual refinement after the initial query-mask prediction.

OHMRM uses the initial query mask as an instance-specific spatial prior and predicts a residual correction term. A local difficulty-aware gate derived from low-confidence and boundary-sensitive regions is used to selectively refine degraded mask areas while preserving reliable high-confidence regions.

Main characteristics:

- Query-conditioned residual mask refinement
- Local difficulty-aware residual gating
- Low-confidence and boundary-sensitive region modeling
- Optional multi-scale residual compensation
- Boundary auxiliary supervision

OHMRM is designed to improve:

- Small-person recovery
- Locally incomplete masks
- Occluded body regions
- Boundary quality in contact regions

---

### 2. Adjacent Instance Separation Constraint Module

The **Adjacent Instance Separation Constraint Module (AISCM)** is a training-only query-level separation constraint.

AISCM constructs a differentiable soft pairwise mask-overlap matrix among high-confidence human queries. Query pairs with excessive spatial overlap are selected as conflicting pairs, and a separation loss is imposed to reduce redundant mask responses.

Main characteristics:

- High-confidence query selection
- Soft pairwise mask-overlap computation
- Selective adjacent-query pair mining
- Query-level overlap suppression
- No additional inference-time branch

AISCM is designed to reduce:

- Duplicate predictions
- Mask adhesion
- Ambiguous query assignment
- Merged adjacent human instances

Since AISCM is only used during training, it does not introduce additional inference-time computational branches.

---

## Datasets

### CIHP

The main experiments are conducted on the **CIHP** dataset. Since CIHP provides human parsing annotations, we convert the original part-level labels into COCO-style complete human instance masks.

The conversion protocol is as follows:

1. All visible body-part regions belonging to the same annotated person are merged into one binary person-instance mask.
2. Disconnected visible regions of the same person are retained as one instance mask.
3. Missing body parts are not artificially completed.
4. Adjacent or touching persons are kept as separate instances according to their original person identity labels.
5. Each converted mask is encoded in COCO-style RLE format with its bounding box, area, and category label.
6. All converted annotations are evaluated using the COCO-style instance segmentation protocol.

### OCHuman

To evaluate cross-dataset generalization under severe occlusion, models trained only on CIHP are directly evaluated on **OCHuman** without additional fine-tuning.

The OCHuman experiment is used as a challenging occlusion-intensive generalization test.

---

## Experimental Results

### CIHP-val

On CIHP-val, QHIS-Net improves the M2FP baseline under the same query-based framework.

| Method | AP$_{50:95}$ | AP$_{50}$ | AP$_{75}$ | AP$_s$ | AP$_m$ | AP$_l$ |
|---|---:|---:|---:|---:|---:|---:|
| M2FP baseline | 0.764 | 0.926 | 0.841 | 0.186 | 0.630 | 0.853 |
| QHIS-Net | 0.774 | 0.933 | 0.850 | 0.212 | 0.643 | 0.856 |

QHIS-Net improves AP$_{50:95}$ from **0.764** to **0.774**, AP$_{75}$ from **0.841** to **0.850**, and small-instance AP from **0.186** to **0.212**.

### OCHuman Cross-dataset Evaluation

All models are trained on CIHP and directly evaluated on OCHuman without fine-tuning.

| Method | AP$_{50:95}$ | AP$_{50}$ | AP$_{75}$ | AP$_m$ | AP$_l$ |
|---|---:|---:|---:|---:|---:|
| M2FP baseline | 0.416 | 0.560 | 0.467 | 0.231 | 0.419 |
| Baseline + OHMRM | 0.413 | 0.556 | 0.466 | 0.252 | 0.416 |
| Baseline + AISCM | 0.422 | 0.560 | 0.473 | 0.250 | 0.426 |
| QHIS-Net | 0.425 | 0.565 | 0.476 | 0.255 | 0.430 |

Although the absolute AP on OCHuman is lower than that on CIHP-val due to severe occlusion, limb interleaving, truncation, unusual poses, and non-exhaustive surrounding-person annotations, QHIS-Net still achieves the best overall cross-dataset performance among all variants.

---

## Additional Evaluation

In addition to standard COCO-style mask AP, this repository supports targeted crowded-scene analysis, including:

- Touching AP
- Touching AP$_{75}$
- Touching mIoU
- Duplicate Rate
- Merge Error Rate
- Boundary IoU
- Boundary F-score
- Crowd-density grouped evaluation

These metrics are used to evaluate whether the method improves crowded-scene instance separation rather than only increasing average mask AP.

---

## Installation

Please refer to:

INSTALL.md

A typical environment includes:

* Python
* PyTorch
* Detectron2
* CUDA
* pycocotools

The detailed package versions are provided in:


requirements.txt


---

## Dataset Preparation

The expected dataset structure is:


datasets/
├── CIHP/
│   ├── images/
│   ├── annotations/
│   └── cihp_person_coco.json
│
└── OCHuman/
    ├── images/
    └── ochuman_person_all_full_outer.json

To convert CIHP human parsing annotations into COCO-style person instance masks, run:


python tools/convert_cihp_to_person_coco.py \
    --input-root datasets/CIHP \
    --output-json datasets/CIHP/cihp_person_coco.json


Please modify the dataset paths according to your local environment.

---

## Training

To train QHIS-Net on CIHP, run:


python train_net.py \
    --config-file configs/cihp/m2fp_R101_bs16_265k.yaml


If using distributed training, run:


python train_net.py \
    --num-gpus 4 \
    --config-file configs/cihp/m2fp_R101_bs16_265k.yaml


Please check the configuration file before training, especially the dataset path, output directory, batch size, and pretrained weights.

---

## Evaluation

### Standard Evaluation

To evaluate a trained model on CIHP-val, run:


python train_net.py \
    --eval-only \
    --config-file configs/cihp/m2fp_R101_bs16_265k.yaml \
    MODEL.WEIGHTS weights/final.pkl


### COCO-style Instance Segmentation AP

To evaluate COCO-style mask AP, including AP$*{50:95}$, AP$*{50}$, AP$_{75}$, AP$_s$, AP$_m$, and AP$_l$, run:


python tools/eval_qhisnet_person_coco_ap.py \
    --config configs/cihp/m2fp_R101_bs16_265k.yaml \
    --weights weights/final.pkl \
    --gt-json datasets/CIHP/cihp_person_coco.json \
    --image-root datasets/CIHP/images


The evaluation results will be saved in:


outputs/
├── predictions.json
└── metrics.json


### OCHuman Generalization Evaluation

To evaluate a CIHP-trained model directly on OCHuman, run:


python tools/eval_qhisnet_person_coco_ap.py \
    --config configs/cihp/m2fp_R101_bs16_265k.yaml \
    --weights weights/final.pkl \
    --gt-json datasets/OCHuman/ochuman_person_all_full_outer.json \
    --image-root datasets/OCHuman/images


No additional fine-tuning is used for OCHuman evaluation.

---

## Project Structure


QHIS-Net/
├── configs/                 # Configuration files
├── docs/                    # Framework and module illustrations
│   ├── framework.png
│   ├── ohmrm.png
│   └── aiscm.png
│
├── m2fp/                    # Core implementation
├── tools/                   # Dataset conversion and evaluation tools
├── weights/                 # Model checkpoints or download instructions
│   └── README.md
│
├── train_net.py             # Training and evaluation entry
├── train.sh                 # Training script
├── test.sh                  # Evaluation script
│
├── README.md
├── INSTALL.md
├── GETTING_STARTED.md
├── requirements.txt
├── LICENSE
└── .gitignore

---

## Model Weights

Due to file size limitations, trained model weights may not be directly included in the repository.

If weights are not available in the repository, please refer to:


weights/README.md


for download instructions or release updates.

---

## Notes

* AISCM is used only during training and does not add inference-time branches.
* OHMRM introduces only lightweight residual refinement operations.
* The current implementation focuses on human instance segmentation rather than generic multi-class instance segmentation.
* The CIHP evaluation protocol depends on converted COCO-style person-instance masks.

---

## Acknowledgement

This work is built upon the following projects:

* M2FP
* Mask2Former
* Detectron2

We sincerely thank the authors for making their code publicly available.

---

## Citation

If you find this repository useful, please cite:


@article{xu2026qhisnet,
  title={Query-Guided Refinement and Separation for Crowded Human Instance Segmentation},
  author={Xu, Kexin and Wang, Xiuhui},
  journal={Under Review},
  year={2026}
}

