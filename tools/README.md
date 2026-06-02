# Tools

This directory contains utility scripts for dataset conversion, model evaluation, and pretrained weight conversion.

* `convert_cihp_to_person_coco.py`

Tool to convert CIHP human instance annotations into COCO-style person instance segmentation format.

* `convert_ochuman_to_person_coco.py`

Tool to convert OCHuman annotations into COCO-style person instance segmentation format for cross-dataset generalization evaluation.

* ``eval_qhisnet_person_coco_ap.py``

Tool to run inference with a trained QHIS-Net/M2FP model and evaluate COCO-style instance segmentation AP, including AP, AP50, AP75, AP_small, AP_medium, and AP_large.

* `evaluate_coco_boundary_ap.py`

Tool to evaluate boundary-aware COCO-style AP for instance segmentation masks.

* `analyze_model.py`

Tool to analyze model complexity and structure.

* `convert-pretrained-swin-model-to-d2.py`

Tool to convert Swin Transformer pretrained weights for Detectron2.

* `convert-torchvision-to-d2.py`

Tool to convert torchvision pretrained weights into Detectron2-compatible format.
