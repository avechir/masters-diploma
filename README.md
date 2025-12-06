# Multispectral Explosive Object Detection (RGB + LWIR)
The aim of this research is to enhance the identification of explosive ordnance in images from potentially contaminated areas by employing a multispectral ensemble with a verification mechanism using machine learning techniques.
The goal is achieved by reducing the number of false detections through a sequential use of two deep learning models. The performance of the proposed ensemble was evaluated using the YOLO architecture (YOLOv8 and YOLO11) and a dataset containing both visible (RGB) and infrared (LWIR) spectrum images.
## Project Overview
This repository contains the code for a diploma project focused on detecting explosive objects (mines) using a dual-sensor approach. The project utilizes a custom fusion algorithm to improve detection precision and eliminate false positives compared to single-modal approaches.
* **Fusion logic (main steps):**
1. The RGB model generates a maximum number of potential candidates (high recall strategy).
2. Low-confidence RGB detections are verified using LWIR data to filter out false positives.
3. The system analyzes the LWIR image to recover objects completely missed by the RGB camera.
## Repository Overview
* `diploma_main_method_code.ipynb` - main entry point, contains the pipeline and evaluation workflow.
* `functions/` - helper modules:
    * `ensemble.py` - fusion logic;
    * `evaluation.py` - metric calculation.

This research utilized the [MineInsight dataset](https://github.com/mariomlz99/MineInsight).
