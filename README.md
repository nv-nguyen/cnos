<div align="center">
<h2>
CNOS: A Strong Baseline for CAD-based Novel Object Segmentation
</h2>

<h3>
<a href="https://nv-nguyen.github.io/" target="_blank"><nobr>Van Nguyen Nguyen</nobr></a> &emsp;
<a href="https://cmp.felk.cvut.cz/~hodanto2/" target="_blank"><nobr>Tomáš Hodaň</nobr></a> &emsp; <br>
<a href="https://ponimatkin.github.io/" target="_blank"><nobr>Georgy Ponimatkin</nobr></a> &emsp;
<a href="http://imagine.enpc.fr/~groueixt/" target="_blank"><nobr>Thibault Groueix</nobr></a> &emsp;
<a href="https://vincentlepetit.github.io/" target="_blank"><nobr>Vincent Lepetit</nobr></a>

<p></p>
<a href="http://arxiv.org/abs/2307.11067"><img 
src="https://img.shields.io/badge/-Paper-blue.svg?colorA=333&logo=arxiv" height=35em></a>
<p></p>

![framework](./media/framework.png)

![qualitative](./media/qualitative.png)
</h3>
</div>

CNOS is a simple three-stage approach for CAD-based novel object segmentation. It is based on [Segmenting Anything](https://github.com/facebookresearch/segment-anything), [DINOv2](https://github.com/facebookresearch/dinov2) and can be used for any objects without retraining. CNOS outperforms the supervised MaskRCNN (in CosyPose) which was trained on target objects. CNOS has been used as the baseline for [Task 5](https://bop.felk.cvut.cz/leaderboards/detection-unseen-bop23/core-datasets/) and [Task 6](https://bop.felk.cvut.cz/leaderboards/segmentation-unseen-bop23/core-datasets/) in [BOP challenge 2023](https://bop.felk.cvut.cz/challenges/bop-challenge-2023/)!

![bo results](./media/bop_results.png)

Here are some qualitative results of CNOS on the YCBV dataset. We are displaying only detections with a confidence score greater than 0.5 for better visibility.
![ycbv](./media/ycbv.gif)
If our project is helpful for your research, please consider citing : 
```latex
@article{nguyen2023cnos,
title={CNOS: A Strong Baseline for CAD-based Novel Object Segmentation},
author={Nguyen, Van Nguyen and Hodan, Tomas and Ponimatkin, Georgy and Groueix, Thibault and Lepetit, Vincent},
journal={arXiv preprint arXiv:2307.11067},
year={2023}}
```
You can also put a star :star:, if the code is useful to you.

If you like this project, check out related works from our group:
- [NOPE: Novel Object Pose Estimation from a Single Image (arXiv 2023)](https://github.com/nv-nguyen/nope)
- [Templates for 3D Object Pose Estimation Revisited: Generalization to New objects and Robustness to Occlusions (CVPR 2022)](https://github.com/nv-nguyen/template-pose) 
- [PIZZA: A Powerful Image-only Zero-Shot Zero-CAD Approach to 6DoF Tracking (3DV 2022)](https://github.com/nv-nguyen/pizza)
- [BOP visualization toolkit](https://github.com/nv-nguyen/bop_viz_kit)

## Installation :construction_worker:

<details><summary>Click to expand</summary>

This repository is running with the Weight and Bias logger. Ensure that you update this [user's configuration](https://github.com/nv-nguyen/cnos/blob/main/configs/user/default.yaml) before conducting any experiments. 

### 1. Create conda environment
```
conda env create -f environment.yml
conda activate cnos

# for using SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# for using fastSAM
pip install ultralytics
```

### 2. Datasets and model weights

#### 2.1. Download datasets from [BOP challenge](https://bop.felk.cvut.cz/datasets/):
```
python -m src.scripts.download_bop
```

#### 2.2. Rendering templates with [Pyrender](https://github.com/mmatl/pyrender):

This rendering is fast. For example, using a single V100 GPU, it can be done within 10 minutes. Alternatively, you can access the rendered output through [this Google Drive link (73MB)]() and unzip it into $ROOT_DIR.

```
python -m src.scripts.render_template_with_pyrender
```

#### 2.3. Download model weights of [Segmenting Anything](https://github.com/facebookresearch/segment-anything):
```
python -m src.scripts.download_sam
```

#### 2.4. Download model weights of [Fast Segmenting Anything](https://github.com/CASIA-IVA-Lab/FastSAM):
```
python -m src.scripts.download_fastsam
```

#### 2.5. Download [BlenderProc4BOP](https://bop.felk.cvut.cz/datasets/) set:
This is only required when you want to use realistic rendering with BlenderProc for seven core datasets of BOP challenge.
```
python -m src.scripts.download_train_pbr
```


</details>

##  Testing on [BOP datasets](https://bop.felk.cvut.cz/datasets/) :rocket:

We provide CNOS's predictions for seven core dataset of BOP challenge with both SAM and FastSAM models in [this link](https://drive.google.com/drive/folders/1yGRKpz1RI4h5-u0drusVeXPuAsg_GIO5?usp=sharing).

<details><summary>Click to expand</summary>

1. Run CNOS to get predictions:

```
export DATASET_NAME=lmo 

# with FastSAM + PBR
python run_inference.py dataset_name=$DATASET_NAME model=cnos_fast

# with FastSAM + PBR + denser viewpoints
python run_inference.py dataset_name=$DATASET_NAME model=cnos_fast model.onboarding_config.level_templates=1

# with FastSAM + PyRender
python run_inference.py dataset_name=$DATASET_NAME model=cnos_fast model.onboarding_config.rendering_type=pyrender

# with SAM + PyRender
python run_inference.py dataset_name=$DATASET_NAME model.onboarding_config.rendering_type=pyrender

# with SAM + PBR
python run_inference.py dataset_name=$DATASET_NAME

```
After running this script, CNOS will output a prediction file at [this dir](https://github.com/nv-nguyen/cnos/blob/main/configs/run_inference.yaml#L9). You can then evaluate this prediction on [BOP challenge website](https://bop.felk.cvut.cz/).

2. Visualize the predictions:

There are two options:

2.a. Using our custom visualization without Detectron2 (display only masks)

```
python -m src.scripts.visualize dataset_name=$DATASET_NAME input_file=$INPUT_FILE output_dir=$OUTPUT_DIR
```

2.b. Using Detectron2 (display both masks, objectID, scores)
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m src.scripts.visualize_detectron2 dataset_name=$DATASET_NAME input_file=$INPUT_FILE output_dir=$OUTPUT_DIR

```

</details>

## Acknowledgement

The code is adapted from [Nope](https://github.com/nv-nguyen/nope), [Segmenting Anything](https://github.com/facebookresearch/segment-anything), [DINOv2](https://github.com/facebookresearch/dinov2). 

## Contact
If you have any question, feel free to create an issue or contact the first author at van-nguyen.nguyen@enpc.fr