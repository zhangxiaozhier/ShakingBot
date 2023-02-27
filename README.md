<h1> ShakingBot: Dynamic Manipulation for Bagging</h1>

<div style="text-align: center;">

[Gu NingQuan](https://github.com/happydog-gu), [Zhang ZhiZhong](https://github.com/zhangxiaozhier)

[Wuhan Textile University](https://www.wtu.edu.cn/)
</div>

<img style="left-margin:50px; right-margin:50px;" src="assets/shakingbot.jpg">

<div style="margin:50px; text-align: justify;">
Bag manipulation through robots is complex and challenging due to the deformability of the bag.
Based on dynamic manipulation strategy, we propose the ShakingBot for the bagging tasks.
Our approach utilizes a perception module to identify the key region of the plastic bag from arbitrary initial configurations. 
According to the segmentation, ShakingBot iteratively executes a set of actions, including Bag Adjustment, Dual-arm Shaking and One-arm Holding, to open the bag. Then, we insert the items and lift the bag for transport. 
We perform our method on a dual-arm robot and achieve a success rate of 21/33 for inserting at least one item across a variety of initial bag configurations.
In this work, we demonstrate the performance of dynamic shaking actions compared to the quasi-static manipulation in the bagging task.
We also show that our method generalizes to variations despite the bag's size, pattern, and color.
</div>

<br>

This repository contains code for training and evaluating ShakingBot in both simulation and real-world settings on a dual-UR5 robot arm setup for Ubuntu 18.04.
It has been tested on machines with Nvidia GeForce RTX 2080 Ti.

# Table of Contents
- 1 [Data Collection](#data-collection)
- 2 [Get Datasets](#get-datasets)
- 3 [Network Training](#network-training)
  - 3.1 [Installation and Code Usage](#installation-and-code-usage)
  - 3.2 [Train ShakingBot](#train-shakingbot)
  - 3.3 [Evaluate ShakingBot](#evaluate-shakingbot)

# Data Collection

# Get Datasets

# Network Training
## Installation and Code Usage
1. Make a new virtualenv or conda env. For example, if you're using conda envs, run this to make and then activate the environment:
    ```
    conda create -n shakingbot python=3.6 -y
    conda activate shakingbot
    ```
2. Run pip install -r requirements.txt to install dependencies.
    ```
    cd network_training
    pip install -r requirements.txt
    ```

## Train Region Perception Model
1. In the repo's root, get rgb and depth map

2. In the `configs` folder modify `segmentation.json`

3. Train Region Perception model
    ```
        python train.py
    ```


## Evaluate Region Perception Model
<img style="left-margin:10px; right-margin:10px;" src="assets/model_prediction.png">


<center class="half">
<figure>
    <img src="assets/loss.png" >
    <img src="assets/meanIOU.png">
</figure>
</center>





1. In the repo's root, download the [model weights](https://drive.google.com/file/d/1-BuhIfmZCCvlW4gIxxTCj5XPGdFebea6/view?usp=sharing)


2. Then validate the model from scratch with
    ```
    python visualize.py
    ```
3.  Training details can be viewed in the bag
    ```
    cd network_training
    tensorboard --logdir train_runs/
    ```

