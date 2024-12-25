# <div align="center"> Outfit Transformer: Outfit Representations for Fashion Recommendation </div>

## 🤗 Introduction

This repository contains the implementation of the Outfit Transformer, inspired by the original paper:

> Rohan Sarkar et al. [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812). CVPR 2023.

Our implementation not only faithfully reproduces the method presented in the paper but also introduces several enhancements to improve performance. These improvements elevate the model to a state-of-the-art (SoTA) level, achieving superior results in fashion recommendation tasks.

<div align="center"> <img src = https://github.com/owj0421/outfit-transformer/assets/98876272/fc39d1c7-b076-495d-8213-3b98ef038b64 width = 512> </div>

## 🎯 Performance

<div align="center">

|Model|CP(AUC)|FITB(Accuracy)|CIR(Recall@10)|
|:-:|:-:|:-:|:-:|
|Type-Aware|0.86|57.83|3.50|
|SCE-Net|0.91|59.07|5.10|
|CSA-Net|0.91|63.73|8.27|
|OutfitTransformer(Paper)|<u>0.93</u>|**67.10**|9.58|
|**Implemented <br> (Original)**|0.92|?|No Dataset|
|**Implemented <br> (w/ CLIP Backbone)**|**0.94 <br> (SOTA)**|<u>65.92</u>|No Dataset|

</div>

## 📥 Download
The model is trained on the Polyvore dataset. Since the official download link is no longer available, you can download the dataset from [here](https://drive.google.com/drive/folders/1cMTvmC6vWV9F9j08GX1MppNm6DDnSiZl?usp=drive_link).

Pretrained model checkpoints are also available [here](https://drive.google.com/drive/folders/1cMTvmC6vWV9F9j08GX1MppNm6DDnSiZl?usp=drive_link).

## 🛠️ Settings

Follow the instructions below to install the required dependencies:
```
pip install -r requirements.txt
```

## 🚀 Training
Follow the steps below to train the model:

**Step 1: Train the model for Compatibility Prediction**
Start by training the model for the Compatibility Prediction (CP) task:
<details>
<summary>Click to expand</summary>

```
python -m run.train \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--polyvore_type nondisjoint \
--task cp \
--batch_sz 64 \
--n_workers 4 \
--n_epochs 16 \
--lr 1e-4 \
--accumulation_steps 2 \
--wandb_key $WANDB_KEY \
--save_dir $CHECKPOINT_DIR
```
</details>

<br>

**Step 2: Train for Complementary Item Retrieval using the best CP checkpoint**
After completing Step 1, use the checkpoint with the best accuracy from the Compatibility Prediction task to train the model for the Complementary Item Retrieval (CIR) task:
<details>
<summary>Click to expand</summary>

```
python -m run.train \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--polyvore_type nondisjoint \
--task cir \
--batch_sz 64 \
--n_workers 4 \
--n_epochs 6 \
--lr 1e-4 \
--accumulation_steps 4 \
--wandb_key $WANDB_KEY \
--save_dir $CHECKPOINT_DIR
--checkpoint $CHECKPOINT
```
</details>

## 🧪 Evaluation

Follow the steps below to evaluate model for each task:

**Compatibility Prediction**
<details>
<summary>Click to expand</summary>

```
python -m run.test \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--polyvore_type nondisjoint \
--task cp \
--batch_sz 64 \
--n_workers 4 \
--result_dir $RESULT_DIR \
--checkpoint $CHECKPOINT
```
</details>

<br>

**Fill-in-the-blank**
<details>
<summary>Click to expand</summary>

```
python -m run.test \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--polyvore_type nondisjoint \
--task fitb \
--batch_sz 32 \
--n_workers 4 \
--result_dir $RESULT_DIR \
--checkpoint $CHECKPOINT
```
</details>

## 🎬 Demo

Follow the steps below to run the demo for each task:

**Compatibility Prediction**
<details>
<summary>Click to expand</summary>

1. Run demo
    ```
    python -m run.demo \
    --model_type clip \
    --polyvore_dir $POLYVORE_DIR \
    --task cp \
    --checkpoint $CHECKPOINT \
    ```
</details>

<br>

**Complementary Item Retrieval**
<details>
<summary>Click to expand</summary>

1. Generate Item Embeddings
    ```
    python -m run.generate_embeddings \
    --model_type clip \
    --polyvore_dir $POLYVORE_DIR \
    --batch_sz 16 \
    --checkpoint $CHECKPOINT \
    ```
2. Build Faiss Index.
    ```
    python -m run.build_index \
    ```
3. Run Demo
    ```
    python -m run.demo \
    --model_type clip \
    --polyvore_dir $POLYVORE_DIR \
    --task cir \
    --checkpoint $CHECKPOINT \
    ```
</details>

## 🔔 Note
This is **NON-OFFICIAL** implementation. (The official repo has not been released.)

## 📜 License
This code is licensed under the MIT License.