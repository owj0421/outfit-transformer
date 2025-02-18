git filter-branch --commit-filter '
  	if [ "$GIT_AUTHOR_NAME" = "rlawldud53" ];
        then
          	GIT_AUTHOR_NAME="big_oh_one";
   		    GIT_AUTHOR_EMAIL="owj0421@naver.com";
      	git commit-tree "$@";
	else
      	git commit-tree "$@";
	  	fi' HEAD
# Outfit Transformer: Outfit Representations for Fashion Recommendation

## Introduction

This repository contains the implementation of the Outfit Transformer, inspired by the original paper:

> Rohan Sarkar et al. [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812). CVPR 2023.

Our implementation not only faithfully reproduces the method presented in the paper but also introduces several enhancements to improve performance. These improvements elevate the model to a state-of-the-art (SoTA) level, achieving superior results in fashion recommendation tasks.

<div align="center"> <img src = https://github.com/owj0421/outfit-transformer/assets/98876272/fc39d1c7-b076-495d-8213-3b98ef038b64 width = 512> </div>

## Performance

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

## Settings

### Environment Setting
```
conda create -n outfit-transformer python=3.12.4
conda activate outfit-transformer
conda env update -f environment.yml
```
### Download Dataset
```
cd outfit-transformer
gdown --id 1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu
unzip polyvore.zip -d polyvore
```
### Download Checkpoint
Pretrained model checkpoints are also available [here](https://drive.google.com/drive/folders/1cMTvmC6vWV9F9j08GX1MppNm6DDnSiZl?usp=drive_link).

## Training
Follow the steps below to train the model:

### Step 1: Compatibility Prediction
Start by training the model for the Compatibility Prediction (CP) task

**Train**
```
python -m src.run.1_train_compatibility \
--wandb_key $YOUR/WANDB/API/KEY
```
**Test**
```
python -m src.run.1-1_test_compatibility \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```

<br>

### Step 2: Complementary Item Retrieval

After completing Step 1, use the checkpoint with the best accuracy from the Compatibility Prediction task to train the model for the Complementary Item Retrieval (CIR) task:

**Train**
```
python -m src.run.2_train_complementary \
--wandb_key $YOUR/WANDB/API/KEY \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```
**Test**
```
python -m src.run.2-1_test_complemenatry \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```

<br>

## Demo

Follow the steps below to run the demo for each task:

**Compatibility Prediction**

1. Run demo
    ```
    python -m src.run.5_demo \
    --task cp \
    --model_type clip \
    --checkpoint $PATH/OF/MODEL/.PT/FILE
    ```

<br>

**Complementary Item Retrieval**

1. Build Database
    ```
    python -m run.3_build_db \
    --polyvore_dir $PATH/TO/LOAD/POLYVORE \
    --db_dir $PATH/TO/SAVE/ITEM/METADATA
    ```
1. Generate Item Embeddings
    ```
    python -m src.run.3_generate_embeddings \
    --model_type clip \
    --batch_sz 64 \
    --checkpoint $PATH/OF/MODEL/.PT/FILE
    ```
2. Build Faiss Index.
    ```
    python -m src.run.4_build_index
    ```
3. Run Demo
    ```
    python -m src.run.5_demo \
    --task cir \
    --model_type clip \
    --checkpoint $PATH/OF/MODEL/.PT/FILE
    ```

## Note
This is **NON-OFFICIAL** implementation. (The official repo has not been released.)

## License
This code is licensed under the MIT License.