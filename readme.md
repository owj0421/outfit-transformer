# Outfit Transformer: Outfit Representations for Fashion Recommendation

<div align="center"> <img src="https://github.com/owj0421/outfit-transformer/assets/98876272/fc39d1c7-b076-495d-8213-3b98ef038b64" width="512"> </div>

## 📌 Introduction

This repository provides the implementation of **Outfit Transformer**, a model designed for fashion recommendation, inspired by:

> Rohan Sarkar et al. [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812). CVPR 2023.

Our implementation faithfully reproduces the original paper’s method while introducing key improvements for enhanced performance and efficiency.

## 🚀 Key Improvements

✅ **Enhanced Performance**: Upgraded the backbone architecture for better accuracy and generalization.

✅ **Faster Training**: Precomputes item features, significantly reducing computational overhead.

✅ **Refined CIR Task**: Eliminates data leakage from item descriptions/categories and replaces them with learnable embeddings, making the model more robust.

## 📊 Performance

<div align="center">

| Model | CP (AUC) | FITB (Accuracy) |  
|:-|:-:|:-:|  
| **Type-Aware** | 0.86 | 57.83 |  
| **SCE-Net** | 0.91 | 59.07 |  
| **CSA-Net** | 0.91 | 63.73 |  
| **OutfitTransformer (Paper)** | 0.93 | 67.10 |  
| **OutfitTransformer (Our Impl.)** | 0.93 | 67.02 |  
| **OutfitTransformer (Our Impl. + CLIP)** | **_0.95_**<br>_(SOTA, ↑0.02)_ | **_69.24_**<br>_(SOTA, ↑2.14)_ |  

</div>

## 🛠️ Installation

```bash
conda create -n outfit-transformer python=3.12.4
conda activate outfit-transformer
conda env update -f environment.yml
```

## 📥 Download Datasets & Checkpoints

```bash
mkdir -p datasets
gdown --id 1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu -O polyvore.zip
unzip polyvore.zip -d ./datasets/polyvore
rm polyvore.zip

mkdir -p checkpoints
gdown --id 1mzNqGBmd8UjVJjKwVa5GdGYHKutZKSSi -O checkpoints.zip
unzip checkpoints.zip -d ./checkpoints
rm checkpoints.zip
```

## 🏋️ Training & Evaluation

### Step 1: Precompute CILP Embeddings
Before proceeding with training, make sure to precompute the CLIP embeddings, as all subsequent steps rely on these precomputed features.

```bash
python -m src.run.1_generate_clip_embeddings
```

### Step 2: Compatibility Prediction
Train the model for the Compatibility Prediction (CP) task.

#### 🔥 Train
```bash
python -m src.run.2_train_compatibility \
--wandb_key $YOUR/WANDB/API/KEY
```

#### 🎯 Test
```bash
python -m src.run.2_test_compatibility \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```

### Step 3: Complementary Item Retrieval
After completing Step 1, use the best checkpoint from the Compatibility Prediction task to train for the Complementary Item Retrieval (CIR) task.

#### 🔥 Train
```bash
python -m src.run.3_train_complementary \
--wandb_key $YOUR/WANDB/API/KEY \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```

#### 🎯 Test
```bash
python -m src.run.3_test_complemenatry \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```

## Demo

Follow the steps below to run the demo:

#### Build Database
```
python -m src.demo.1_generate_rec_embeddings \
--checkpoint $PATH/OF/MODEL/.PT/FILE
```

#### Build Faiss Index.
```
python -m src.demo.2_build_index
```

#### Run Demo
```
python -m src.demo.3_run \
--checkpoint $PATH/OF/MODEL/.PT/FILE
```

## ⚠️ Note

This is a non-official implementation of the Outfit Transformer model. The official repository has not been released yet.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

If you use this repository, please mention the original GitHub repository by linking to [outfit-transformer](https://github.com/owj0421/outfit-transformer). This helps support the project and acknowledges the contributors.