# <div align="center"> Outfit-Transformer </div>

## 🤗 Introduction
Implementation of paper - [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812)<br>

<div align="center"> <img src = https://github.com/owj0421/outfit-transformer/assets/98876272/fc39d1c7-b076-495d-8213-3b98ef038b64 width = 512> </div>

## 🎯 Performance

<div align="center">

|Model|CP(AUC)|FITB(Accuracy)|CIR(Recall@10)|
|:-|-:|-:|-:|
|Type-Aware|0.86|57.83|3.50|
|SCE-Net|0.91|59.07|5.10|
|CSA-Net|0.91|63.73|8.27|
|OutfitTransformer(Paper)|0.93|67.10|9.58|
|**Implemented <br> (Original)**|**0.912**|**?**|Not Trained|
|**Implemented <br> (w/ CLIP Backbone)**|**0.941**|**?**|Not Trained|

</div>



## ⚙ Settings
**Install Dependencies**
```
pip install -r requirements.txt
```

**Download Checkpoint**
Download the checkpoint from [here]()

**Download Dataset**
Download the polyvore dataset from [here]()
## 🧱 Train

**Train CP**
```
python -m train \
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
--save_dir $SAVE_DIR
```

**Train CIR**
```
python -m train \
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
--save_dir $SAVE_DIR
--checkpoint $CHECKPOINT
```

## 🔍 Test

**Test CP**
```
python -m test \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--polyvore_type nondisjoint \
--task cp \
--batch_sz 64 \
--n_workers 4 \
--result_dir $RESULT_DIR \
--checkpoint $CHECKPOINT
```

**Test FITB**
```
python -m test \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--polyvore_type nondisjoint \
--task fitb \
--batch_sz 32 \
--n_workers 4 \
--result_dir $RESULT_DIR \
--checkpoint $CHECKPOINT
```


## Demo
**Demo CP**
1. Run demo
```
python -m demo \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--task cp \
--checkpoint $CHECKPOINT \
```

**Demo CIR**
1. Generate Item Embeddings
```
python -m generate_embeddings \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--batch_sz 16 \
--checkpoint $CHECKPOINT \
```
2. Build Faiss Index for Similarity Search
```
python -m build_index \
--embeddings_dir ./index \
--save_dir ./index
```
3. Run demo
```
python -m demo \
--model_type clip \
--polyvore_dir $POLYVORE_DIR \
--task cir \
--checkpoint $CHECKPOINT \
--index_dir ./index
```

## 🔔 Note
This is **NON-OFFICIAL** implementation. (The official repo has not been released.)
