# CS 769 Group Project: Multimodal Classification on Persuasion

## Teammate: Minyi Dai, Yepeng Jin, Xue Li

This repo contains the code for the final project of CS 769 in UW-Madison. The code is built based on the [AIMH model](https://github.com/mesnico/MemePersuasionDetection) used for [the SemEval-2021 Task 6 challenge](https://propaganda.math.unipd.it/semeval2021task6/). We aim to develop an effective multimodal approach for automatically identifying the rhetorical and psychological techniques used in memes by considering both visual and textual elements. Here, Convolutional Neural Networks (CNN) are used for image embedding, and Bidirectional Encoder Representations from Transformers (BERT) are used for text embedding. A double visual-textual transformation (DVTT) model combines these embeddings to classify memes into one of the 22 possible techniques. Additionally, we delve into the impact of model selection strategy, text embedding models, and multimodal fusion methods on model performance.


## Setup
Clone this repo:
```
git clone  https://github.com/xli2245/CS769_Project
```

## Model running

The model training, validation and testing are performed using the [Monai Docker](https://hub.docker.com/r/projectmonai/monai).


### Extract the images in the data folder
```
cd data
for z in *.zip; do unzip $z; done
cd ..
```

### Train, Validation and testing

1.  Model training
```
python train.py --config cfg/config_dual_transformer_task3.yaml --logger_name runs/task3_dual_transformer_weighted --val_step 100 --num_epochs 30
```

2. Model validation

```
python inference.py --checkpoint runs/task3_dual_transformer_weighted/model_latest_fold0.pt --validate
```

3. Model testing
```
python inference.py --checkpoint runs/task3_dual_transformer_weighted/model_latest_fold0.pt --test
```

### Load Model weight directly

The weight of the best model obtained can be found in the zipped folder in [google drive](https://drive.google.com/drive/folders/1Kk_RAtu0HnvQYur3SldjiLbeznCHQJ1K?usp=sharing). The name of the saved model weight is "model_latest_fold0.pt"



