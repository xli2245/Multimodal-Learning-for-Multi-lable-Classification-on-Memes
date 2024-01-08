# Multimodal Classification on Persuasion
### Teammate: Xue Li, Minyi Dai, Yepeng Jin, 
The code is built based on the [AIMH model](https://github.com/mesnico/MemePersuasionDetection) used for [the SemEval-2021 Task 6 challenge](https://propaganda.math.unipd.it/semeval2021task6/). We aim to develop an effective multimodal approach for automatically identifying the rhetorical and psychological techniques used in memes by considering both visual and textual elements. To achieve this, we utilize Convolutional Neural Networks (CNN) for image embedding and Bidirectional Encoder Representations from Transformers (BERT) for text embedding. We explore and compare various model fusion strategies, such as arithmetic operations, single transformers, and dual transformers. Furthermore, we investigate the impact of alternative text embedding models and experiment with methods like CLIP and ChatGPT. 
## Table of Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [Multimodal classification framework](#multimodal-classification-framework)
  - [Environment](#environment)
  - [Model running](#model-running)
  - [Model weight](#model-weight)
- [CLIP model](#clip-model)
  - [Introduction](#introduction)
  - [CLIP Model running](#clip-model-running)
- [ChatGPT as Text Annotator](#chatgpt-as-text-annotator)
- [Others](#others)
## Setup
Clone this repo:
```
git clone  https://github.com/xli2245/Multimodal-Learning-for-Multi-lable-Classification-on-Memes
```
## Dataset
Dataset is downloaded from [SemEval-2021 Task6](https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus) and uploaded to the data folder.
To extract the images in the data folder
```
cd data
for z in *.zip; do unzip $z; done
cd ..
```
## Multimodal classification framework
![Main framework](https://github.com/xli2245/CS769_Project/blob/main/main%20framework.png)
### Environment
The model training, validation and testing are performed using the [MONAI Docker](https://hub.docker.com/r/projectmonai/monai).
### Model running
1.  Model training
```
python train.py --config cfg/config_task3_simple.yaml --logger_name runs/task3_single_transformer --val_step 200 --num_epochs 40
```
2. Model validation
```
python inference.py --checkpoint runs/saved_transfer_model/saved_best_model.pt --validate
```
3. Model testing
```
python inference.py --checkpoint runs/saved_transfer_model/saved_best_model.pt --test
```
### Model weight
The weight of the best model obtained can be found in the [google drive](https://drive.google.com/drive/folders/1Kk_RAtu0HnvQYur3SldjiLbeznCHQJ1K?usp=sharing). The name of the saved model weight is "saved_transfer_model.tar.gz".

To unzip the model weight
```
tar -xvf ./saved_transfer_model.tar.gz
```

The downloaded pretrained models (BERT, RoBERTa ...) can be also be found in the google drive folder. The name is "model.tar.gz".

## CLIP model
### Introduction
The [Contrastive Language-Image Pre-training (CLIP) model](https://github.com/openai/CLIP) is a pre-trained visual-language model that utilizes image-text pairs with contrastive loss. The model typically takes an image and multiple text inputs, generating a similarity score between the image and each text, ranked in descending order. We use the original meme images, complete with text, as input for the image component. To accommodate the multi-label nature of this problem, we experiment with various methods for constructing text inputs and processing the results. 
### CLIP model running

For Method 1 and Method 2, which use two contradictory texts as inputs and perform 22 predictions for 22 techniques. 
```
python sgl_class_based_clip.py 
```
For Method 3 and Method 4, which inputs texts for 22 techniques at the same time.
```
python all_class_based_clip.py
```

## ChatGPT as Text Annotator

Run the gpt4_classification.ipynb, replace the key with your own openai key.
In the code:  

  &nbsp;&nbsp; There are two versions of **tech_20** for detailed or simplified description.  
 &nbsp;&nbsp; Active **instruction3** if you want gpt-4 to return the confident score of the classifcation

## Others
Three types of data augmentation, random insertion, random substitution and back translation are implemented in the file "traindata_augment.ipynb" in the folder "data augmentation". This can be run using google colab. Note that "techniques_list_task3.txt", "training_set_task3.txt", "dev_set_task3_labeled.txt", "folds.json" are needed for the data augmentation.
