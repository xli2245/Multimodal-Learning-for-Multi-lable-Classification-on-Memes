import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import clip
import random
import numpy as np
from PIL import Image
from utils import load_data, find_first_smaller, evaluate, find_first_sum_smaller


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_everything(44)

    # load pretrained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # test data preparation
    test_folder = '../data/test_set_task3/'
    data, classes = load_data(test_folder + 'test_set_task3.txt')
    classes = list(classes)

    length = len(data)
    pred_labels, gold_labels = {}, {}

    # set up the prompt method, 'no' and 'imply' are implemented here
    prompt_method = 'imply' # or 'no'

    # iterate each data point
    for i in range(length):
        print(f'processing {i + 1} / {length}')
        # process image
        image = preprocess(Image.open(test_folder + data[i]['image'])).unsqueeze(0).to(device)

        pred = []
        for j in range(len(classes)):
            # tokenize text
            if prompt_method == 'no':
                text = clip.tokenize([classes[j], 'no ' + classes[j]]).to(device)
                # ************ final results ************
                # Macro_F1:  0.14849517479516472
                # Micro_F1:  0.20749542961608777
            elif prompt_method == 'imply':
                text = clip.tokenize(['The Meme implies ' + classes[j], 'The Meme does not imply ' + classes[j]]).to(device)
                # ************ final results ************
                # Macro_F1:  0.18715687127161748
                # Micro_F1:  0.21447562776957166
            else:
                raise ValueError(f'{prompt_method} is not implemented. Please implement here')

            # make predictions
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                prob = list(probs[0])
            
            if prob[0] >= prob[1]:
                pred.append(classes[j])
        pred_labels[data[i]['id']] = pred
        gold_labels[data[i]['id']] = data[i]['labels']

    # calculate Macro_F1 and Micro_F1
    macro_f1, micro_f1 = evaluate(pred_labels, gold_labels, classes)
    print('************ final results ************')
    print('Macro_F1: ', macro_f1)
    print('Micro_F1: ', micro_f1)