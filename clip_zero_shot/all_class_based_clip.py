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

    threshold = 0.1
    length = len(data)
    pred_labels, gold_labels = {}, {}

    # set up the prompt method, 'no' and 'imply' are implemented here
    prompt_method = 'imply'

    # set up the top_idx selection strategy, 'default', 'minimum threshold', and 'sum threshold' are implemented here
    top_idx_selection = 'default'

    # iterate each data point
    for i in range(length):
        print(f'processing {i + 1} / {length}')

        # process image
        image = preprocess(Image.open(test_folder + data[i]['image'])).unsqueeze(0).to(device)

        # tokenize text
        if prompt_method == 'no':
            text = clip.tokenize(classes).to(device)
        elif prompt_method == 'imply':
            text = clip.tokenize([f'The Meme implies {cur_class}' for cur_class in classes]).to(device)
        else:
            raise ValueError(f'{prompt_method} is not implemented. Please implement here')

        # make predictions
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            prob = list(probs[0])

        sort_class = [x for _, x in sorted(zip(prob, classes), reverse=True)]
        prob.sort(reverse=True)
        if top_idx_selection == 'default':
            top_idx = 8
            # ************ final results (align with no prompt) ************
            # Macro_F1:  0.14165401659135549
            # Micro_F1:  0.18370230805463963
            # ************ final results (align with imply prompt) ************
            # Macro_F1:  0.1507112732319879
            # Micro_F1:  0.17993405558172396
        elif top_idx_selection == 'minimum threshold':
            threshold = 0.1
            top_idx = find_first_smaller(prob, threshold)
            # ************ final results (align with no prompt) ************
            # Macro_F1:  0.10955395180257436
            # Micro_F1:  0.14528301886792452
            # ************ final results (align with imply prompt) ************
            # Macro_F1:  0.09953610210635863
            # Micro_F1:  0.10848126232741616
        elif top_idx_selection == 'sum threshold':
            threshold = 0.99
            top_idx = find_first_sum_smaller(prob, threshold)
            # ************ final results (align with no prompt) ************
            # Macro_F1:  0.17934407268176827
            # Micro_F1:  0.2131645569620253
            # ************ final results (align with imply prompt) ************
            # Macro_F1:  0.18941959557293683
            # Micro_F1:  0.21856424325560128
        else:
            raise ValueError(f'{top_idx_selection} is not implemented. Please implement here')

        pred_labels[data[i]['id']] = sort_class[:top_idx]
        gold_labels[data[i]['id']] = data[i]['labels']

    # calculate Macro_F1 and Micro_F1
    macro_f1, micro_f1 = evaluate(pred_labels, gold_labels, classes)
    print('************ final results ************')
    print('Macro_F1: ', macro_f1)
    print('Micro_F1: ', micro_f1)