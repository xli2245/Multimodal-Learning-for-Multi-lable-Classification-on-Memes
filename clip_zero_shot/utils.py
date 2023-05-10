import json
import bisect
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(file_path):
    '''
    input
        file_path: file path
    output
        data: dictionary
        classes: all different classes
    description
        load txt file to dictionary, and return all potential classes
    '''
    classes = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for i in range(len(data)):
          classes.extend(data[i]['labels'])
    return data, set(classes)


def find_first_smaller(arr, val):
    '''
    input
        arr: sorted arr in descending order
        val: threshold for selection
    output
        top_idx: the index used for selection in original arr
    description
        find the index of the largest value that is smaller than the given value
    '''
    arr.reverse()
    index = bisect.bisect_left(arr, val)
    arr.reverse()
    top_idx = len(arr) - index
    return top_idx


def find_first_sum_smaller(arr, percentile):
    '''
    input
        arr: sorted arr in descending order
        percentile: threshold for selection
    output
        top_idx: the index used for selection in original arr
    description
        find the largest index such that sum(arr[:top_idx]) >= percentile
    '''
    idx = -1
    cur_sum = 0
    for i in range(len(arr)):
        cur_sum += arr[i]
        if cur_sum >= 0.99:
            idx = i
            break
    top_idx = idx + 1
    return top_idx


def evaluate(pred_labels, gold_labels, CLASSES):
    '''
    input
        pred_labels: {id: pred_labels}
        gold_labels: {id: truth_labels}
        CLASSES: all classes
    output
        macro_f1: the average F1 calculated based on each class
        micro_f1: calculated based on all samples
    description
        calculate the macro_f1 and micro_f1
    '''
    gold = []
    pred = []
    for id in gold_labels:
        gold.append(gold_labels[id])
        pred.append(pred_labels[id])

    mlb = MultiLabelBinarizer()
    mlb.fit([CLASSES])
    gold = mlb.transform(gold)
    pred = mlb.transform(pred)

    macro_f1 = f1_score(gold, pred, average="macro", zero_division=1)
    micro_f1 = f1_score(gold, pred, average="micro", zero_division=1)
    return macro_f1, micro_f1


def get_img_text_path(data):
    '''
    input
        data: dictionary of dataset
    output
        list_image_path: list of image path
        list_txt: list of text shown in each image
        list_label: list of labels for each image
    description
        get the image path, text, and label information for each data point
    '''
    list_image_path, list_txt, list_label = [], [], []
    for i in range(len(data)):
        list_image_path.append(data[i]['image'])
        list_txt.append(data[i]['text'])
        list_label.append(data[i]['labels'])
    return list_image_path, list_txt, list_label


def convert_to_multi_hot_vector(list_label, classes):
    '''
    input
        list_label: dictionary of dataset
        classes: all classes
    output
        labels: list of multi-label in 0-1 format
    description
        convert the labels into multi-hot vector according to given classes
    '''
    dicts = {}
    for i in range(len(classes)):
        dicts[classes[i]] = i
    labels = [[0 for j in range(len(classes))] for i in range(len(list_label))]
    for i in range(len(list_label)):
        cur_labels = list_label[i]
        for cur_label in cur_labels:
            idx = dicts[cur_label]
            labels[i][idx] = 1
    return labels

