import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import tqdm
import yaml
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

import torch

import logging
from torch.utils.tensorboard import SummaryWriter

import argparse

from dataset import SemEvalDataset, Collate
from models import MemeMultiLabelClassifier
from sampler import MultilabelBalancedRandomSampler

from scorer.task1_3 import evaluate
from format_checker.task1_3 import read_classes
from shutil import copyfile


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=200, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--test_step', default=100000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/simple_transformer',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    parser.add_argument('--load-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--config', default='./cfg/config_task3_simple.yaml', type=str, help="Which configuration to use. See into 'config' folder")
    parser.add_argument('--cross-validation', action='store_true', help='Enables cross validation')

    opt = parser.parse_args()
    print(opt)
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if opt.cross_validation:
        # read splits from file
        with open('data/folds.json', 'r') as f:
            folds = json.load(f)
            num_folds = len(folds)
        for fold in tqdm.trange(num_folds):
            train(opt, config, val_fold=fold)
    else:
        # train using fold 0 as validation fold
        train(opt, config, val_fold=0)

def train(opt, config, val_fold=0):
    if 'task' not in config['dataset']:
        config['dataset']['task'] = 3 # for back compatibility
        print('Manually assigning: task 3')

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()

    # Dump configuration to experiment path
    copyfile(opt.config, os.path.join(experiment_path, 'config.json'))

    # Load Vocabulary Wrapper

    # Load data loaders
    test_transforms = T.Compose([T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    train_transforms = T.Compose([T.Resize(256),
                    T.RandomCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    train_dataset = SemEvalDataset(config, split='train', transforms=train_transforms, val_fold=val_fold)
    val_dataset = SemEvalDataset(config, split='val', transforms=test_transforms, val_fold=val_fold)

    id_intersection = set([x['id'] for x in train_dataset.targets]).intersection([x['id'] for x in val_dataset.targets])
    assert len(id_intersection) == 0

    if config['dataset']['task'] == 3:
        classes = read_classes('techniques_list_task3.txt')
    elif config['dataset']['task'] == 1:
        classes = read_classes('techniques_list_task1-2.txt')

    collate_fn = Collate(config, classes)
    if 'balanced-sampling' in config['training'] and config['training']['balanced-sampling']:
        classes_ids = [[train_dataset.class_list.index(x) for x in info['labels']] for info in train_dataset.targets]
        labels = np.zeros((len(classes_ids), len(train_dataset.class_list)))
        for l, c in zip(labels, classes_ids):
            l[c] = 1
        sampler = MultilabelBalancedRandomSampler(labels)
    else:
        sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True if sampler is None else False, num_workers=opt.workers, collate_fn=collate_fn, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['bs'], shuffle=False,
                                  num_workers=opt.workers, collate_fn=collate_fn)

    # Construct the model
    model = MemeMultiLabelClassifier(config, labels=classes)
    if torch.cuda.is_available() and not (opt.resume or opt.load_model):
        model.cuda()

    # Construct the optimizer
    if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune']:
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if 'textual_module' not in n and 'visual_module' not in n], lr=config['training']['lr'])
    else:
        if config['dataset']['task'] == 3:
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'textual_module' not in n and 'visual_module' not in n]},
                {'params': model.textual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']},
                {'params': model.visual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']}]
                , lr=config['training']['lr'])
        elif config['dataset']['task'] == 1:
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if
                            'textual_module' not in n and 'visual_module' not in n]},
                {'params': model.textual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']}]
                , lr=config['training']['lr'])
    # LR scheduler
    scheduler_name = config['training']['scheduler']
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=config['training']['gamma'], milestones=config['training']['milestones'])
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))


    # # optionally resume from a checkpoint
    start_epoch = 0
    model.train()

    # Train loop
    mean_loss = 0
    progress_bar = tqdm.trange(start_epoch, opt.num_epochs)
    progress_bar.set_description('Train')
    best_f1 = 0.0
    best_sum = 0.0
    best_loss = float("Inf")
    cnt = 0
    micro_f1_saved_epoch = -1
    f1_sum_saved_epoch = -1
    for epoch in progress_bar:
        cnt += 1
        for it, (image, text, text_len, labels, ids) in enumerate(train_dataloader):
            global_iteration = epoch * len(train_dataloader) + it

            if torch.cuda.is_available():
                image = image.cuda() if image is not None else None
                text = text.cuda()
                labels = labels.cuda()

            # forward the model
            optimizer.zero_grad()

            loss = model(image, text, text_len, labels)
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()

            if global_iteration % opt.log_step == 0:
                mean_loss /= opt.log_step
                progress_bar.set_postfix(dict(loss='{:.2}'.format(mean_loss)))
                mean_loss = 0

            tb_logger.add_scalar("Training/Epoch", epoch, global_iteration)
            tb_logger.add_scalar("Training/Loss", loss.item(), global_iteration)
            tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], global_iteration)

        # validate (using different thresholds)
        metrics = validate(val_dataloader, model, classes, thresholds=[0.3])
        tb_logger.add_scalars("Validation/F1", metrics, global_iteration)
        training_loss = loss.item()
        val_loss = metrics['val_loss'] / len(val_dataloader)
        cur_F1_micro = metrics['microF1_thr=0.3']
        cur_F1_sum = metrics['macroF1_thr=0.3'] + metrics['microF1_thr=0.3']
        print(f'Epoch {cnt}: loss: {training_loss}, val loss: {val_loss}, micro F1: {cur_F1_micro}, F1 sum: {cur_F1_sum}')
        print(f'last micro f1 saved: epoch {micro_f1_saved_epoch}, value: {best_f1}')
        print(f'last f1 sum saved: epoch {f1_sum_saved_epoch}, value: {best_sum}')

        # save best model
        if cur_F1_micro >= best_f1 or (cnt - micro_f1_saved_epoch >= 5 and cur_F1_micro - best_f1 > -0.005):
            print(f'Epoch {cnt}: Saving the highest microF1 model...')
            best_f1 = cur_F1_micro
            micro_f1_saved_epoch = cnt
            checkpoint = {
                'cfg': config,
                'epoch': epoch,
                'model': model.joint_processing_module.state_dict() if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune'] else model.state_dict()}
            latest = os.path.join(experiment_path, 'model_best_micro_fold{}.pt'.format(val_fold))
            torch.save(checkpoint, latest)

        if cur_F1_sum >= best_sum or (cnt - f1_sum_saved_epoch >= 5 and cur_F1_sum - best_sum > -0.005):
            best_sum = cur_F1_sum
            f1_sum_saved_epoch = cnt
            print(f'Epoch {cnt}: Saving the highest sum F1 model ...')
            checkpoint = {
                'cfg': config,
                'epoch': epoch,
                'model': model.joint_processing_module.state_dict() if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune'] else model.state_dict()}
            latest = os.path.join(experiment_path, 'model_best_sum_fold{}.pt'.format(val_fold))
            torch.save(checkpoint, latest)

        # save the last model
        if cnt == opt.num_epochs:
            print('Saving lastest model...')
            checkpoint = {
                'cfg': config,
                'epoch': epoch,
                'model': model.joint_processing_module.state_dict() if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune'] else model.state_dict()}
            latest = os.path.join(experiment_path, 'model_latest_fold{}.pt'.format(val_fold))
            torch.save(checkpoint, latest)

        scheduler.step()


def validate(val_dataloader, model, classes_list, thresholds=[0.3, 0.5, 0.8]):
    model.eval()
    predictions = []
    metrics = {}
    progress_bar = tqdm.tqdm(thresholds)
    progress_bar.set_description('Validation')
    val_loss = 0.0
    for thr in progress_bar:
        for it, (image, text, text_len, labels, ids) in enumerate(val_dataloader):
            if torch.cuda.is_available():
                image = image.cuda() if image is not None else None
                text = text.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                pred_classes, cur_val_loss = model(image, text, text_len, labels, inference_threshold=thr)
                val_loss += cur_val_loss.item()

            for id, labels in zip(ids, pred_classes):    # loop over every element of the batch
                predictions.append({'id': id, 'labels': labels})

        macro_f1, micro_f1 = evaluate(predictions, val_dataloader.dataset.targets, classes_list)
        metrics['macroF1_thr={}'.format(thr)] = macro_f1
        metrics['microF1_thr={}'.format(thr)] = micro_f1
        metrics['val_loss'] = val_loss

    model.train()
    return metrics

if __name__ == '__main__':
    seed_everything(44)
    main()

