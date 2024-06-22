import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from train import get_model
from helpers import _thresh, dsc, iou, precision, recall


def get_confusion_matrix(label, seg_pred, size, num_class, ignore=-1):
    seg_gt = label

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                i_pred] = label_count[cur_index]
    return confusion_matrix


def get_predictions(model, dataset, device):
    all_xs = []
    all_ys = []
    all_predicted_ys = []

    with torch.no_grad():
        for (x, y) in dataset:
            x = x.to(device)
            prediction = model(x.unsqueeze(0).detach())

            predicted_y = prediction
            predicted_y = predicted_y.squeeze(0).squeeze(0).detach().cpu().numpy()

            all_predicted_ys.append(predicted_y)

            x = x.squeeze(0).detach().cpu().numpy()
            all_xs.append(x)

            y = y.squeeze(0).detach().cpu().numpy()
            all_ys.append(y)

    return all_xs, all_ys, all_predicted_ys


def main(args):
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    dataset_class = train.get_dataset_class(args)
    dataset = dataset_class('test')

    model = get_model(args, dataset_class, device)
    model.to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.train(False)

    num_classes = 2 + 1  # abstain is an additional class
    abstain = num_classes - 1
    out_confusion_matrix = {}
    out_confusion_matrix['baseline'] = np.zeros((num_classes, num_classes))
    outname_base = "certify"
    tau = 0.75
    n = 100
    out_confusion_matrix[f"{outname_base}_holm_{n}_{tau}"] = np.zeros((num_classes, num_classes))
    all_xs, all_ys, all_predicted_ys = [], [], []
    all_dice_arrays, all_iou_arrays, all_accuracy = [], [], []

    with torch.no_grad():
        for (x, y) in dataset:
            x = x.to(device)
            prediction = model(x.unsqueeze(0).detach())

            predicted_y = prediction
            predicted_y = predicted_y.squeeze(0).detach().cpu().numpy()

            all_predicted_ys.append(predicted_y)

            x = x.squeeze(0).detach().cpu().numpy()
            all_xs.append(x)

            y = y.detach().cpu().numpy()
            all_ys.append(y)

            classes_baseline = np.array(_thresh(predicted_y)).astype(int)
            label = np.array(_thresh(y)).astype(int)
            size = label.shape
            print(size)
            confusion_matrix = get_confusion_matrix(label, classes_baseline, size, num_classes)
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            pixel_acc = tp.sum() / pos.sum()
            all_accuracy.append(pixel_acc)
            # mean_acc = (tp / np.maximum(1.0, pos)).mean()
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            all_iou_arrays.append(IoU_array)
            # mean_IoU = IoU_array.mean()
            dice_array = 2 * tp / np.maximum(1.0, pos + res)
            all_dice_arrays.append(dice_array)
            # mean_dice = dice_array.mean()

            print('accuracy', pixel_acc)
            print('IoU', IoU_array)
            print('Dice', dice_array)
            # print('mean_acc', mean_acc)
            # print('mean_IoU', mean_IoU)
            # print('Mean dice', mean_dice)
            out_confusion_matrix["baseline"] += confusion_matrix

    print('baseline')
    print('All accuracy (mean)', np.array(all_accuracy).mean())
    print('All IoU', np.array(all_iou_arrays).mean(0))
    print('All Dice', np.array(all_dice_arrays).mean(0))

    dscs = np.array([dsc(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
    ious = np.array([iou(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
    precisions = np.array([precision(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])
    recalls = np.array([recall(all_predicted_ys[i], all_ys[i]) for i in range(len(all_ys))])

    print(
        f'DSC: {dscs.mean():.4f} | IoU: {ious.mean():.4f} | prec: {precisions.mean():.4f} | rec: {recalls.mean():.4f}')
    return dscs.mean(), ious.mean(), precisions.mean(), recalls.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Testing'
    )
    parser.add_argument(
        '--weights', type=str, help='path to weights'
    )
    parser.add_argument(
        '--model', type=str, choices=train.model_choices, default='unet', help='model type'
    )
    parser.add_argument(
        '--dataset', type=str, choices=train.dataset_choices, default='liver', help='dataset type'
    )

    args = parser.parse_args()
    main(args)
