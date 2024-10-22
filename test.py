import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from train import get_model
from helpers import dsc, iou, precision, recall
import polar_transformations


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
    dataset = dataset_class('test', polar=args.polar)

    model = get_model(args, dataset_class, device)
    model.to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.train(False)

    _, all_ys, all_predicted_ys = get_predictions(model, dataset, device)

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
        '--model', type=str, choices=train.model_choices, default='liver', help='dataset type'
    )
    parser.add_argument(
        '--dataset', type=str, choices=train.dataset_choices, default='liver', help='dataset type'
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help='multiclass')
    args = parser.parse_args()
    main(args)
