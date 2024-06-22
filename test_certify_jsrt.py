import argparse
import sys
import os.path as p

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

import train
from train import get_model
from helpers import _thresh, dsc, iou, precision, recall
from multiclassify_utils import certify, Log, setup_segmentation_args, str2bool

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def diffusion_args():
    defaults = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule='linear',
        timestep_respacing='',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False
    )
    return defaults


def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def save_segmentation(seg, name, palette, abstain, abstain_mapping):
    path = '/output/test_images_jsrt/' + name
    img = np.asarray(seg[0, ...], dtype=np.uint8)
    I = (img == abstain)
    img[I] = abstain_mapping
    img = Image.fromarray(img)
    img.putpalette(palette)
    img.save(path)


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
    path = '/output/test_images_jsrt/'
    dataset_class = train.get_dataset_class(args)
    dataset = dataset_class('test')

    model = get_model(args, dataset_class, device)
    model.to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.train(False)

    num_classes = 4 + 1  # abstain is an additional class
    abstain = num_classes - 1
    abstain_mapping = 254
    out_confusion_matrix = {}
    out_confusion_matrix['baseline'] = np.zeros((num_classes, num_classes))
    outname_base = "certify"
    tau = args.tau
    n0 = args.n0
    n = args.n
    alpha = args.alpha
    total = n + n0
    sigma = args.sigma
    out_confusion_matrix[f"{outname_base}_holm_{n}_{tau}"] = np.zeros((num_classes, num_classes))
    all_dice_arrays, all_iou_arrays, all_accuracy, all_non_abstain = [], [], [], []

    palette = get_palette(256)
    palette[abstain_mapping * 3 + 0] = 255
    palette[abstain_mapping * 3 + 1] = 255
    palette[abstain_mapping * 3 + 2] = 255

    if args.denoise:
        ddm_args = diffusion_args()
        dd_model, diffusion = create_model_and_diffusion(
            **ddm_args
        )
        dd_model.load_state_dict(
            torch.load("./models/256x256_diffusion_uncond.pt")
        )
        dd_model.cuda()
        dd_model.eval()
        downscale = transforms.Compose([transforms.Resize((256, 256)), ])

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataset):
            channels, ori_height, ori_width = x.shape
            if channels < 3:
                x = x.expand(3, *x.shape[1:])
                x = x - 0.5
            out = []
            remaining = total
            while remaining > 0:
                if args.denoise:
                    x_noised = x + sigma * np.random.randn(*x.shape)
                    img = downscale(img)
                    img = img.unsqueeze(0)
                    t_star = np.abs(diffusion.alphas_cumprod - 1 / (1 + sigma ** 2)).argmin()
                    img = img * np.sqrt(diffusion.alphas_cumprod[t_star])
                    img = img.to(device=device, dtype=torch.float)
                    t = torch.full((1,), t_star).long().cuda()

                    with torch.no_grad():
                        sample = diffusion.p_sample(
                            dd_model,
                            img,
                            t,
                            clip_denoised=False,
                            denoised_fn=None,
                            cond_fn=None,
                            model_kwargs=None,
                        )

                    img_out = sample['pred_xstart']
                    img_out = 0.5 * img_out

                    upscale = transforms.Compose([
                        transforms.Resize((ori_height, ori_width)),
                    ])

                    img_out_up = upscale(img_out)
                    if channels < 3:
                        img_out_up = img_out_up[:, 0:1, :, :]
                    prediction = model(img_out_up.detach())
                else:
                    x_noised = x + sigma * np.random.randn(*x.shape)
                    x_noised = x_noised.to(device=device, dtype=torch.float)
                    prediction = model(x_noised.unsqueeze(0).detach())
                predicted_y = prediction
                predicted_y = predicted_y.detach().cpu()
                predicted_y = torch.argmax(predicted_y, dim=1).numpy().astype(int)
                out.append(predicted_y)
                remaining -= 1
            s = np.concatenate(out)
            s_shape = s.shape
            s = np.reshape(s, (s_shape[0], -1))

            # Certify
            classes_certify, radius, timings = certify(num_classes - 1, s, n0, n, sigma, tau, alpha,
                                                       abstain=abstain, parallel=False, correction='holm')
            classes_certify = np.reshape(classes_certify, (1, *s_shape[1:]))
            save_segmentation(classes_certify, 'classes_certify_{}.png'.format(idx), palette, abstain, abstain_mapping)

            y = y.detach().cpu().numpy()
            label = np.array(y.astype(int))
            size = label.shape
            save_segmentation(label, 'label_{}.png'.format(idx), palette, abstain, abstain_mapping)

            confusion_matrix = get_confusion_matrix(label, classes_certify, size, num_classes)
            I = (classes_certify != abstain)
            cnt_nonabstain = np.sum(I)
            out_confusion_matrix[f"{outname_base}_holm_{n}_{tau}"] += confusion_matrix

            print(f"{outname_base}_holm_{n}_{tau}")
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            pixel_acc = tp.sum() / pos.sum()
            all_accuracy.append(pixel_acc)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            all_iou_arrays.append(IoU_array)
            dice_array = 2 * tp / np.maximum(1.0, pos + res)
            all_dice_arrays.append(dice_array)
            non_abstain = cnt_nonabstain / classes_certify.size
            all_non_abstain.append(non_abstain)
            print('accuracy', pixel_acc)
            print('non-abstain', non_abstain)
            print('IoU', IoU_array)
            print('Dice', dice_array)

    print('Summary')
    print('All accuracy (mean)', np.array(all_accuracy).mean())
    print('All non abstain (mean)', np.array(all_non_abstain).mean())
    print('All IoU', np.array(all_iou_arrays).mean(0))
    print('All Dice', np.array(all_dice_arrays).mean(0))
    return 0


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
    parser.add_argument(
        '--sigma', type=float, default=0.25, help='noise sigma'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.001, help='alpha'
    )
    parser.add_argument(
        '--tau', type=float, default=0.75, help='tau'
    )
    parser.add_argument(
        '--n0', type=int, default=10, help='n0'
    )
    parser.add_argument(
        '--n', type=int, default=100, help='n'
    )
    parser.add_argument(
        '--denoise', action='store_true', help='use denoiser'
    )
    parser.add_argument(
        '--multi',
        action='store_true',
        help='multiclasss')

    args = parser.parse_args()
    print("TEST CERTIFY MULTICLASS")
    print(args)
    main(args)
