# --------------------------------------------------------
# Context Cluster Visualization
# A script to visualize the clustering results of CoC for a given stage, block, head.
# Different layers/heads will present different clustering patterns.
# Licensed under the Apache-2.0 license [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)

# Use case (generated image will saved to images/cluster_vis/{model}):
# python cluster_visualize.py --image {path_to_image} --model {model} --checkpoint {path_to_checkpoint} --stage {stage} --block {block} --head {head}
# --------------------------------------------------------

import models
import timm
import os
import torch
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from torchvision import transforms
from einops import rearrange
import random
from timm.models import load_checkpoint
from torchvision.utils import draw_segmentation_masks

object_categories = []
with open("./imagenet1k_id_to_label.txt", "r") as f:
    for line in f:
        _, val = line.strip().split(":")
        object_categories.append(val)

parser = argparse.ArgumentParser(description='Context Cluster visualization')
parser.add_argument('--image', type=str, default="images/A.JPEG", help='path to image')
parser.add_argument('--shape', type=int, default=224, help='image size')
parser.add_argument('--model', default='coc_tiny_plain', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--stage', default=0, type=int, help='Index of visualized stage, 0-3')
parser.add_argument('--block', default=0, type=int, help='Index of visualized stage, -1 is the last block ,2,3,4,1')
parser.add_argument('--head', default=1, type=int,  help='Index of visualized head, 0-3 or 0-7')
parser.add_argument('--resize_img', action='store_true', default=False, help='Resize img to feature-map size')
parser.add_argument('--checkpoint', type=str, default="coc_tiny_plain.pth.tar", metavar='PATH', help='path to pretrained checkpoint (default: none)')
parser.add_argument('--alpha', default=0.5, type=float, help='Transparent, 0-1')
args = parser.parse_args()
assert args.model in timm.list_models(), "Please use a timm pre-trined model, see timm.list_models()"


# Preprocessing
def _preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,M,D]
    :param x2: [B,N,D]
    :return: similarity matrix [B,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.permute(0, 2, 1))
    return sim


# forward hook function
def get_attention_score(self, input, output):
    x = input[0]  # input tensor in a tuple
    value = self.v(x)
    x = self.f(x)
    x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
    value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
    if self.fold_w > 1 and self.fold_h > 1:
        b0, c0, w0, h0 = x.shape
        assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
            f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
        x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                      f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
        value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
    b, c, w, h = x.shape
    centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
    value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
    b, c, ww, hh = centers.shape
    sim = torch.sigmoid(self.sim_beta +
                        self.sim_alpha * pairwise_cos_sim(
                            centers.reshape(b, c, -1).permute(0, 2, 1),
                            x.reshape(b, c, -1).permute(0, 2,1)
                        )
                    )  # [B,M,N]
    # sololy assign each point to one center
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)  # binary #[B,M,N]
    mask.scatter_(1, sim_max_idx, 1.)  # binary #[B,M,N]
    # changed, for plotting mask.
    mask = mask.reshape(mask.shape[0], mask.shape[1], w, h)  # [(head*fold*fold),m, w,h]
    mask = rearrange(mask, "(h0 f1 f2) m w h -> h0 (f1 f2) m w h",
                     h0=self.heads, f1=self.fold_w, f2=self.fold_h)  # [head, (fold*fold),m, w,h]
    mask_list = []
    for i in range(self.fold_w):
        for j in range(self.fold_h):
            for k in range(mask.shape[2]):
                temp = torch.zeros(self.heads, w * self.fold_w, h * self.fold_h)
                temp[:, i * w:(i + 1) * w, j * h:(j + 1) * h] = mask[:, i * self.fold_w + j, k, :, :]
                mask_list.append(temp.unsqueeze(dim=0))  # [1, heads, w, h]

    mask2 = torch.concat(mask_list, dim=0)  # [ n, heads, w, h]
    global attention
    attention = mask2.detach()


def main():
    global attention
    image, raw_image = _preprocess(args.image)
    image = image.unsqueeze(dim=0)
    model = timm.create_model(model_name=args.model, pretrained=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, True)
        print(f"\n\n==> Loaded checkpoint")
    else:
        print(f"\n\n==> NO checkpoint is loaded")
    model.network[args.stage * 2][args.block].token_mixer.register_forward_hook(get_attention_score)
    out = model(image)
    if type(out) is tuple:
        out = out[0]
    possibility = torch.softmax(out, dim=1).max()
    value, index = torch.max(out, dim=1)
    print(f'==> Prediction is: {object_categories[index]} possibility: {possibility * 100:.3f}%')

    try:
        os.makedirs(f"images/cluster_vis/{args.model}")
    except:
        pass

    image_name = os.path.basename(args.image).split(".")[0]

    from torchvision.io import read_image
    img = read_image(args.image)
    ### deprecated feature resize_img
    # if args.resize_img:
    #     img = F.adaptive_avg_pool2d(img.unsqueeze(dim=0).double(),
    #                                 output_size=(attention.shape[1], attention.shape[2])).squeeze(dim=0)
    #     img = img.to(dtype=torch.uint8)
    # process the attention map
    attention = attention[:, args.head, :, :]
    mask = attention.unsqueeze(dim=0)
    mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5
    # randomly selected some good colors.
    colors = ["brown", "green", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]  # deepskyblue
    if mask.shape[0] == 4:
        colors = colors[0:4]
    if mask.shape[0] > 4:
        colors = colors * (mask.shape[0] // 16)
        random.seed(123)
        random.shuffle(colors)

    img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    save_path = f"images/cluster_vis/{args.model}/{image_name}_Stage{args.stage}_Block{args.block}_Head{args.head}.png"
    cv2.imwrite(save_path, img_with_masks)
    print(f"==> Generated image is saved to: {save_path}")


if __name__ == '__main__':
    main()
