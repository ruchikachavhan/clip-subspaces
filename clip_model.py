import torch
import numpy as np
import torch.nn as nn
import torchvision
import argparse
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
from torchvision.datasets import ImageFolder
# from utils import AverageMeter
from tqdm import tqdm
import os
from torchvision.datasets import CIFAR10
from functools import partial
from utils import NViewsCIFAR10CLIP, NViewsCIFAR10CLIP_C, convert_batch_to_pil, ImageFolderCLIP


class CLIPModelWrapper(nn.Module):
    def __init__(self):
        super(CLIPModelWrapper, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        # convert to cuda
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        image_output = self.model.get_image_features(**inputs)
        # normalise
        image_output = image_output / image_output.norm(dim=-1, keepdim=True)
        return image_output

def run(dataloader, model, texts, transform_names, args):

    average_accuracy = {}
    # for t in transform_names:
        # average_accuracy[t] = AverageMeter(f"accuracy_{t}", ":.4f")

    all_orig_features = []
    all_aug_features = []
    labels = []
    encodings = {}

    for it, (image, class_txt) in tqdm(enumerate(dataloader), total=len(dataloader), desc='Generating features'):
        # convert image to PIL
        if args.dbg and it > 10:
            break
        original_image = convert_batch_to_pil(image['original'])
        augmented_image = convert_batch_to_pil(image['augmented'])

        # if only one transform has been passed
        # if len(image['augmented'].shape) == 4:
        #     augmented_image = [augmented_image]
        
    
        label = [texts.index(f"a photo of a {class_txt[i]}") for i in range(len(class_txt))]

        orig_features = model(original_image)
        aug_features = []
        for k in range(len(augmented_image[0])):
            aug_batch = [augmented_image[i][k] for i in range(len(augmented_image))]
            aug_feat = model(aug_batch)
            aug_features.append(aug_feat)
        aug_features = torch.stack(aug_features).transpose(0, 1)
        print(aug_features.shape)

        all_orig_features.append(orig_features.detach().cpu())
        all_aug_features.append(aug_features.detach().cpu())
        labels.append(torch.tensor(label))

    all_orig_features = torch.cat(all_orig_features)
    all_aug_features = torch.cat(all_aug_features)
    labels = torch.cat(labels)

    print("All original features", all_orig_features.shape)
    print("All augmented features", all_aug_features.shape)
    print("All labels", labels.shape)

    encodings['original'] = all_orig_features
    for t in range(len(transform_names)):
        encodings[transform_names[t]] = all_aug_features[:, t, :]

    return encodings, labels



def main(args):
    if args.eval_dataset == "imagenet":
        dataset = ImageFolderCLIP(root=args.root)
    elif args.eval_dataset == "cifar10":
        dataset = NViewsCIFAR10CLIP(root=args.root, img_size=32)
    elif args.eval_dataset == "cifar10-c":
        dataset = NViewsCIFAR10CLIP_C(root=args.root, img_size=32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = CLIPModelWrapper()
    model.cuda(args.gpu)
    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    texts = [f"a photo of a {t}" for t in dataset.labels2class.values()]

    print("All texts for zero shot classification", texts)

    text_inputs = model.tokenizer(texts, return_tensors="pt", padding=True)
    for k, v in text_inputs.items():
        text_inputs[k] = v.cuda()
    text_output = model.model.get_text_features(**text_inputs)
    text_output = text_output / text_output.norm(dim=-1, keepdim=True)
    # save text output
    print("Saving text features")
    torch.save(text_output, os.path.join(args.output_dir, "text_output.pt"))

    transform_names = list(dataset.transforms_dict.keys())
    del transform_names[transform_names.index('to_tensor')], transform_names[transform_names.index('resize')]
    # , transform_names[transform_names.index('normalize')]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, "encodings.pt")):
        encodings, labels = run(dataloader, model, texts, transform_names, args)
        # save encodings and labels
        torch.save(encodings, os.path.join(args.output_dir, "encodings.pt"))
        torch.save(labels, os.path.join(args.output_dir, "labels.pt"))
    else:
        encodings = torch.load(os.path.join(args.output_dir, "encodings.pt"))
        labels = torch.load(os.path.join(args.output_dir, "labels.pt"))

    accuracies = {}
    text_output = text_output.cpu()
    for k, v in encodings.items():
        similarity = torch.matmul(text_output, v.T)
        similarity = similarity.T
        predicted = torch.argmax(similarity, dim=1)
        accuracy = (predicted == labels).float().mean()
        print(f"Accuracy for {k} is {accuracy}")
        accuracies[k] = accuracy

    print(accuracies)
    # save accuracies
    with open(os.path.join(args.output_dir, "accuracies.txt"), 'w') as f:
        for k, v in accuracies.items():
            f.write(f"{k} : {v}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP demo - Save the output of model for each augmentation')
    parser.add_argument('--eval_dataset', type=str, default=" ", help='Evaluation dataset')
    parser.add_argument('--root', type=str, default="../../image-net100/val", help='path to data')
    parser.add_argument('--output-dir', type=str, default='clip-features-imagenet', help='path to the checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--dbg', action='store_true', help='Debug mode')
    args = parser.parse_args()
    main(args)
