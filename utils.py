import torch
import os
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.distributed as dist
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from torchvision.datasets import ImageFolder
import random
from clip_model import CLIPModelWrapper
from torchvision.datasets import CIFAR10


class ImageFolderCLIP(ImageFolder):
    def __init__(self, root, num_groups=6):
        super().__init__(root=root)
        self.root = root
        self.mean = [0.485, 0.456, 0.406]  # OFA model mean
        self.std = [0.229, 0.224, 0.225]  # OFA model std
        self.resolution = 224  # OFA model resolution
        if num_groups == 6:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                'h_flip': transforms.RandomHorizontalFlip(p=1.0),
                'rotation': transforms.RandomRotation(270),
                'translation': transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                'color_jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                'grayscale': transforms.Grayscale(num_output_channels=3),
                'blur': transforms.GaussianBlur(kernel_size=5),
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
            }
        elif num_groups == 3:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                'rotation': transforms.RandomRotation(270),
                'grayscale': transforms.Grayscale(num_output_channels=3),
                'color_jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
            }
        elif num_groups == 2:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                'rotation': transforms.RandomRotation(270),
                'grayscale': transforms.Grayscale(num_output_channels=3),
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
            }
        elif num_groups == 1:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),                
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
            }
        # No augmentation applied, resizing + center crop + norm
        self.base_transform = transforms.Compose([
            self.transforms_dict['resize'],
            transforms.CenterCrop(self.resolution),
            self.transforms_dict['to_tensor'],
        ])

        self.augmentations = []
        for key in list(self.transforms_dict.keys()):
            if key not in ['to_tensor', 'normalize', 'resize']:
                t = transforms.Compose([
                    self.transforms_dict['resize'],
                    transforms.CenterCrop(self.resolution),
                    self.transforms_dict[key],
                    self.transforms_dict['to_tensor'],
                ])
                self.augmentations.append(t)

        self.labels2class_txt = self.root.split("/val")[0] + '/labels.txt'
        self.labels2class = {}
        with open(self.labels2class_txt, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.labels2class[line[0]] = " ".join(line[2].split('_'))
        print(self.labels2class)

    def __getitem__(self, index):
        #  Get name of image
        image_name = self.samples[index][0]
        # get folder name
        folder_name = image_name.split('/')[-2]
        img, _ = super().__getitem__(index)
        img = img.convert('RGB')
        data = {}
        data['original'] = self.base_transform(img)
        data['augmented'] = []
        for t in self.augmentations:
            data['augmented'].append(t(img))

        data['augmented'] = torch.stack(data['augmented'])
        return data, self.labels2class[folder_name]

# make NviewsDataset for CIFAR10
class NViewsCIFAR10CLIP(CIFAR10):
    ''' CIFAR10 data loader
    Returns
    Dict: Original image, augmented image, negative-augmented image
    '''
    def __init__(self, root, img_size=224, num_groups=6):
        super().__init__(root=root, train=False, transform=None, target_transform=None)
        self.root = root
        self.mean = [0.485, 0.456, 0.406]  # OFA model mean
        self.std = [0.229, 0.224, 0.225]  # OFA model std
        self.resolution = img_size  # OFA model resolution
        self.labels2class = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer', '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', '9': 'truck'}

        if num_groups == 6:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                'h_flip': transforms.RandomHorizontalFlip(p=1.0),
                'rotation': partial(transforms.functional.rotate, angle=90),
                'translation': transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                'color_jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                'grayscale': transforms.Grayscale(num_output_channels=3),
                'blur': transforms.GaussianBlur(kernel_size=5),
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
                }
        elif num_groups == 3:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                'rotation': transforms.RandomRotation(270),
                'grayscale': transforms.Grayscale(num_output_channels=3),
                'color_jitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
                }
        elif num_groups == 2:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                'rotation': transforms.RandomRotation(270),
                'grayscale': transforms.Grayscale(num_output_channels=3),
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
            }
        elif num_groups == 1:
            self.transforms_dict = {
                'resize': transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                'to_tensor': transforms.ToTensor(),
                'normalize': transforms.Normalize(mean=self.mean, std=self.std)
            }
        # No augmentation applied, resizing + center crop + norm
        self.base_transform = transforms.Compose([
            self.transforms_dict['resize'],
            transforms.CenterCrop(self.resolution),
            self.transforms_dict['to_tensor'],
            # self.transforms_dict['normalize']
        ])

        self.augmentations = []
        for key in list(self.transforms_dict.keys()):
            if key not in ['to_tensor', 'normalize', 'resize']:
                t = transforms.Compose([
                    self.transforms_dict['resize'],
                    transforms.CenterCrop(self.resolution),
                    self.transforms_dict[key],
                    self.transforms_dict['to_tensor'],
                    # self.transforms_dict['normalize']
                ])
                self.augmentations.append(t)

    def __getitem__(self, index):
        #  Get name of image
        img, label = super().__getitem__(index)
        img = img.convert('RGB')
        data = {}
        data['original'] = self.base_transform(img)
        data['augmented'] = []
        for t in self.augmentations:
            data['augmented'].append(t(img))

        data['augmented'] = torch.stack(data['augmented'])
        return data, self.labels2class[str(label)]


class NViewsCIFAR10CLIP_C(torch.utils.data.Dataset):
    ''' CIFAR10-C data loader
    Returns
    Dict: Original image, augmented image, negative-augmented image
    '''
    def __init__(self, root, img_size=224, num_groups=6):
        super().__init__()
        self.root = root
        self.mean = [0.485, 0.456, 0.406]  # OFA model mean
        self.std = [0.229, 0.224, 0.225]  # OFA model std
        self.resolution = img_size  # OFA model resolution
        self.labels2class = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer', '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', '9': 'truck'}

        # read numpy file
        self.data = {}
        self.transforms_dict = {}
        for f in os.listdir(root):
            if f not in ['data.npy', 'labels.npy']:
                eval_transform = f.split('.')[0]
                self.data[eval_transform] = np.load(os.path.join(root, f))
                # select first 10000 samples
                self.data[eval_transform] = self.data[eval_transform][:10000]
                self.transforms_dict[eval_transform] = eval_transform

        self.original_data = np.load(os.path.join(root, 'data.npy'))
        self.labels = np.load(os.path.join(root, 'labels.npy'))

        self.transforms_dict['resize'] = transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC)
        self.transforms_dict['to_tensor'] = transforms.ToTensor()
        
        self.base_transform = transforms.Compose([
            self.transforms_dict['resize'],
            transforms.CenterCrop(self.resolution),
            self.transforms_dict['to_tensor'],
            # self.transforms_dict['normalize']
        ])
    
    def __len__(self):
        return len(self.data['gaussian_noise'])
    
    def __getitem__(self, index):
        corrupted_images = []
        data = {}
        data['augmented'] = []
        for k, v in self.data.items():
            corrupted_images.append(v[index])
            data['augmented'].append(self.base_transform(Image.fromarray(v[index])))

        data['augmented'] = torch.stack(data['augmented'])
        original_img = self.original_data[index]
        label = self.labels[index]
        data['original'] = self.base_transform(Image.fromarray(original_img))
        return data, self.labels2class[str(label)]
        


def convert_batch_to_pil(batch):
    if len(batch.shape) == 4:
        batch = [transforms.ToPILImage()(i) for i in batch]
    elif len(batch.shape) == 5:
        batch = [[transforms.ToPILImage()(i) for i in b] for b in batch]
    return batch
    

def multiply_elementwise(states, mask):
    # element wise multiplication of mask and hidden states
    masked_states = states * mask
    return masked_states

def cosine_sim_matrix(a, b):
    # normalise rows of both
    a = a / torch.norm(a, p=2, dim=1, keepdim=True)
    b = b / torch.norm(b, p=2, dim=1, keepdim=True)
    # calculate cosine similarity matrix
    dot_product = torch.matmul(a, b.T)
    return dot_product

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset the meter"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        """Update the meter"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class OutputProjectionCLIP(nn.Module):
    def __init__(self, text_feats):
        super().__init__()
        self.text_feats = text_feats
        self.temperature = 10.0

    def forward(self, x, use_temperature=False):
        # covert to cpu
        x = x.cpu()
        # normalise
        x = x / x.norm(dim=-1, keepdim=True)
        self.text_feats = self.text_feats.cpu()
        similarity = torch.matmul(self.text_feats, x.T)
        similarity = similarity.T
        if use_temperature:
            similarity /= self.temperature
        return similarity

def get_model(args, ngpus_per_node, gpu):
    if args.model == 'clip':
        model = CLIPModelWrapper()
        text_features = os.path.join(args.output_dir, 'text_output.pt')
        model.output_projection = OutputProjectionCLIP(torch.load(text_features))
    else:
        model = OFAModel.from_pretrained(args.ckpt_dir, use_cache=False)
    if args.dist:
        args.local_rank = args.local_rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init_method,
                                world_size=args.world_size, rank=args.local_rank)
        torch.distributed.barrier()
        torch.cuda.set_device(args.local_rank)
        model = model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        assert args.gpu is not None and args.gpu >= 0
        model = model.cuda(args.gpu)

    model.eval()
    return model

def get_dataloader(args,  train=False):
    if args.eval_dataset == 'imagenet':
        dataset = NViewsImageNetDataset(root=args.data_dir, eval_transform=args.eval_transform, num_groups=args.num_aug_groups)
    elif args.eval_dataset == 'cifar10':
        dataset = NViewsCIFAR10CLIP(root=args.data_dir, eval_transform=args.eval_transform, num_groups=args.num_aug_groups)
    elif args.eval_dataset == 'cifar10-c':
        dataset = NViewsCIFAR10CLIP_C(root=args.data_dir, eval_transform=args.eval_transform, num_groups=args.num_aug_groups)
    if args.dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                                                 num_workers=0, sampler=sampler)
        #  Dataloader
        args.batch_size = int(args.batch_size / args.world_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                 drop_last=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=0,
                                                 drop_last=True)
    return dataloader

def calculate_invariances(orig_features, aug_features, cosine_sim=nn.CosineSimilarity(dim=0)):
    mean = orig_features.mean(0)
    covariance = np.cov(orig_features.numpy(), rowvar=False)
    covariance = covariance + 1e-10 * np.eye(covariance.shape[0])

    inv_covariance = np.linalg.inv(covariance)
    cholesky_matrix, _ = torch.linalg.cholesky_ex(torch.from_numpy(inv_covariance).to(torch.float32))

    sim = 0.0
    for b in range(orig_features.shape[0]):
        a = (mean - orig_features[b]) @ cholesky_matrix
        b = (mean - aug_features[b]) @ cholesky_matrix
        sim += cosine_sim(a, b)
    sim /= orig_features.shape[0]

    return sim.item()