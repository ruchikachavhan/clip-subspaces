import os
import torch
import json
import pandas as pd
import argparse
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.multiprocessing as mp
from transformers import OFATokenizer
from torch.autograd import Variable
from utils import AverageMeter, get_model, get_dataloader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from statsmodels.stats.contingency_tables import mcnemar



# Reference - https://github.com/james-oldfield/PoS-subspaces/tree/main
def orthogonal_projection(s, w, device='cpu'):
    """Orthogonally project the (n+1)-dimensional vectors w onto the tangent space T_sS^n.

    Args:
        s (torch.Tensor): point on S^n
        w (torch.Tensor): batch of (n+1)-dimensional vectors to be projected on T_sS^n

    Returns:
        Pi_s(w) (torch.Tensor): orthogonal projections of w onto T_sS^n

    """
    # Get dimensionality of the ambient space (dim=n+1)
    dim = s.shape[0]

    # Calculate orthogonal projection
    I_ = torch.eye(dim, device=device)
    P = I_ - s.unsqueeze(1) @ s.unsqueeze(1).T

    return w.view(-1, dim) @ P.T

# Reference - https://github.com/james-oldfield/PoS-subspaces/tree/main
def logarithmic_map(s, q, epsilon=torch.finfo(torch.float32).eps):
    """Calculate the logarithmic map of a batch of sphere points q onto the tangent space TsS^n.

    Args:
        s (torch.Tensor): point on S^n defining the tangent space TsS^n
        q (torch.Tensor): batch of points on S^n
        epsilon (uint8) : small value to prevent division by 0

    Returns:
        log_s(q) (torch.Tensor): logarithmic map of q onto the tangent space TsS^n.

    """
    torch._assert(len(s.size()) == 1, 'Only a single point s on S^n is supported')
    dim = s.shape[0]
    q = q.view(-1, dim)  # ensure batch dimension
    q = q / torch.norm(q, p=2, dim=-1, keepdim=True)  # ensure unit length

    pi_s_q_minus_s = orthogonal_projection(s, (q - s))

    return (torch.arccos(torch.clip((q * s).sum(axis=-1), -1.0, 1.0)).unsqueeze(1)) * pi_s_q_minus_s / \
        (torch.norm(pi_s_q_minus_s, p=2, dim=1, keepdim=True) + epsilon)


# Reference - https://github.com/james-oldfield/PoS-subspaces/tree/main
def exponential_map(s, q):
    """Calculate the exponential map at point s for a batch of points q in the tangent space TsS^n.

    Args:
        s (torch.Tensor): point on S^n defining the tangent space TsS^n
        q (torch.Tensor): batch of points in TsS^n.

    Returns:
        exp_s(q) (torch.Tensor): exponential map of q from points in the tangent space TsS^n to S^n.

    """
    torch._assert(len(s.size()) == 1, 'Only a single point s on S^n is supported')
    dim = s.shape[0]
    q = q.view(-1, dim)  # ensure batch dimension

    q_norm = torch.norm(q, p=2, dim=1).unsqueeze(1)
    out = torch.cos(q_norm) * s + torch.sin(q_norm) * q / q_norm
    return out / torch.norm(out, p=2, dim=-1, keepdim=True)


# Reference - https://github.com/james-oldfield/PoS-subspaces/tree/main
def calculate_intrinstic_mean(data, iters=1, lr=1.000, init=None):
    """Calculate the intrinsic mean"""
    mean = data[0] if init is None else init  # init with first datapoint if not specified

    with torch.no_grad():
        for i in range(iters):
            grad = torch.mean(logarithmic_map(mean, data), dim=0)
            mean = exponential_map(mean, lr * grad).squeeze()
    return mean / torch.norm(mean, p=2)

def mcnemar_test(model1_predictions, model2_predictions):
    '''
        Statistical test to compare the performance of two models
        Input : a binary list of accuracy for two models - size is (n)
        Output : p-value of the test
    '''
    # Ensure both lists have the same length
    assert len(model1_predictions) == len(model2_predictions), "Input lists must have the same length"

    # Initialize counters
    a, b, c, d = 0, 0, 0, 0

    # Count the number of samples for each case
    for pred1, pred2 in zip(model1_predictions, model2_predictions):
        if pred1 == 1 and pred2 == 1:
            a += 1
        elif pred1 == 1 and pred2 == 0:
            b += 1
        elif pred1 == 0 and pred2 == 1:
            c += 1
        elif pred1 == 0 and pred2 == 0:
            d += 1

    m = mcnemar([[d, c], 
                 [b, a]], exact=False, correction=False)

    return m.statistic, m.pvalue


class Model():
    '''
        Class to learn the subspace with closed form solution
    '''
    def __init__(self, model, dataloader, device, input_ids, decoder_input):
        self.E = {}  # stores the encodings for the target word categories
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.input_ids = input_ids
        self.decoder_input = decoder_input

    def encode(self, dataloader, model, args):
        '''
            Encode the original and augmented images
            params : dataloader, model, args
            return : encodings
        '''
        original_features, augmented_features = [], []
        transform_list = list(dataloader.dataset.transforms_dict.keys())
        del transform_list[transform_list.index('resize')], transform_list[transform_list.index('to_tensor')], transform_list[transform_list.index('normalize')]

        for iter, data in tqdm(enumerate(dataloader), total=len(dataloader),
                               desc='Learning subspace with closed form solution'):
            if args.dbg and iter > 10:
                break
            orig_image = data['original'].cuda(args.gpu)
            aug_image = data['augmented'].cuda(args.gpu)

            if args.dbg:
                print("orig_image", orig_image.shape)
                print("aug_image", aug_image.shape)

            orig_outputs = model(input_ids=self.input_ids, patch_images=orig_image,
                                 patch_masks=torch.tensor([True]), output_hidden_states=True,
                                 decoder_input_ids=self.decoder_input, output_attentions=True).logits.squeeze(1).detach().cpu()
            # orig_outputs /= torch.norm(orig_outputs, p=2, dim=-1).unsqueeze(1)
            aug_outputs = []
            for b in range(aug_image.shape[1]):
                aug_out = model(input_ids=self.input_ids, patch_images=aug_image[:, b, :, :, :],
                                patch_masks=torch.tensor([True]), output_hidden_states=True,
                                decoder_input_ids=self.decoder_input, output_attentions=True).logits.squeeze(1).detach().cpu()
                # aug_out /= torch.norm(aug_out, p=2, dim=-1).unsqueeze(1)
                aug_outputs.append(aug_out)

            aug_outputs = torch.stack(aug_outputs)
            aug_outputs = torch.transpose(aug_outputs, 0, 1)

            original_features.append(orig_outputs)
            augmented_features.append(aug_outputs)

        original_features = torch.cat(original_features, dim=0)
        augmented_features = torch.cat(augmented_features, dim=0)

        self.E['original'] = original_features
        for i in range(augmented_features.shape[1]):
            self.E[transform_list[i]] = augmented_features[:, i, :]

        return self.E

    def pca(self, encodings, num_principal_components):
        '''
        Perform PCA of feature vectors and project onto subspace spanned by principal components
        :param encodings:
        :return: Eigenvectors, eigenvalues, projected encodings (after change of coordinate space), encodings for tsne plot
        '''
        all_eigenvectors, all_eigenvalues = {}, {}
        encodings_pre = {}
        encodings_tsne, labels = [], []
        for key in encodings.keys():
            # standardise the data
            mean = torch.mean(encodings[key], dim=0)
            scaled_data = (encodings[key] - mean)
            # calculate covariance matrix
            cov_matrix = scaled_data.T @ scaled_data
            cov_matrix /= (scaled_data.shape[0] - 1)
            # calculate eigenvalues and eigenvectors, torch.linalg.eigh returns eigenmatrix where columns are eigenvectors
            eig_values, eig_vectors = torch.linalg.eigh(cov_matrix)
            # covert to real
            eig_values = torch.real(eig_values)
            eig_vectors = torch.real(eig_vectors)
            # sort eigenvalues and eigenvectors in descending order
            idx = torch.sort(eig_values, descending=True).indices
            eig_values = eig_values[idx]
            eig_vectors = eig_vectors[:, idx]
            V = eig_vectors 
            # choose largest singular values
            num_components = [i for i in range(num_principal_components)]
            rank = torch.linalg.matrix_rank(V)
            # project onto subspace spanned by principal components
            principal_components = V[:, num_components] 
            encodings_pre[key] = encodings[key].clone()
            # low dimension and change coordinate system
            encodings[key] = encodings[key] @ principal_components @ principal_components.T
            # encodings[key] /= torch.norm(encodings[key], p=2, dim=-1).unsqueeze(1)

            encodings_tsne.append(encodings[key])
            labels.append(np.array([key] * encodings[key].shape[0]))
            all_eigenvectors[key] = principal_components
            all_eigenvalues[key] = eig_values

        return all_eigenvectors, all_eigenvalues, encodings, encodings_pre, encodings_tsne, labels
    

def run(gpu, ngpus_per_node, args):

    # load pretrained model and tokenizer
    model = get_model(args, ngpus_per_node, gpu)
    tokenizer = OFATokenizer.from_pretrained(args.ckpt_dir)
    args.gpu = gpu

    if args.model == 'ofa':
        tokenizer = OFATokenizer.from_pretrained(args.ckpt_dir)
        model.decoder.output_projection = nn.Identity()
        output_projection = model.decoder.output_projection.cpu()
    elif args.model == 'clip':
        output_projection = model.output_projection

    args.gpu = gpu

    print(model)

    inputs = tokenizer([args.txt], return_tensors="pt").input_ids.cuda(args.gpu)
    inputs = inputs.repeat(args.batch_size, 1)
    decoder_input = torch.tensor([tokenizer.bos_token_id]).repeat(args.batch_size, 1).cuda(args.gpu)
    
    if not os.path.exists(os.path.join(args.output_dir, f'all_aug_features_{args.txt}.pt')):
        #  Dataset
        dataloader = get_dataloader(args)

        subspace_model = Model(model, dataloader, args.gpu, inputs, decoder_input)
        encodings = subspace_model.encode(dataloader, model, args)
        torch.save(encodings, os.path.join(args.output_dir, f'all_aug_features_{args.txt}.pt'))
    else:
        subspace_model = Model(model, None, args.gpu, inputs, decoder_input)
        encodings = torch.load(os.path.join(args.output_dir, f'all_aug_features_{args.txt}.pt'))

    all_eigenvectors, all_eigenvalues, encodings, encodings_pre, encodings_tsne, labels = subspace_model.pca(encodings,
                                                            num_principal_components=args.num_components)
    
    # Learn a orthogonal projection matrix from encodings of augmented images to the original image
    all_augs = list(encodings.keys())
    all_augs.remove('original')
    print("All augs for evaluation", all_augs)

    # Split data into train and validation
    train_data, val_data = {}, {}
    split = int(encodings['original'].shape[0] * 0.8)
    train_data['original'] = encodings_pre['original'][:split]
    val_data['original'] = encodings_pre['original'][split:]
    for aug in all_augs:
        train_data[aug] = encodings_pre[aug][:split]
        val_data[aug] = encodings_pre[aug][split:]

    # Calculate intrinsic means and logarithmic mappings of the encodings
    intrinsic_means = {}
    log_encodings_train,  log_encodings_val = {}, {}
    for aug in all_augs:
        intrinsic_mean = calculate_intrinstic_mean(torch.cat([train_data[aug], train_data['original']], 0))
        intrinsic_means[aug] = intrinsic_mean
        log_encodings_train[aug] = logarithmic_map(intrinsic_mean, train_data[aug])
        log_encodings_val[aug] = logarithmic_map(intrinsic_mean, val_data[aug])

    if args.train:
        for i, aug in enumerate(all_augs):

            # get original and augmented features to tangent space
            intrinsic_mean = intrinsic_means[aug]
            original = logarithmic_map(intrinsic_mean, train_data['original']).detach().clone().cuda(args.gpu)
            aug_image = log_encodings_train[aug].detach().clone().cuda(args.gpu)
            
            # get closed form solution to minimize the loss between original and projected features
            closed_form_sol = torch.linalg.lstsq(aug_image, original)
            proj_out = aug_image @ closed_form_sol.solution
            loss =  torch.norm(original - proj_out, dim=1).mean()

            # project back onto sphere
            proj_out_sphere = exponential_map(intrinsic_mean.cpu(), proj_out.cpu())
            distance_on_sphere = torch.norm(train_data['original'] - proj_out_sphere, dim=1).mean()
            print(f"Loss {loss.item()}", "Distance on sphere", distance_on_sphere.item())
            # save projection matrix
            torch.save(closed_form_sol.solution, os.path.join(args.output_dir, 'projection_matrics', f'{aug}_projection.pt'))

    else:
        projection_matrices = []
        for i, aug in enumerate(all_augs):
            print("Loading projection matrix", os.path.join(args.output_dir, 'projection_matrics', f'{aug}_projection.pt'))
            try:
                proj_mat = torch.load(os.path.join(args.output_dir, 'projection_matrics', f'{aug}_projection.pt'))
            except:
                print(f"Projection matrix for {aug} not found")
                raise FileNotFoundError
                
            projection_matrices.append(proj_mat)

        results = pd.DataFrame(columns=['augmentation', 'distance_pre', 'distance_post', 'cosine_sim_pre', 'cosine_sim_post', 'acc_pre', 'acc_post'])
        true_labels = torch.load(os.path.join(args.output_dir, 'labels.pt'))
        true_labels = true_labels[split:]


        for i, aug in enumerate(all_augs):
            proj_mat = projection_matrices[i]

            # convert original features to tangent space
            original = val_data['original'].detach().clone().cpu()
            # get augmented features to tangent space
            aug_feats = log_encodings_val[aug].detach().clone().cpu()
            # get closed form solution from saved projection matrix
            closed_form_sol = proj_mat.cpu()
            proj_out = aug_feats @ closed_form_sol
            # project back onto sphere
            proj_out = exponential_map(intrinsic_means[aug].cpu(), proj_out.cpu())

            distance_pre = torch.norm(original - val_data[aug], dim=1).mean()
            distance_post = torch.norm(original - proj_out, dim=1).mean()
            cosine_sim = F.cosine_similarity(original, proj_out, dim=1).mean()
            cosine_sim_pre = F.cosine_similarity(original, val_data[aug], dim=1).mean()

            pre_proj_aug_pred = output_projection(val_data[aug])
            post_proj_aug_pred = output_projection(proj_out)
            pred_orig = output_projection(original)
            pre_proj_aug_pred = pre_proj_aug_pred.argmax(dim=-1)
            post_proj_aug_pred = post_proj_aug_pred.argmax(dim=-1)
            pred_orig = pred_orig.argmax(dim=-1)

            orig_acc = (pred_orig == true_labels).sum().item() / len(true_labels)
            pre_proj_aug_acc = (pre_proj_aug_pred == true_labels).sum().item() / len(true_labels)
            post_proj_aug_acc = (post_proj_aug_pred == true_labels).sum().item() / len(true_labels)            
            
            results = results._append({'augmentation': aug, 'distance_pre': distance_pre.item(),
                                       'distance_post': distance_post.item(), 'cosine_sim_pre': cosine_sim_pre.item(),
                                         'cosine_sim_post': cosine_sim.item(), 'acc_pre': pre_proj_aug_acc, 'acc_post': post_proj_aug_acc}, ignore_index=True)
        
        print(results)

        results.to_csv(os.path.join(args.output_dir, 'results_tangent_projection.csv'))


if __name__ == '__main__':
    #  Define args 
    parser = argparse.ArgumentParser(description='OFA demo')
    parser.add_argument('--ckpt_dir', type=str, default='../../OFA-tiny', help='path to the checkpoint directory')
    parser.add_argument('--eval_dataset', type=str, default='', help='Dataset for evaluation')
    parser.add_argument('--data_dir', type=str, default='../../image-net100/val', help='path to the data directory')
    parser.add_argument('--output-dir', type=str, default='results/', help='path to the data directory')
    parser.add_argument('--model', type=str, default='clip', help='Model')
    parser.add_argument('--gpu', type=int, default=None, help='gpu device id')
    parser.add_argument('--seed', type=int, default=42, help='seed for random number generator')
    parser.add_argument('--num-aug-groups', default=6, type=int, help='number of augmentation groups')
    parser.add_argument('--num-components', type=int, default=64, help='Weight between covariance and inverse covariance')
    parser.add_argument('--train', action="store_true", help='Whether to train the model')

    # Feature subspace args
    parser.add_argument('--rank', type=int, default=16, help='rank of subspace')
    args = parser.parse_args()
    args.low_rank = True

    #  seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    if args.dist:
        args.world_size = args.world_size * args.num_gpus
        mp.spawn(run, nprocs=args.num_gpus, args=(args.num_gpus, args))  
    else:
        run(args.gpu, 1, args)