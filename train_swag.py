import argparse
import warnings
import torch
import os
import sys
sys.path.append(os.getcwd())
import wandb
import random
import json
import re
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import logging
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import strftime, localtime
from datasets import load_dataset
from transformers import EsmModel, BertModel
from transformers import EsmTokenizer, EsmForMaskedLM, BertForMaskedLM, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from src.utils.data_utils import BatchSampler
from src.models.adapter import AdapterModel
from src.utils.metrics import MultilabelF1Max
from src.utils.loss_fn import MultiClassFocalLossWithAlpha
from src.data.get_esm3_structure_seq import VQVAE_SPECIAL_TOKENS
import numpy as np
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def mutate_protein(sequence, mutation_rate:float = 0.01):
    mutated_sequence = list(sequence)  # Convert to a list for mutation
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:  # Check if this position should be mutated
            # Replace with a random amino acid
            mutated_sequence[i] = random.choice(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
    return "".join(mutated_sequence)

def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset):
    labels = [e["label"] for e in train_dataset]
    min_label, max_label = min(labels), max(labels)
    for e in train_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in val_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    for e in test_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
    return train_dataset, val_dataset, test_dataset

"""
    implementation of SWAG
"""

import gpytorch
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

def swag_parameters(module, params, no_cov_mat=True):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            module.register_buffer(
                "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            )

        params.append((module, name))

class SWAG(torch.nn.Module):
    def __init__(
        self, base, base_args, no_cov_mat=True, max_num_models=0, var_clamp=1e-30, initial_seq_layer_norm:bool=False, self_attn:bool=False
    ):
        super(SWAG, self).__init__()

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp

        self.base_args = base_args
        self.initial_seq_layer_norm = initial_seq_layer_norm
        self.self_attn = self_attn
        self.base = base(self.base_args, self.initial_seq_layer_norm, self.self_attn)
        # self.base = base
        self.base.apply(
            lambda module: swag_parameters(
                module=module, params=self.params, no_cov_mat=self.no_cov_mat
            )
        )

    def forward(self, plm_model, batch):
        return self.base(plm_model, batch)

    def sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):
        if seed is not None:
            torch.manual_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov, fullrank)
        else:
            self.sample_blockwise(scale, cov, fullrank)

    def sample_blockwise(self, scale, cov, fullrank):
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name, w)

    def sample_fullrank(self, scale, cov, fullrank):
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample.cuda())

    def collect_model(self, base_model):
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (base_param.data - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)

    def load_state_dict(self, state_dict, strict=True):
        if not self.no_cov_mat:
            n_models = state_dict["n_models"].item()
            rank = min(n_models, self.max_num_models)
            for module, name in self.params:
                mean = module.__getattr__("%s_mean" % name)
                module.__setattr__(
                    "%s_cov_mat_sqrt" % name,
                    mean.new_empty((rank, mean.numel())).zero_(),
                )
        super(SWAG, self).load_state_dict(state_dict, strict)

    def export_numpy_params(self, export_cov_mat=False):
        mean_list = []
        sq_mean_list = []
        cov_mat_list = []

        for module, name in self.params:
            mean_list.append(module.__getattr__("%s_mean" % name).cpu().numpy().ravel())
            sq_mean_list.append(
                module.__getattr__("%s_sq_mean" % name).cpu().numpy().ravel()
            )
            if export_cov_mat:
                cov_mat_list.append(
                    module.__getattr__("%s_cov_mat_sqrt" % name).cpu().numpy().ravel()
                )
        mean = np.concatenate(mean_list)
        sq_mean = np.concatenate(sq_mean_list)
        var = sq_mean - np.square(mean)

        if export_cov_mat:
            return mean, var, cov_mat_list
        else:
            return mean, var

    def import_numpy_weights(self, w):
        k = 0
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            s = np.prod(mean.shape)
            module.__setattr__(name, mean.new_tensor(w[k : k + s].reshape(mean.shape)))
            k += s

    def generate_mean_var_covar(self):
        mean_list = []
        var_list = []
        cov_mat_root_list = []
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)
            cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

            mean_list.append(mean)
            var_list.append(sq_mean - mean ** 2.0)
            cov_mat_root_list.append(cov_mat_sqrt)
        return mean_list, var_list, cov_mat_root_list

    def compute_ll_for_block(self, vec, mean, var, cov_mat_root):
        vec = flatten(vec)
        mean = flatten(mean)
        var = flatten(var)

        # cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        # var_lt = DiagLazyTensor(var + 1e-6)
        # covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)
        # qdist = MultivariateNormal(mean, covar_lt)

        cov_root = cov_mat_root.t()  # [D, r]
        low_rank = cov_root @ cov_root.t()  # [D, D]
        cov = low_rank + torch.diag(var + 1e-6)
        qdist = MultivariateNormal(mean, cov)

        with gpytorch.settings.num_trace_samples(
            1
        ) and gpytorch.settings.max_cg_iterations(25):
            return qdist.log_prob(vec)

    def block_logdet(self, var, cov_mat_root):
        var = flatten(var)

        # cov_mat_lt = RootLazyTensor(cov_mat_root.t())
        # var_lt = DiagLazyTensor(var + 1e-6)
        # covar_lt = AddedDiagLazyTensor(var_lt, cov_mat_lt)

        cov_root = cov_mat_root.t()
        low_rank = cov_root @ cov_root.t()
        cov = low_rank + torch.diag(var + 1e-6)
        
        assert torch.linalg.slogdet(cov).sign > 0

        return torch.linalg.slogdet(cov).logabsdet #covar_lt.log_det()

    def block_logll(self, param_list, mean_list, var_list, cov_mat_root_list):
        full_logprob = 0
        for i, (param, mean, var, cov_mat_root) in enumerate(
            zip(param_list, mean_list, var_list, cov_mat_root_list)
        ):
            # print('Block: ', i)
            block_ll = self.compute_ll_for_block(param, mean, var, cov_mat_root)
            full_logprob += block_ll

        return full_logprob

    def full_logll(self, param_list, mean_list, var_list, cov_mat_root_list):
        cov_mat_root = torch.cat(cov_mat_root_list, dim=1)
        mean_vector = flatten(mean_list)
        var_vector = flatten(var_list)
        param_vector = flatten(param_list)
        return self.compute_ll_for_block(
            param_vector, mean_vector, var_vector, cov_mat_root
        )

    def compute_logdet(self, block=False):
        _, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if block:
            full_logdet = 0
            for (var, cov_mat_root) in zip(var_list, covar_mat_root_list):
                block_logdet = self.block_logdet(var, cov_mat_root)
                full_logdet += block_logdet
        else:
            var_vector = flatten(var_list)
            cov_mat_root = torch.cat(covar_mat_root_list, dim=1)
            full_logdet = self.block_logdet(var_vector, cov_mat_root)

        return full_logdet

    def diag_logll(self, param_list, mean_list, var_list):
        logprob = 0.0
        for param, mean, scale in zip(param_list, mean_list, var_list):
            logprob += Normal(mean, scale).log_prob(param).sum()
        return logprob

    def compute_logprob(self, vec=None, block=False, diag=False):
        mean_list, var_list, covar_mat_root_list = self.generate_mean_var_covar()

        if vec is None:
            param_list = [getattr(param, name) for param, name in self.params]
        else:
            param_list = unflatten_like(vec, mean_list)

        if diag:
            return self.diag_logll(param_list, mean_list, var_list)
        elif block is True:
            return self.block_logll(
                param_list, mean_list, var_list, covar_mat_root_list
            )
        else:
            return self.full_logll(param_list, mean_list, var_list, covar_mat_root_list)

def train(args, model, swag_model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, 
          loss_fn, optimizer, device):
    # best_val_loss, best_val_metric_score = float("inf"), -float("inf")
    print("---------- Initial Evaluation ----------")
    model.eval()
    swag_model.eval()
    with torch.no_grad():
        best_val_loss, val_metric_dict = eval_loop(args, swag_model, plm_model, metrics_dict, val_loader, loss_fn, device)
        best_val_metric_score = val_metric_dict[args.monitor]

    val_loss_list, val_metric_list = [], []
    path = os.path.join(args.ckpt_dir, args.model_name)
    global_steps = 0
    for epoch in range(args.max_train_epochs):
        print(f"---------- Epoch {epoch} ----------")
        # train
        model.train()
        swag_model.train()
        epoch_train_loss = 0
        epoch_iterator = tqdm(train_loader)
        for batch in epoch_iterator:
            with accelerator.accumulate(model):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                label = batch["label"]
                logits = model(plm_model, batch)
                if args.problem_type == 'regression' and args.num_labels == 1:
                    loss = loss_fn(logits.squeeze(), label.squeeze())
                elif args.problem_type == 'multi_label_classification':
                    loss = loss_fn(logits, label.float())
                else:
                    loss = loss_fn(logits, label)
                epoch_train_loss += loss.item() * len(label)                
                global_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                swag_model.collect_model(model)

                epoch_iterator.set_postfix(train_loss=loss.item())
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_steps)
                    
        train_loss = epoch_train_loss / len(train_loader.dataset)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f}')
        
        # eval every epoch
        model.eval()
        swag_model.eval()
        with torch.no_grad():
            val_loss, val_metric_dict = eval_loop(args, swag_model, plm_model, metrics_dict, val_loader, loss_fn, device)
            val_metric_score = val_metric_dict[args.monitor]
            val_metric_list.append(val_metric_score)
            val_loss_list.append(val_loss)
            
            if args.wandb:
                val_log = {"valid/loss": val_loss}
                for metric_name, metric_score in val_metric_dict.items():
                    val_log[f"valid/{metric_name}"] = metric_score
                wandb.log(val_log)
            print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} {args.monitor}: {val_metric_score:.4f}')
    
        torch.save(swag_model.state_dict(), path)
        
    #     # early stopping
    #     if args.monitor == "loss":
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             torch.save(swag_model.state_dict(), path)
    #             print(f'>>> BEST at epcoh {epoch}, loss: {best_val_loss:.4f}')
    #             for metric_name, metric_score in val_metric_dict.items():
    #                 print(f'>>> {metric_name}: {metric_score:.4f}')
    #             print(f'>>> Save model to {path}')
            
    #         if len(val_loss_list) - val_loss_list.index(min(val_loss_list)) > args.patience:
    #             print(f'>>> Early stopping at epoch {epoch}')
    #             break
    #     else:
    #         if val_metric_score > best_val_metric_score:
    #             best_val_metric_score = val_metric_score
    #             torch.save(swag_model.state_dict(), path)
    #             print(f'>>> BEST at epcoh {epoch}, loss: {val_loss:.4f} {args.monitor}: {best_val_metric_score:.4f}')
    #             for metric_name, metric_score in val_metric_dict.items():
    #                 print(f'>>> {metric_name}: {metric_score:.4f}')
    #             print(f'>>> Save model to {path}')
            
    #         if len(val_metric_list) - val_metric_list.index(max(val_metric_list)) > args.patience:
    #             print(f'>>> Early stopping at epoch {epoch}')
    #             break
    
    # print(f"TESTING: loading from {path}")
    # swag_model.load_state_dict(torch.load(path))
    # swag_model.eval()
    # with torch.no_grad():
    #     test_loss, test_metric_dict = eval_loop(args, swag_model, plm_model, metrics_dict, test_loader, loss_fn, device)
    #     test_metric_score = test_metric_dict[args.monitor]
        
    #     if args.wandb:
    #         test_log = {"test/loss": test_loss}
    #         for metric_name, metric_score in test_metric_dict.items():
    #             test_log[f"test/{metric_name}"] = metric_score
    #         wandb.log(test_log)
    #     print(f'EPOCH {epoch} TEST loss: {test_loss:.4f} {args.monitor}: {test_metric_score:.4f}')
    #     for metric_name, metric_score in test_metric_dict.items():
    #         print(f'>>> {metric_name}: {metric_score:.4f}')


def eval_loop(args, swag_model, plm_model, metrics_dict, dataloader, loss_fn, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        swag_model.sample(scale=1, cov=True)
        logits = swag_model(plm_model, batch)
        for metric_name, metric in metrics_dict.items():
            if args.problem_type == 'regression' and args.num_labels == 1:
                loss = loss_fn(logits.squeeze(), label.squeeze())
                metric(logits.squeeze(), label.squeeze())
            elif args.problem_type == 'multi_label_classification':
                loss = loss_fn(logits, label.float())
                metric(logits, label)
            else:
                loss = loss_fn(logits, label)
                metric(torch.argmax(logits, 1), label)
        total_loss += loss.item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    metrics_result_dict = {}
    epoch_loss = total_loss / len(dataloader.dataset)
    for metric_name, metric in metrics_dict.items():
        metrics_result_dict[metric_name] = metric.compute().item()
        metric.reset()
    return epoch_loss, metrics_result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--pooling_method', type=str, default='attention1d', help='pooling method')
    parser.add_argument('--return_attentions', action='store_true', help='return attentions')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--dataset_config', type=str, default=None, help='config of dataset')
    parser.add_argument('--num_labels', type=int, default=None, help='number of labels')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--pdb_type', type=str, default=None, choices=[None, 'AlphaFold2', 'ESMFold'], help='pdb type')
    parser.add_argument('--train_file', type=str, default=None, help='train file')
    parser.add_argument('--valid_file', type=str, default=None, help='val file')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    
    # train model
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--max_batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--max_train_epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--monitor', type=str, default=None, help='monitor metric')
    parser.add_argument('--structure_seqs', type=str, default=None, help='structure token')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'], help='loss function')
    parser.add_argument('--load_pretrained', action='store_true', help='Do retraining of a pretrained model, filepath must be same')
    parser.add_argument('--mutation_train', action='store_true', help='mutate training set sequences')
    parser.add_argument('--mutation_rate', type=float, default=0.005, help='mutation rate of each sequence in the training set')
    parser.add_argument('--mutation_prob', type=float, default=0.5, help='mutation probability of sequences in training set')

    # save model
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--ckpt_root', default="ckpt", help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    
    # wandb log
    parser.add_argument('--wandb', action='store_true', help='use wandb to log')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_project', type=str, default='SES-Adapter')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.structure_seqs is not None:
        args.structure_seqs = args.structure_seqs.split(',')
        
    dataset_config = json.loads(open(args.dataset_config).read())
    if args.dataset is None:
        args.dataset = dataset_config['dataset']
    if args.pdb_type is None:
        args.pdb_type = dataset_config['pdb_type']
    if args.train_file is None:
        if dataset_config.get('train_file'):
            args.train_file = dataset_config['train_file']
    if args.valid_file is None:
        if dataset_config.get('valid_file'):
            args.valid_file = dataset_config['valid_file']
    if args.test_file is None:
        if dataset_config.get('test_file'):
            args.test_file = dataset_config['test_file']
    if args.num_labels is None:
        args.num_labels = dataset_config['num_labels']
    if args.problem_type is None:
        args.problem_type = dataset_config['problem_type']
    if args.monitor is None:
        args.monitor = dataset_config['monitor']
    
    metrics_dict = {}
    if args.metrics is None:
        args.metrics = dataset_config['metrics']
        if args.metrics != 'None':
            args.metrics = args.metrics.split(',')
            for m in args.metrics:
                if m == 'accuracy':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryAccuracy()
                    else:
                        metrics_dict[m] = Accuracy(task="multiclass", num_classes=args.num_labels)
                elif m == 'recall':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryRecall()
                    else:
                        metrics_dict[m] = Recall(task="multiclass", num_classes=args.num_labels)
                elif m == 'precision':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryPrecision()
                    else:
                        metrics_dict[m] = Precision(task="multiclass", num_classes=args.num_labels)
                elif m == 'f1':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryF1Score()
                    else:
                        metrics_dict[m] = F1Score(task="multiclass", num_classes=args.num_labels)
                elif m == 'mcc':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryMatthewsCorrCoef()
                    else:
                        metrics_dict[m] = MatthewsCorrCoef(task="multiclass", num_classes=args.num_labels)
                elif m == 'auc':
                    if args.num_labels == 2:
                        metrics_dict[m] = BinaryAUROC()
                    else:
                        metrics_dict[m] = AUROC(task="multiclass", num_classes=args.num_labels)
                elif m == 'f1_max':
                    metrics_dict[m] = MultilabelF1Max(num_labels=args.num_labels)
                elif m == 'spearman_corr':
                    metrics_dict[m] = SpearmanCorrCoef()
                else:
                    raise ValueError(f"Invalid metric: {m}")
            for metric_name, metric in metrics_dict.items():
                metric.to(device)            
        
    
    # create checkpoint directory
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    else:
        args.ckpt_dir = os.path.join(args.ckpt_root, args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # init wandb
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"Adapter-{args.dataset}"
        if args.model_name is None:
            args.model_name = f"{args.wandb_run_name}.pt"
        
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, 
            entity=args.wandb_entity, config=vars(args)
        )
    
    # build tokenizer and protein language model
    if "esm2" in args.plm_model:
        print(f"Loading ESM model: {args.plm_model}")
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model, output_hidden_states=True).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        print(f"Loading BERT model: {args.plm_model}")
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "ProstT5" in args.plm_model:
        print(f"Loading ProstT5 model: {args.plm_model}")
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    elif "ankh" in args.plm_model:
        print(f"Loading Ankh model: {args.plm_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    elif "esmc" in args.plm_model:
        print(f"Loading ESMC model: {args.plm_model}")
        tokenizer = get_esmc_model_tokenizers()
        plm_model = ESMC.from_pretrained(args.plm_model, device=torch.device(device)).to(torch.float32)
        args.hidden_size = plm_model.embed.embedding_dim

    if args.structure_seqs is not None:
        if isinstance(plm_model, ESMC):
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = plm_model.config.vocab_size
        if 'esm3_structure_seq' in args.structure_seqs: 
            # args.vocab_size = max(plm_model.config.vocab_size, 4100)
            args.vocab_size = max(vocab_size, 4100)
        else:
            # args.vocab_size = plm_model.config.vocab_size
            args.vocab_size = vocab_size
    else:
        args.structure_seqs = []
    
    # add 8 dimension for e-descriptor and z-descriptor
    # if 'ez_descriptor' in args.structure_seqs:
    #     args.hidden_size += 8
    # elif 'aac' in args.structure_seqs:
    #     args.hidden_size += 64

    # load adapter model
    model = AdapterModel(args, initial_seq_layer_norm="esmc" in args.plm_model)
    #######
    # model_path = f"./ckpt/Virus.pt"
    # model.load_state_dict(torch.load(model_path, weights_only=False), strict=True) 
    pretrained_path = os.path.join(args.ckpt_dir, args.model_name)
    if os.path.exists(pretrained_path) and args.load_pretrained:
        model.load_state_dict(torch.load(pretrained_path, weights_only=False), strict=True) 
        print("LOADING PRETRAINED MODEL")
    #######
    model.to(device)

    swag_model = SWAG(AdapterModel, 
                  args, 
                  initial_seq_layer_norm="esmc" in args.plm_model,
                  self_attn=False,
                  no_cov_mat=False, 
                  max_num_models=64).to(device)

    print("---------- Architecture of the model ----------")
    print(model)

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
    
    def process_data_line(data):
        if args.problem_type == 'multi_label_classification':
            label_list = data['label'].split(',')
            data['label'] = [int(l) for l in label_list]
            binary_list = [0] * args.num_labels
            for index in data['label']:
                binary_list[index] = 1
            data['label'] = binary_list
        
        if args.max_seq_len is not None:
            data["aa_seq"] = data["aa_seq"][:args.max_seq_len]
            for seq in args.structure_seqs:
                data[seq] = data[seq][:args.max_seq_len]
            token_num = min(len(data["aa_seq"]), args.max_seq_len)
        else:
            token_num = len(data["aa_seq"])
        return data, token_num
    
    # process dataset from json file
    def process_dataset_from_json(file):
        dataset, token_nums = [], []
        for l in open(file):
            data = json.loads(l)
            data, token_num = process_data_line(data)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums


    if args.train_file is not None and args.train_file[-4:] == 'json':
        train_dataset, train_token_num = process_dataset_from_json(args.train_file)
        val_dataset, val_token_num = process_dataset_from_json(args.valid_file)
        test_dataset, test_token_num = process_dataset_from_json(args.test_file)
    
    # process dataset from list
    def process_dataset_from_list(data_list):
        dataset, token_nums = [], []
        for l in data_list:
            data, token_num = process_data_line(l)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums
    
    if args.train_file == None:
        train_dataset, train_token_num = process_dataset_from_list(load_dataset(args.dataset)['train'])
        val_dataset, val_token_num = process_dataset_from_list(load_dataset(args.dataset)['validation'])
        test_dataset, test_token_num = process_dataset_from_list(load_dataset(args.dataset)['test'])
    
    if dataset_config['normalize'] == 'min_max':
        train_dataset, val_dataset, test_dataset = min_max_normalize_dataset(train_dataset, val_dataset, test_dataset)
    
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(val_dataset))
    print(">>> testset: ", len(test_dataset))
    print("---------- Smple 3 data point from trainset ----------")
    
    for i in random.sample(range(len(train_dataset)), 3):
        print(">>> ", train_dataset[i])

    # def e_descriptor_embedding(aa_input_ids):
    #     aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
    #     e1 = {'A': 0.008, 'R': 0.171, 'N': 0.255, 'D': 0.303, 'C': -0.132, 'Q': 0.149, 'E': 0.221, 'G': 0.218,
    #         'H': 0.023, 'I': -0.353, 'L': -0.267, 'K': 0.243, 'M': -0.239, 'F': -0.329, 'P': 0.173, 'S': 0.199,
    #         'T': 0.068, 'W': -0.296, 'Y': -0.141, 'V': -0.274}
    #     e2 = {'A': 0.134, 'R': -0.361, 'N': 0.038, 'D': -0.057, 'C': 0.174, 'Q': -0.184, 'E': -0.28, 'G': 0.562,
    #         'H': -0.177, 'I': 0.071, 'L': 0.018, 'K': -0.339, 'M': -0.141, 'F': -0.023, 'P': 0.286, 'S': 0.238,
    #         'T': 0.147, 'W': -0.186, 'Y': -0.057, 'V': 0.136}
    #     e3 = {'A': -0.475, 'R': 0.107, 'N': 0.117, 'D': -0.014, 'C': 0.07, 'Q': -0.03, 'E': -0.315, 'G': -0.024,
    #         'H': 0.041, 'I': -0.088, 'L': -0.265, 'K': -0.044, 'M': -0.155, 'F': 0.072, 'P': 0.407, 'S': -0.015,
    #         'T': -0.015, 'W': 0.389, 'Y': 0.425, 'V': -0.187}
    #     e4 = {'A': -0.039, 'R': -0.258, 'N': 0.118, 'D': 0.225, 'C': 0.565, 'Q': 0.035, 'E': 0.157, 'G': 0.018,
    #         'H': 0.28, 'I': -0.195, 'L': -0.274, 'K': -0.325, 'M': 0.321, 'F': -0.002, 'P': -0.215, 'S': -0.068,
    #         'T': -0.132, 'W': 0.083, 'Y': -0.096, 'V': -0.196}
    #     e5 = {'A': 0.181, 'R': -0.364, 'N': -0.055, 'D': 0.156, 'C': -0.374, 'Q': -0.112, 'E': 0.303, 'G': 0.106,
    #         'H': -0.021, 'I': -0.107, 'L': 0.206, 'K': -0.027, 'M': 0.077, 'F': 0.208, 'P': 0.384, 'S': -0.196,
    #         'T': -0.274, 'W': 0.297, 'Y': -0.091, 'V': -0.299}
    #     # Build descriptor tensors
    #     descriptor_dicts = [e1, e2, e3, e4, e5]
    #     descriptors = {}
    #     for aa in e1.keys():
    #         descriptors[aa] = [d[aa] for d in descriptor_dicts]   #每个氨基酸对应一个5维的描述符
    #     e_embeds = []
    #     for seq in aa_seqs:
    #         seq_embeds = [descriptors.get(aa, [0.0]*5) for aa in seq]
    #         e_embeds.append(seq_embeds)
    #     e_embeds = torch.tensor(e_embeds).float()
    #     return e_embeds

    def e_descriptor_embedding(aa_seqs):
        # aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
        e1 = {'A': 0.008, 'R': 0.171, 'N': 0.255, 'D': 0.303, 'C': -0.132, 'Q': 0.149, 'E': 0.221, 'G': 0.218,
            'H': 0.023, 'I': -0.353, 'L': -0.267, 'K': 0.243, 'M': -0.239, 'F': -0.329, 'P': 0.173, 'S': 0.199,
            'T': 0.068, 'W': -0.296, 'Y': -0.141, 'V': -0.274}
        e2 = {'A': 0.134, 'R': -0.361, 'N': 0.038, 'D': -0.057, 'C': 0.174, 'Q': -0.184, 'E': -0.28, 'G': 0.562,
            'H': -0.177, 'I': 0.071, 'L': 0.018, 'K': -0.339, 'M': -0.141, 'F': -0.023, 'P': 0.286, 'S': 0.238,
            'T': 0.147, 'W': -0.186, 'Y': -0.057, 'V': 0.136}
        e3 = {'A': -0.475, 'R': 0.107, 'N': 0.117, 'D': -0.014, 'C': 0.07, 'Q': -0.03, 'E': -0.315, 'G': -0.024,
            'H': 0.041, 'I': -0.088, 'L': -0.265, 'K': -0.044, 'M': -0.155, 'F': 0.072, 'P': 0.407, 'S': -0.015,
            'T': -0.015, 'W': 0.389, 'Y': 0.425, 'V': -0.187}
        e4 = {'A': -0.039, 'R': -0.258, 'N': 0.118, 'D': 0.225, 'C': 0.565, 'Q': 0.035, 'E': 0.157, 'G': 0.018,
            'H': 0.28, 'I': -0.195, 'L': -0.274, 'K': -0.325, 'M': 0.321, 'F': -0.002, 'P': -0.215, 'S': -0.068,
            'T': -0.132, 'W': 0.083, 'Y': -0.096, 'V': -0.196}
        e5 = {'A': 0.181, 'R': -0.364, 'N': -0.055, 'D': 0.156, 'C': -0.374, 'Q': -0.112, 'E': 0.303, 'G': 0.106,
            'H': -0.021, 'I': -0.107, 'L': 0.206, 'K': -0.027, 'M': 0.077, 'F': 0.208, 'P': 0.384, 'S': -0.196,
            'T': -0.274, 'W': 0.297, 'Y': -0.091, 'V': -0.299}
        # Build descriptor tensors
        descriptor_dicts = [e1, e2, e3, e4, e5]
        descriptors = {}
        for aa in e1.keys():
            descriptors[aa] = [d[aa] for d in descriptor_dicts]   #每个氨基酸对应一个5维的描述符
        e_embeds = []
        for seq in aa_seqs:
            seq_embeds = [descriptors.get(aa, [0.0]*5) for aa in seq]
            e_embeds.append(seq_embeds[0])
        # e_embeds = torch.tensor(e_embeds).float()
        return e_embeds

    # def z_descriptor_embedding(aa_input_ids):
    #     aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
    #     z1 = {'A': 0.07, 'R': 2.88, 'N': 3.22, 'D': 3.64, 'C': 0.71, 'Q': 2.18, 'E': 3.08, 'G': 2.23, 'H': 2.41,
    #         'I': -4.44, 'L': -4.19, 'K': 2.84, 'M': -2.49, 'F': -4.92, 'P': -1.22, 'S': 1.96, 'T': 0.92, 'W': -4.75,
    #         'Y': -1.39, 'V': -2.69}
    #     z2 = {'A': -1.73, 'R': 2.52, 'N': 1.45, 'D': 1.13, 'C': -0.97, 'Q': 0.53, 'E': 0.39, 'G': -5.36, 'H': 1.74,
    #         'I': -1.68, 'L': -1.03, 'K': 1.41, 'M': -0.27, 'F': 1.30, 'P': 0.88, 'S': -1.63, 'T': -2.09, 'W': 3.65,
    #         'Y': 2.32, 'V': -2.53}
    #     z3 = {'A': 0.09, 'R': -3.44, 'N': 0.84, 'D': 2.36, 'C': 4.13, 'Q': -1.14, 'E': -0.07, 'G': 0.30, 'H': 1.11,
    #         'I': -1.03, 'L': -0.98, 'K': -3.14, 'M': -0.41, 'F': 0.45, 'P': 2.23, 'S': 0.57, 'T': -1.40, 'W': 0.85,
    #         'Y': 0.01, 'V': -1.29}
    #     # Build descriptor tensors
    #     descriptor_dicts = [z1, z2, z3]
    #     descriptors = {}
    #     for aa in z1.keys():
    #         descriptors[aa] = [d[aa] for d in descriptor_dicts]
    #     z_embeds = []
    #     for seq in aa_seqs:
    #         seq_embeds = [descriptors.get(aa, [0.0]*3) for aa in seq]
    #         z_embeds.append(seq_embeds)
    #     z_embeds = torch.tensor(z_embeds).float()
    #     return z_embeds
    
    def z_descriptor_embedding(aa_seqs):
        # aa_seqs = [tokenizer.convert_ids_to_tokens(aa_input_ids[i]) for i in range(len(aa_input_ids))]
        z1 = {'A': 0.07, 'R': 2.88, 'N': 3.22, 'D': 3.64, 'C': 0.71, 'Q': 2.18, 'E': 3.08, 'G': 2.23, 'H': 2.41,
            'I': -4.44, 'L': -4.19, 'K': 2.84, 'M': -2.49, 'F': -4.92, 'P': -1.22, 'S': 1.96, 'T': 0.92, 'W': -4.75,
            'Y': -1.39, 'V': -2.69}
        z2 = {'A': -1.73, 'R': 2.52, 'N': 1.45, 'D': 1.13, 'C': -0.97, 'Q': 0.53, 'E': 0.39, 'G': -5.36, 'H': 1.74,
            'I': -1.68, 'L': -1.03, 'K': 1.41, 'M': -0.27, 'F': 1.30, 'P': 0.88, 'S': -1.63, 'T': -2.09, 'W': 3.65,
            'Y': 2.32, 'V': -2.53}
        z3 = {'A': 0.09, 'R': -3.44, 'N': 0.84, 'D': 2.36, 'C': 4.13, 'Q': -1.14, 'E': -0.07, 'G': 0.30, 'H': 1.11,
            'I': -1.03, 'L': -0.98, 'K': -3.14, 'M': -0.41, 'F': 0.45, 'P': 2.23, 'S': 0.57, 'T': -1.40, 'W': 0.85,
            'Y': 0.01, 'V': -1.29}
        # Build descriptor tensors
        descriptor_dicts = [z1, z2, z3]
        descriptors = {}
        for aa in z1.keys():
            descriptors[aa] = [d[aa] for d in descriptor_dicts]
        z_embeds = []
        for seq in aa_seqs:
            seq_embeds = [descriptors.get(aa, [0.0]*3) for aa in seq]
            z_embeds.append(seq_embeds[0])
        # z_embeds = torch.tensor(z_embeds).float()
        return z_embeds

    def aac_embedding(aa_input_ids):
        e_embeds = e_descriptor_embedding(aa_input_ids)  # Shape: (batch_size, seq_len, 5)
        z_embeds = z_descriptor_embedding(aa_input_ids)  # Shape: (batch_size, seq_len, 3)
        ez_embeds = torch.cat([e_embeds, z_embeds], dim=-1)  # Shape: (batch_size, seq_len, 8)
        batch_size, seq_len, k = ez_embeds.shape  # k = 8

        # Initialize a tensor to hold the autocovariance matrices
        covariances = []
        for l in range(seq_len):
            seq_len_l = seq_len - l
            x1 = ez_embeds[:, :seq_len_l, :]  # Shape: (batch_size, seq_len_l, k)
            x2 = ez_embeds[:, l:, :]          # Shape: (batch_size, seq_len_l, k)
            # Compute covariance matrices without explicit loops over features
            cov_l = torch.matmul(x1.transpose(1, 2), x2) / seq_len_l  # Shape: (batch_size, k, k)
            covariances.append(cov_l)

        # Stack and reshape the covariances
        covariances = torch.stack(covariances, dim=1)  # Shape: (batch_size, seq_len, k, k)
        covariances = covariances.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, k * k)
        return covariances.float()

    def collate_fn(
        examples,
        train:bool = False,
        mutation_rate:float = 0.01,
        mutation_prob:float = 0.5,
    ):
        aa_seqs, labels = [], []
        e_descriptor, z_descriptor = [], []

        if 'foldseek_seq' in args.structure_seqs:
            foldseek_seqs = []
        if 'esm3_structure_seq' in args.structure_seqs:
            esm3_structure_seqs = []
        
            
        for e in examples:
            aa_seq = e["aa_seq"]
            if random.random() < mutation_prob and train:
                print("MUTATING SEQUENCE")
                aa_seq = mutate_protein(aa_seq, mutation_rate=mutation_rate)
                if 'ez_descriptor' in args.structure_seqs:
                    e_descriptor.append(torch.tensor(e_descriptor_embedding(aa_seq)))
                    z_descriptor.append(torch.tensor(z_descriptor_embedding(aa_seq)))
            else:
                if 'ez_descriptor' in args.structure_seqs:
                    e_descriptor.append(torch.tensor(e["e_descriptor"]))
                    z_descriptor.append(torch.tensor(e["z_descriptor"]))

            if 'foldseek_seq' in args.structure_seqs:
                foldseek_seq = e["foldseek_seq"]
            
            if 'prot_bert' in args.plm_model or "ProstT5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
                if 'foldseek_seq' in args.structure_seqs:
                    foldseek_seq = " ".join(list(foldseek_seq))
            elif 'ankh' in args.plm_model or "esmc" in args.plm_model:
                aa_seq = list(aa_seq)
                if 'foldseek_seq' in args.structure_seqs:
                    foldseek_seq = list(foldseek_seq)
            
            aa_seqs.append(aa_seq)
            if 'foldseek_seq' in args.structure_seqs:
                foldseek_seqs.append(foldseek_seq)
            if 'esm3_structure_seq' in args.structure_seqs:
                esm3_structure_seq = [VQVAE_SPECIAL_TOKENS["BOS"]] + e["esm3_structure_seq"] + [VQVAE_SPECIAL_TOKENS["EOS"]]
                esm3_structure_seqs.append(torch.tensor(esm3_structure_seq))
            
            labels.append(e["label"])
            # if 'ez_descriptor' in args.structure_seqs:
            #     e_descriptor.append(torch.tensor(e["e_descriptor"]))
            #     z_descriptor.append(torch.tensor(e["z_descriptor"]))
        
        if 'ankh' in args.plm_model or 'esmc' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            if 'foldseek_seq' in args.structure_seqs:
                foldseek_input_ids = tokenizer.batch_encode_plus(foldseek_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
        else:
            aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
            if 'foldseek_seq' in args.structure_seqs:
                foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        if args.problem_type == 'regression':
            labels = torch.as_tensor(labels, dtype=torch.float)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)

        data_dict = {"aa_input_ids": aa_input_ids, "attention_mask": attention_mask, "label": labels}

        if 'ez_descriptor' in args.structure_seqs:
            # add e-descriptor and z-descriptor embedding
            # pad e_descriptor_embeds to the same length as aa_input_ids
            e_descriptor_embeds = torch.stack([torch.cat([e_descriptor[i], torch.zeros(len(aa_input_ids[i]) - len(e_descriptor[i]), 5)], dim=0) for i in range(len(e_descriptor))])
            z_descriptor_embeds = torch.stack([torch.cat([z_descriptor[i], torch.zeros(len(aa_input_ids[i]) - len(z_descriptor[i]), 3)], dim=0) for i in range(len(z_descriptor))])
            data_dict["e_descriptor_embeds"] = e_descriptor_embeds
            data_dict["z_descriptor_embeds"] = z_descriptor_embeds

        if 'aac' in args.structure_seqs:
            # add aac embedding
            aac_embeds = aac_embedding(aa_input_ids)
            data_dict["aac_embeds"] = aac_embeds

        if 'foldseek_seq' in args.structure_seqs:
            data_dict["foldseek_input_ids"] = foldseek_input_ids
        if 'esm3_structure_seq' in args.structure_seqs:
            # pad the list of esm3_structure_seq and convert to tensor
            esm3_structure_input_ids = torch.nn.utils.rnn.pad_sequence(
                esm3_structure_seqs, batch_first=True, padding_value=VQVAE_SPECIAL_TOKENS["PAD"]
                )
            if 'ankh' in args.plm_model or "ProstT5" in args.plm_model:
                esm3_structure_input_ids = esm3_structure_input_ids[:,:-1]
            data_dict["esm3_structure_input_ids"] = esm3_structure_input_ids
        return data_dict
        
    # metrics, optimizer, dataloader
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.problem_type == "single_label_classification":
        if args.loss_fn == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss_fn == "focal_loss":
            train_labels = [e["label"] for e in train_dataset]
            alpha = [len(train_labels) / train_labels.count(i) for i in range(args.num_labels)]
            print(">>> alpha: ", alpha)
            loss_fn = MultiClassFocalLossWithAlpha(num_classes=args.num_labels, alpha=alpha, device=device)
    elif args.problem_type == "regression":
        loss_fn = nn.MSELoss()
    elif args.problem_type == "multi_label_classification":
        loss_fn = nn.BCEWithLogitsLoss()
    
    if args.mutation_train:
        train_collate_fn = lambda batch:collate_fn(batch, train=True, mutation_rate=args.mutation_rate, mutation_prob=args.mutation_prob)
    else:
        train_collate_fn = collate_fn

    train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers, 
        collate_fn=train_collate_fn,
        batch_sampler=BatchSampler(train_token_num, args.max_batch_token)
        )
    val_loader = DataLoader(
        val_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, args.max_batch_token, False)
        )
    test_loader = DataLoader(
        test_dataset, num_workers=args.num_workers, collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, args.max_batch_token, False)
        )
    
    model, swag_model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, swag_model, optimizer, train_loader, val_loader, test_loader
    )
    
    print("---------- Start Training ----------")
    train(args, model, swag_model, plm_model, accelerator, metrics_dict, train_loader, val_loader, test_loader, loss_fn, optimizer, device)
    
    if args.wandb:
        wandb.finish() 