import argparse 
import numpy as np
import torch 
import json
import pandas as pd
import re
from transformers import EsmModel, BertModel
from transformers import EsmTokenizer, EsmForMaskedLM, BertForMaskedLM, BertTokenizer
from transformers import BertTokenizer, BertModel, T5EncoderModel, AutoTokenizer, T5Tokenizer
from transformers import logging
from torch.utils.data import DataLoader
from src.utils.data_utils import BatchSampler
from src.models.adapter import AdapterModel
from src.data.get_esm3_structure_seq import VQVAE_SPECIAL_TOKENS
import os
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from tqdm import tqdm

model_params = {
    'hidden_size': None,  # 将由PLM模型自动设置
    'num_attention_heads': 8,
    'attention_probs_dropout_prob': 0,
    'num_labels': 2,
    'pooling_method': 'attention1d',
    'pooling_dropout': 0.1,
    'return_attentions': False,
    'structure_seqs': ['ez_descriptor', 'foldseek_seq', 'esm3_structure_seq'], 
    'vocab_size': 4100,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--datasource', type=str, default="Virus", choices=["Virus", "Bacteria", "Tumor", "ToxDL", "SDAP2"])
    parser.add_argument('--targetsource', type=str, default="Virus", choices=["Virus", "Bacteria", "Tumor", "ToxDL", "SDAP2"])
    parser.add_argument('--num_runs', type=int, default=8)
    parser.add_argument('--testset', type=str, default="test", choices=["test", "independent"])
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()

NUM_RUNS = args.num_runs #50

testset = args.testset # "test" or "independent"
datasource = args.datasource # "Virus" or "Bacteria" or "Tumor"
targetsource = args.targetsource
mutation_prob = "" #"5e-4" #0.2
mutation_rate = "" #"full" # #0.001

# ckpt_root = "./ckpt-{}Immunogen".format(datasource)
# ckpt_root = "./ckpt-ToxDL"
# ckpt_root = "./ckpt-SDAP2-wSelfAttention"
# ckpt_root = "./ckpt-SDAP2-woutFoldSeek"


if args.datasource in ["Virus", "Bacteria", "Tumor"]:
    if 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}Immunogen-SWAG".format(datasource)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}Immunogen-SWAG-woutFoldSeek".format(datasource)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}Immunogen-SWAG-woutESM3".format(datasource)
    else:
        ckpt_root = "./ckpt-{}Immunogen-SWAG-only-ez".format(datasource)
elif args.datasource in ["ToxDL", "SDAP2"]:
    if 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-SWAG".format(datasource)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-SWAG-woutFoldSeek".format(datasource)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-SWAG-woutESM3".format(datasource)
    else:
        ckpt_root = "./ckpt-{}-SWAG-only-ez".format(datasource)
else:
    raise NotImplementedError("Trained Model Unavailable")

if args.seed is not None:
    ckpt_root += f"-seed-{args.seed}"

if args.targetsource in ["Virus", "Bacteria", "Tumor"]:
    datafilepath="./dataset/{}Binary/ESMFold/test.json".format(targetsource)
elif args.targetsource in ["ToxDL"]:
    datafilepath="./ToxDL_Data/json_files/{}_data_with_label.json".format(testset)
elif args.targetsource in ["SDAP2"]:
    datafilepath="./SDAP2_DATA/json_files/test_data_with_label.json"
else:
    raise NotImplementedError("Data Not Available")

# result_save_dir = "./Predict-Results-VBT-UQ-DROPOUT"
# result_save_dir = "./Predict-Results-VBT-UQ-SWAG"
if args.datasource in ["Virus", "Bacteria", "Tumor"]:
    result_save_dir = "./Predict-Results-VBT-UQ-SWAG"
elif args.datasource in ["SDAP2", "ToxDL"]:
    result_save_dir = "./Predict-Results-UQ-SWAG"
else:
    raise NotImplementedError("Data Not Available")

if args.seed is not None:
    result_save_dir += f"-seed-{args.seed}"

os.makedirs(result_save_dir, exist_ok=True)
# pred_filename = "SDAP2_{}.json".format(testset)
# pred_filename = "ToxDL_wSelfAttention_{}_woutESM3.json".format(testset)
# pred_filename = "{}_test.json".format(datasource)

if datasource == targetsource:
    if 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        pred_filename = "{}_{}.json".format(targetsource, testset)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        pred_filename = "{}_{}_woutFoldSeek.json".format(targetsource, testset)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        pred_filename = "{}_{}_woutESM3.json".format(targetsource, testset)
    else:
        pred_filename = "{}_{}_only_ez.json".format(targetsource, testset)
else:
    if 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        pred_filename = "data_{}_target_{}_{}.json".format(datasource, targetsource, testset)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        pred_filename = "data_{}_target_{}_{}_woutFoldSeek.json".format(datasource, targetsource, testset)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        pred_filename = "data_{}_target_{}_{}_woutESM3.json".format(datasource, targetsource, testset)
    else:
        pred_filename = "data_{}_target_{}_{}_only_ez.json".format(datasource, targetsource,testset)

wSelfAttention = False

def process_data_line(
    data, 
    max_seq_len=None, 
    structure_seqs=['ez_descriptor', 'foldseek_seq', 'esm3_structure_seq']
):
    if max_seq_len is not None:
        data["aa_seq"] = data["aa_seq"][:max_seq_len]
        if "foldseek_seq" in structure_seqs:
            data["foldseek_seq"] = data["foldseek_seq"][:max_seq_len]
        token_num = min(len(data["aa_seq"]), max_seq_len)
    else:
        token_num = len(data["aa_seq"])
    return data, token_num

def process_dataset_from_json(
    file, 
    max_seq_len=None,
    structure_seqs=['ez_descriptor', 'foldseek_seq', 'esm3_structure_seq'],
):
    dataset, token_nums = [], []
    for l in open(file):
        data = json.loads(l)
        data, token_num = process_data_line(
            data, 
            max_seq_len,
            structure_seqs=structure_seqs,
        )
        dataset.append(data)
        token_nums.append(token_num)
    return dataset, token_nums

def collate_fn(
    examples,
    structure_seqs=['ez_descriptor', 'foldseek_seq', 'esm3_structure_seq'],
):
    aa_seqs, names, labels = [], [], []
    e_descriptor, z_descriptor = [], []

    if 'foldseek_seq' in structure_seqs:
        foldseek_seqs = []
    if 'esm3_structure_seq' in structure_seqs:
        esm3_structure_seqs = []
    
        
    for e in examples:
        aa_seq = e["aa_seq"]
        if 'foldseek_seq' in structure_seqs:
            foldseek_seq = e["foldseek_seq"]
        
        if 'prot_bert' in plm_model_name or "ProstT5" in plm_model_name:
            aa_seq = " ".join(list(aa_seq))
            aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
            if 'foldseek_seq' in structure_seqs:
                foldseek_seq = " ".join(list(foldseek_seq))
        elif 'ankh' in plm_model_name or "esmc" in plm_model_name:# or "ProstT5" in args.plm_model:
            aa_seq = list(aa_seq)
            if 'foldseek_seq' in structure_seqs:
                foldseek_seq = list(foldseek_seq)
        
        aa_seqs.append(aa_seq)
        names.append(e.get("name", ""))

        if 'foldseek_seq' in structure_seqs:
            foldseek_seqs.append(foldseek_seq)
        if 'esm3_structure_seq' in structure_seqs:
            esm3_structure_seq = [VQVAE_SPECIAL_TOKENS["BOS"]] + e["esm3_structure_seq"] + [VQVAE_SPECIAL_TOKENS["EOS"]]
            esm3_structure_seqs.append(torch.tensor(esm3_structure_seq))
        
        if 'ez_descriptor' in structure_seqs:
            e_descriptor.append(torch.tensor(e["e_descriptor"]))
            z_descriptor.append(torch.tensor(e["z_descriptor"]))

        # Get labels
        labels.append(e["label"])
    
    if 'ankh' in plm_model_name or 'esmc' in plm_model_name:
        aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        if 'foldseek_seq' in structure_seqs:
            foldseek_input_ids = tokenizer.batch_encode_plus(foldseek_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
    else:
        aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
        if 'foldseek_seq' in structure_seqs:
            foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    
    aa_input_ids = aa_inputs["input_ids"]
    attention_mask = aa_inputs["attention_mask"]

    labels = torch.as_tensor(labels, dtype=torch.long)

    data_dict = {
        "aa_input_ids": aa_input_ids, 
        "attention_mask": attention_mask,
        "names": names,
        "label": labels,
    }

    if 'ez_descriptor' in structure_seqs:
        # add e-descriptor and z-descriptor embedding
        # pad e_descriptor_embeds to the same length as aa_input_ids
        e_descriptor_embeds = torch.stack([torch.cat([e_descriptor[i], torch.zeros(len(aa_input_ids[i]) - len(e_descriptor[i]), 5)], dim=0) for i in range(len(e_descriptor))])
        z_descriptor_embeds = torch.stack([torch.cat([z_descriptor[i], torch.zeros(len(aa_input_ids[i]) - len(z_descriptor[i]), 3)], dim=0) for i in range(len(z_descriptor))])
        data_dict["e_descriptor_embeds"] = e_descriptor_embeds
        data_dict["z_descriptor_embeds"] = z_descriptor_embeds

    if 'foldseek_seq' in structure_seqs:
        data_dict["foldseek_input_ids"] = foldseek_input_ids
    if 'esm3_structure_seq' in structure_seqs:
        # pad the list of esm3_structure_seq and convert to tensor
        esm3_structure_input_ids = torch.nn.utils.rnn.pad_sequence(
            esm3_structure_seqs, batch_first=True, padding_value=VQVAE_SPECIAL_TOKENS["PAD"]
            )
        if 'ankh' in plm_model_name or "ProstT5" in plm_model_name:
            esm3_structure_input_ids = esm3_structure_input_ids[:,:-1]
        data_dict["esm3_structure_input_ids"] = esm3_structure_input_ids
    return data_dict

# def infer(model, plm_model, dataloader, device, num_runs:int = NUM_RUNS):
#     names, pred_labels, pred_probas, true_labels = [], [], [], []
#     model.train()
#     for batch in tqdm(dataloader):
#         names.extend(batch.pop("names"))
#         true_labels.extend(batch.pop("label").cpu().numpy())
#         for k, v in batch.items():
#             batch[k] = v.to(device)

#         logits_list = []
#         for _ in range(num_runs):
#             with torch.no_grad():
#                 logits = model(plm_model, batch)
#             logits_list.append(logits[:,None,:])
#         logits_list = torch.cat(logits_list, dim=1)

#         pred_labels.extend(logits_list.argmax(dim=-1).cpu().numpy())
#         pred_probas.extend(logits_list.softmax(dim=-1).cpu().numpy())

#     return names, pred_labels, pred_probas, true_labels

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

def infer(model, plm_model, dataloader, device, num_runs:int = NUM_RUNS):
    names, pred_labels, pred_probas, true_labels = [], [], [], []
    # model.train()
    for batch in tqdm(dataloader):
        names.extend(batch.pop("names"))
        true_labels.extend(batch.pop("label").cpu().numpy())
        for k, v in batch.items():
            batch[k] = v.to(device)

        logits_list = []
        for _ in range(num_runs):
            with torch.no_grad():
                model.sample(scale=1, cov=True)
                logits = model(plm_model, batch)
            logits_list.append(logits[:,None,:])
        logits_list = torch.cat(logits_list, dim=1)

        # with torch.no_grad():
        #     logits = model(plm_model, batch)

        pred_labels.extend(logits_list.argmax(dim=-1).cpu().numpy())
        pred_probas.extend(logits_list.softmax(dim=-1).cpu().numpy())

        # pred_labels.extend(logits.argmax(dim=-1).cpu().numpy())
        # pred_probas.extend(logits.softmax(dim=-1)[:,1].cpu().numpy())

    return names, pred_labels, pred_probas, true_labels

pred_dict = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset, test_token_num = process_dataset_from_json(
    file=datafilepath, 
    max_seq_len=None,
    structure_seqs=model_params["structure_seqs"],
)

plm_model_name = "esmc_600m"
tokenizer = get_esmc_model_tokenizers()
plm_model = ESMC.from_pretrained(plm_model_name, device=torch.device(device)).to(torch.float32)
model_params['hidden_size'] = plm_model.embed.embedding_dim

vocab_size = tokenizer.vocab_size
if 'esm3_structure_seq' in model_params['structure_seqs']: 
    model_params['vocab_size'] = max(vocab_size, 4100)
else:
    model_params['vocab_size'] = vocab_size

# model = AdapterModel(argparse.Namespace(**model_params), initial_seq_layer_norm=True, self_attn=wSelfAttention).to(device).eval()

model = SWAG(AdapterModel, 
            argparse.Namespace(**model_params), 
            initial_seq_layer_norm=True, self_attn=wSelfAttention,
            no_cov_mat=False, 
            max_num_models=64).to(device).eval()

model_name = "esmc_wiln"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)
# print(model)

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device)

for name, pred_label, pred_prob, true_label in zip(names, pred_labels, pred_probas, true_labels):
    pred_dict[name] = {}
    pred_dict[name]['pred_label'], pred_dict[name]['pred_prob'] = {}, {}
    pred_dict[name]['pred_label'][plm_model_name] = np.int8(pred_label)
    pred_dict[name]['pred_prob'][plm_model_name] = pred_prob
    pred_dict[name]['true_label'] = np.int8(true_label)

plm_model_name = "Rostlab/ProstT5"
tokenizer = T5Tokenizer.from_pretrained(plm_model_name, do_lower_case=False)
plm_model = T5EncoderModel.from_pretrained(plm_model_name).to(device).eval()
model_params['hidden_size'] = plm_model.config.d_model

vocab_size = plm_model.config.vocab_size
if 'esm3_structure_seq' in model_params['structure_seqs']: 
    model_params['vocab_size'] = max(vocab_size, 4100)
else:
    model_params['vocab_size'] = vocab_size

# model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention).to(device).eval()

model = SWAG(AdapterModel, 
            argparse.Namespace(**model_params), 
            self_attn=wSelfAttention,
            no_cov_mat=False, 
            max_num_models=64).to(device).eval()

model_name = "prost_t5"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device)

for name, pred_label, pred_prob in zip(names, pred_labels, pred_probas):
    # pred_dict[name] = {}
    # pred_dict[name]['pred_label'], pred_dict[name]['pred_prob'] = {}, {}
    pred_dict[name]['pred_label'][plm_model_name] = np.int8(pred_label)
    pred_dict[name]['pred_prob'][plm_model_name] = pred_prob
    # pred_dict[name]['true_label'] = true_label

plm_model_name = "ElnaggarLab/ankh-large"
tokenizer = AutoTokenizer.from_pretrained(plm_model_name, do_lower_case=False, clean_up_tokenization_spaces=True)
plm_model = T5EncoderModel.from_pretrained(plm_model_name).to(device).eval()
model_params['hidden_size'] = plm_model.config.d_model

vocab_size = plm_model.config.vocab_size
if 'esm3_structure_seq' in model_params['structure_seqs']: 
    model_params['vocab_size'] = max(vocab_size, 4100)
else:
    model_params['vocab_size'] = vocab_size

model = SWAG(AdapterModel, 
            argparse.Namespace(**model_params), 
            self_attn=wSelfAttention,
            no_cov_mat=False, 
            max_num_models=64).to(device).eval()

model_name = "ankh"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device)

for name, pred_label, pred_prob in zip(names, pred_labels, pred_probas):
    # pred_dict[name] = {}
    # pred_dict[name]['pred_label'], pred_dict[name]['pred_prob'] = {}, {}
    pred_dict[name]['pred_label'][plm_model_name] = np.int8(pred_label)
    pred_dict[name]['pred_prob'][plm_model_name] = pred_prob
    # pred_dict[name]['true_label'] = true_label

plm_model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(plm_model_name)
plm_model = EsmModel.from_pretrained(plm_model_name, output_hidden_states=True).to(device).eval()
model_params['hidden_size'] = plm_model.config.hidden_size

vocab_size = plm_model.config.vocab_size
if 'esm3_structure_seq' in model_params['structure_seqs']: 
    model_params['vocab_size'] = max(vocab_size, 4100)
else:
    model_params['vocab_size'] = vocab_size

model = SWAG(AdapterModel, 
            argparse.Namespace(**model_params), 
            self_attn=wSelfAttention,
            no_cov_mat=False, 
            max_num_models=64).to(device).eval()

model_name = "esm2"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device), strict=True)

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device)

for name, pred_label, pred_prob in zip(names, pred_labels, pred_probas):
    # pred_dict[name] = {}
    # pred_dict[name]['pred_label'], pred_dict[name]['pred_prob'] = {}, {}
    pred_dict[name]['pred_label'][plm_model_name] = np.int8(pred_label)
    pred_dict[name]['pred_prob'][plm_model_name] = pred_prob
    # pred_dict[name]['true_label'] = true_label

plm_model_name = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(plm_model_name, do_lower_case=False)
plm_model = BertModel.from_pretrained(plm_model_name).to(device).eval()
model_params['hidden_size'] = plm_model.config.hidden_size

vocab_size = plm_model.config.vocab_size
if 'esm3_structure_seq' in model_params['structure_seqs']: 
    model_params['vocab_size'] = max(vocab_size, 4100)
else:
    model_params['vocab_size'] = vocab_size

model = SWAG(AdapterModel, 
            argparse.Namespace(**model_params), 
            self_attn=wSelfAttention,
            no_cov_mat=False, 
            max_num_models=64).to(device).eval()

model_name = "prot_bert"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device), strict=True)

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device)

for name, pred_label, pred_prob in zip(names, pred_labels, pred_probas):
    # pred_dict[name] = {}
    # pred_dict[name]['pred_label'], pred_dict[name]['pred_prob'] = {}, {}
    pred_dict[name]['pred_label'][plm_model_name] = np.int8(pred_label)
    pred_dict[name]['pred_prob'][plm_model_name] = pred_prob
    # pred_dict[name]['true_label'] = true_label

with open(os.path.join(result_save_dir, pred_filename), "w", encoding="utf8") as file:
    json.dump(pred_dict, file, default=str)

# print(pred_dict)