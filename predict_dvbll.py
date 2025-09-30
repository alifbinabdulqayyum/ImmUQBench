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

FEAT_DIM = 64

model_params = {
    'hidden_size': None,  # 将由PLM模型自动设置
    'num_attention_heads': 8,
    'attention_probs_dropout_prob': 0,
    'num_labels': FEAT_DIM, #2,
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
    parser.add_argument('--testset', type=str, default="test", choices=["test", "independent", "valid"])
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()

NUM_RUNS = args.num_runs #50

testset = args.testset # "test" or "independent" or "valid"
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
        ckpt_root = "./ckpt-{}Immunogen-DVBLL".format(datasource)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}Immunogen-DVBLL-woutFoldSeek".format(datasource)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}Immunogen-DVBLL-woutESM3".format(datasource)
    else:
        ckpt_root = "./ckpt-{}Immunogen-DVBLL-only-ez".format(datasource)
elif args.datasource in ["ToxDL", "SDAP2"]:
    if 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-DVBLL".format(datasource)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-DVBLL-woutFoldSeek".format(datasource)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-DVBLL-woutESM3".format(datasource)
    else:
        ckpt_root = "./ckpt-{}-DVBLL-only-ez".format(datasource)
else:
    raise NotImplementedError("Trained Model Unavailable")

if args.seed is not None:
    ckpt_root += f"-seed-{args.seed}"

# datafilepath="./ToxDL_Data/json_files/{}_data_with_label.json".format(testset)
# datafilepath="./SDAP2_DATA/json_files/{}_data_with_label.json".format(testset)
# datafilepath="./dataset/{}Binary/ESMFold/test.json".format(targetsource)

if args.targetsource in ["Virus", "Bacteria", "Tumor"]:
    datafilepath="./dataset/{}Binary/ESMFold/{}.json".format(targetsource, testset)
elif args.targetsource in ["ToxDL"]:
    datafilepath="./ToxDL_Data/json_files/{}_data_with_label.json".format(testset)
elif args.targetsource in ["SDAP2"]:
    datafilepath="./SDAP2_DATA/json_files/{}_data_with_label.json".format(testset)
else:
    raise NotImplementedError("Data Not Available")

# result_save_dir = "./Predict-Results-VBT-UQ-DVBLL"
# os.makedirs(result_save_dir, exist_ok=True)

if args.datasource in ["Virus", "Bacteria", "Tumor"]:
    result_save_dir = "./Predict-Results-VBT-UQ-DVBLL"
elif args.datasource in ["SDAP2", "ToxDL"]:
    result_save_dir = "./Predict-Results-UQ-DVBLL"
else:
    raise NotImplementedError("Data Not Available")

if args.seed is not None:
    result_save_dir += f"-seed-{args.seed}"

os.makedirs(result_save_dir, exist_ok=True)

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

print(pred_filename, datafilepath)

wSelfAttention = False

import vbll

class DiscVBLLMLP(torch.nn.Module):
    def __init__(self, feature_extractor, datapoints:int, num_labels:int = 2):
        super(DiscVBLLMLP, self).__init__()

        self.feature_extractor = feature_extractor
        self.bayes_layer = vbll.DiscClassification(
        # self.bayes_layer = vbll.GenClassification(
            self.feature_extractor.classifier.attention1d_projection.final.out_features, 
            num_labels, 
            1.0/datapoints, 
            parameterization = 'dense', 
            return_ood=True, 
            prior_scale=1.0
        )

    def forward(self, plm_model, batch, n_samples:int = NUM_RUNS):
        features = self.feature_extractor(plm_model, batch)
        output = self.bayes_layer.logit_predictive(features).rsample(sample_shape=torch.Size([n_samples]))
        return output.transpose(1, 0)

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

def infer(model, plm_model, dataloader, device, num_runs:int = NUM_RUNS):
    names, pred_labels, pred_probas, true_labels = [], [], [], []
    for batch in tqdm(dataloader):
        names.extend(batch.pop("names"))
        true_labels.extend(batch.pop("label").cpu().numpy())
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            logits_list = model(plm_model, batch, num_runs)

        pred_labels.extend(logits_list.argmax(dim=-1).cpu().numpy())
        pred_probas.extend(logits_list.softmax(dim=-1).cpu().numpy())

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

feat_model = AdapterModel(argparse.Namespace(**model_params), initial_seq_layer_norm=True, self_attn=wSelfAttention)#.to(device).eval()

model_name = "esmc_wiln"

model = DiscVBLLMLP(feature_extractor=feat_model, datapoints=10000, num_labels=2).to(device).eval()

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

feat_model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention)#.to(device).eval()

model_name = "prost_t5"

model = DiscVBLLMLP(feature_extractor=feat_model, datapoints=10000, num_labels=2).to(device).eval()

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

feat_model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention)#.to(device).eval()

model_name = "ankh"

model = DiscVBLLMLP(feature_extractor=feat_model, datapoints=10000, num_labels=2).to(device).eval()

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

feat_model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention)#.to(device).eval()

model_name = "esm2"

model = DiscVBLLMLP(feature_extractor=feat_model, datapoints=10000, num_labels=2).to(device).eval()

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

feat_model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention)#.to(device).eval()

model_name = "prot_bert"

model = DiscVBLLMLP(feature_extractor=feat_model, datapoints=10000, num_labels=2).to(device).eval()

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