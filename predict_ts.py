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
    'structure_seqs': ['ez_descriptor', 'foldseek_seq', 'esm3_structure_seq'], #
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
        ckpt_root = "./ckpt-{}Immunogen".format(datasource)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}Immunogen-woutFoldSeek".format(datasource)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}Immunogen-woutESM3".format(datasource)
    else:
        ckpt_root = "./ckpt-{}Immunogen-only-ez".format(datasource)
elif args.datasource in ["ToxDL", "SDAP2"]:
    if 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}".format(datasource)
    elif not 'foldseek_seq' in model_params['structure_seqs'] and 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-woutFoldSeek".format(datasource)
    elif 'foldseek_seq' in model_params['structure_seqs'] and not 'esm3_structure_seq' in model_params['structure_seqs']:
        ckpt_root = "./ckpt-{}-woutESM3".format(datasource)
    else:
        ckpt_root = "./ckpt-{}-only-ez".format(datasource)
else:
    raise NotImplementedError("Trained Model Unavailable")

if args.seed is not None:
    ckpt_root += f"-seed-{args.seed}"

if args.targetsource in ["Virus", "Bacteria", "Tumor"]:
    val_datafilepath="./dataset/{}Binary/ESMFold/valid.json".format(datasource)
    test_datafilepath="./dataset/{}Binary/ESMFold/test.json".format(targetsource)
elif args.targetsource in ["ToxDL"]:
    # datafilepath="./ToxDL_Data/json_files/{}_data_with_label.json".format(testset)
    val_datafilepath="./ToxDL_Data/json_files/valid_data_with_label.json"
    test_datafilepath="./ToxDL_Data/json_files/{}_data_with_label.json".format(testset)
elif args.targetsource in ["SDAP2"]:
    # datafilepath="./SDAP2_DATA/json_files/test_data_with_label.json"
    val_datafilepath="./SDAP2_DATA/json_files/valid_data_with_label.json"
    test_datafilepath="./SDAP2_DATA/json_files/test_data_with_label.json"
else:
    raise NotImplementedError("Data Not Available")

# val_datafilepath="./dataset/{}Binary/ESMFold/valid.json".format(targetsource)
# test_datafilepath="./dataset/{}Binary/ESMFold/test.json".format(targetsource)

# result_save_dir = "./Predict-Results-VBT-UQ-DROPOUT"
# result_save_dir = "./Predict-Results-VBT-UQ-LA"
if args.datasource in ["Virus", "Bacteria", "Tumor"]:
    result_save_dir = "./Predict-Results-VBT-TS"
elif args.datasource in ["SDAP2", "ToxDL"]:
    result_save_dir = "./Predict-Results-TS"
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

class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class ModelWithTemperature(torch.nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        # self.model = model
        # self.plm_model = plm_model
        self.temperature = torch.nn.Parameter(torch.ones(1))

    # def forward(self, batch):
    #     batch.pop("names")
    #     logits = self.model(self.plm_model, batch)
    #     return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, model, plm_model, valid_loader, device):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        model.to(device)
        plm_model.to(device)
        nll_criterion = torch.nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in valid_loader:
                batch.pop("names")
                for k, v in batch.items():
                    batch[k] = v.to(device)
                logits = model(plm_model, batch)
                logits_list.append(logits)
                labels_list.append(batch['label'])
            logits = torch.cat(logits_list)#.cuda()
            labels = torch.cat(labels_list)#.cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self.temperature.item()

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

def infer(model, plm_model, dataloader, device, temperature:float=1.0):
    names, pred_labels, pred_probas, true_labels = [], [], [], []
    # model.train()
    for batch in tqdm(dataloader):
        names.extend(batch.pop("names"))
        true_labels.extend(batch.pop("label").cpu().numpy())
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            logits = model(plm_model, batch)/temperature

        pred_labels.extend(logits.argmax(dim=-1).cpu().numpy())
        pred_probas.extend(logits.softmax(dim=-1)[:,1].cpu().numpy())

    return names, pred_labels, pred_probas, true_labels

pred_dict = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

val_dataset, val_token_num = process_dataset_from_json(
    file=val_datafilepath, 
    max_seq_len=None,
    structure_seqs=model_params["structure_seqs"],
)

test_dataset, test_token_num = process_dataset_from_json(
    file=test_datafilepath, 
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

model = AdapterModel(argparse.Namespace(**model_params), initial_seq_layer_norm=True, self_attn=wSelfAttention).to(device).eval()

model_name = "esmc_wiln"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)
# print(model)

val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, 40000, False)
    )

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

temp_model = ModelWithTemperature().to(device)
temperature = temp_model.set_temperature(model, plm_model, val_loader, device=device)

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device, temperature)

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

model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention).to(device).eval()

model_name = "prost_t5"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)

val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, 40000, False)
    )

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

temp_model = ModelWithTemperature().to(device)
temperature = temp_model.set_temperature(model, plm_model, val_loader, device=device)

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device, temperature)

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

model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention).to(device).eval()

model_name = "ankh"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)

val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, 40000, False)
    )

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

temp_model = ModelWithTemperature().to(device)
temperature = temp_model.set_temperature(model, plm_model, val_loader, device=device)

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device, temperature)

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

model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention).to(device).eval()

model_name = "esm2"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device), strict=True)

val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, 40000, False)
    )

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

temp_model = ModelWithTemperature().to(device)
temperature = temp_model.set_temperature(model, plm_model, val_loader, device=device)

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device, temperature)

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

model = AdapterModel(argparse.Namespace(**model_params), self_attn=wSelfAttention).to(device).eval()

model_name = "prot_bert"

model_path = os.path.join(ckpt_root, "{}/ESMFold_{}_attention1d_{}_{}.pt".format(model_name, model_name, mutation_prob, mutation_rate))
model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device), strict=True)

val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(val_token_num, 40000, False)
    )

test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, 40000, False)
    )

temp_model = ModelWithTemperature().to(device)
temperature = temp_model.set_temperature(model, plm_model, val_loader, device=device)

with torch.no_grad():
    names, pred_labels, pred_probas, true_labels = infer(model, plm_model, test_loader, device, temperature)

for name, pred_label, pred_prob in zip(names, pred_labels, pred_probas):
    # pred_dict[name] = {}
    # pred_dict[name]['pred_label'], pred_dict[name]['pred_prob'] = {}, {}
    pred_dict[name]['pred_label'][plm_model_name] = np.int8(pred_label)
    pred_dict[name]['pred_prob'][plm_model_name] = pred_prob
    # pred_dict[name]['true_label'] = true_label

with open(os.path.join(result_save_dir, pred_filename), "w", encoding="utf8") as file:
    json.dump(pred_dict, file, default=str)

# print(pred_dict)