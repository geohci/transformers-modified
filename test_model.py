from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np
import logging
import os
import argparse
from pathlib import Path
logging.basicConfig(level=logging.ERROR)

def prepare_inputs(inputs, device):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
            elif isinstance(v, dict):
                for key, val in v.items():
                    if isinstance(val, torch.Tensor):
                        v[key] = val.to(device)
                    elif isinstance(val, BatchEncoding):
                        for k1,v1 in val.items():
                            if isinstance(v1, torch.Tensor):
                                val[k1] = v1.to(device)
        return inputs

parser = argparse.ArgumentParser()
parser.add_argument("--model_path_or_name", type=str, help="path to folder with trained model")
parser.add_argument("--tokenizer_path", type=str, help="path to folder with tokenizer")
parser.add_argument("--bert_path", default=None, type=str, help="path to folder with bert model (for summary embeddings)")
parser.add_argument("--bert_tokenizer", default=None, type=str, help="path to folder bert tokenizer (for summary embeddings)")
parser.add_argument("--use_graph_embds", help="whether to use graph embeddings", action="store_true")
parser.add_argument("--langs", type=str, help="list of language codes")
parser.add_argument("--data_dir", type=str, help="directory to store the data")
parser.add_argument("--output_folder", type=str, help="path to the folder where to save outputs")

args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

model = MBartForConditionalGeneration.from_pretrained(args.model_path_or_name)
tokenizer = MBartTokenizer.from_pretrained(args.tokenizer_path)
if args.bert_path is not None:
    tokenizer_bert = BertTokenizer.from_pretrained(args.bert_tokenizer)
    bert_model = BertModel.from_pretrained(args.bert_path)
    model.model_bert = bert_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

outputs = []

sources = {}
lang_dict = {}
languages = args.langs.strip().split(',')
for lang in languages:
    lang_dict[lang[0:2]] = lang

sources = {}
targets = {}
    
for lang, lang_code in lang_dict.items():
    f = open(Path(args.data_dir).joinpath("test" + ".source" + lang), 'r', encoding='utf-8')
    p = open(Path(args.data_dir).joinpath("test" + ".target" + lang), 'r', encoding='utf-8')
    lines = f.readlines()
    t_lines = p.readlines()
    sources[lang] = lines
    targets[lang] = t_lines
    f.close()

f = open(Path(args.data_dir).joinpath("test" + ".embd"), 'r', encoding='utf-8')
embds = f.readlines()
f.close()

outputs = open(Path(args.output_folder).joinpath("outputs.txt"), 'w', encoding='utf-8')
target_file = open(Path(args.output_folder).joinpath("mod_targets.txt"), 'w', encoding='utf-8')
lang_file = open(Path(args.output_folder).joinpath("lang_list.txt"), 'w', encoding='utf-8')

for i in range(len(embds)):
    batch = {}
    batch_encodings = {}
    remaining_langs = []
    available_langs = []
    target_langs = []
    for lang, lang_code in lang_dict.items():
        txt = sources[lang][i].strip()

        if targets[lang][i].strip() != "no article":
            target_langs.append(lang)

        if txt == "no article":
            remaining_langs.append(lang)
            continue
        src_lang = lang_code
        tokenizer.src_lang = src_lang
        batch_enc = tokenizer([txt])
        batch_encodings[lang] = batch_enc
        available_langs.append(lang_code)    
    input_ids = {}
    attention_mask = {}
    for key, val in batch_encodings.items():
        inputs = val["input_ids"]
        masks = val["attention_mask"]
        inputs = torch.tensor(inputs)
        masks = torch.tensor(masks)
        input_ids[key] = inputs
        attention_mask[key] = masks
    
    for lang in remaining_langs:
        input_ids[lang] = None
        attention_mask[lang] = None

    #graph embeddings
    if args.use_graph_embds:
        embd = embds[i].strip()
        embds_line = np.array([float(x) for x in embd.split()])
        embds_line = embds_line.astype(np.float32)
        embds_line = torch.tensor([embds_line])
    else:
        embds_line = None

    #summary embeddings
    if args.bert_path is not None:
        bert_inputs = {}
        for lang in target_langs:
            lang = lang[0:2]
            bert_outs = tokenizer_bert(
                [targets[lang][i].strip()], 
                return_tensors="pt",
            )
            bert_inputs[lang] = bert_outs
    else:
        bert_inputs = None
                
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["graph_embeddings"] = embds_line
    batch["bert_inputs"] = bert_inputs

    batch = prepare_inputs(batch, device)
    
    for tgt_lang in target_langs:
        target_lang = lang_dict[tgt_lang]
        target = targets[tgt_lang][i]
        if args.bert_path is not None:
            bert_inputs_modified = bert_inputs.copy()
            bert_inputs_modified.pop(tgt_lang)
            batch["bert_inputs"] = bert_inputs_modified
        translated_tokens = model.generate(**batch, max_length=20, min_length=2, length_penalty=2.0, num_beams=4, early_stopping=True, target_lang = target_lang, decoder_start_token_id=tokenizer.lang_code_to_id[target_lang])
        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        outputs.write(output+"\n")
        target_file.write(target)
        lang_file.write(target_lang+"\n")
    
outputs.close()
target_file.close()
lang_file.close()
