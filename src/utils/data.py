"""Class for preprocessing the data, including tokenization, etc."""

# typing imports
import string
from collections import defaultdict

# typing imports
from typing import Dict, List, Tuple

import torch
from torch.utils.data.sampler import Sampler
from transformers import PreTrainedTokenizerFast

from src.config import BabyLMConfig

#modified to include sem-tags, alongside UPOS (Suchir Salhan)
POS_TAG_MAP = {
    "NOUN": 0,
    "VERB": 1,
    "ADJ": 2,
    "ADV": 3,
    "PRON": 4,
    "DET": 5,
    "ADP": 6,
    "NUM": 7,
    "CONJ": 8,
    "PRT": 9,
    ".": 10,
    "X": 11,
    "INTJ": 12,
    "PROPN": 13,
    "CCONJ": 14,
    "SCONJ": 15,
    "SYM": 16,
    "PUNCT": 17,
    "AUX": 18,
    "PART": 19,
    "ADP": 20,
    "PRO": 22,
    "DEF": 23,
    "HAS": 24,
    "REF": 25,
    "EMP": 26,
    "GRE": 27,
    "ITJ": 28,
    "HES": 29,
    "QUE": 30,
    "QUA": 31,
    "UOM": 32,
    "IST": 33,
    "REL": 34,
    "RLI": 35,
    "SST": 36,
    "PRI": 37,
    "INT": 38,
    "SCO": 39,
    "ALT": 40,
    "EXC": 41,
    "NIL": 42,
    "DIS": 43,
    "IMP": 44,
    "AND": 45,
    "BUT": 46,
    "EQA": 47,
    "MOR": 48,
    "LES": 49,
    "TOP": 50,
    "BOT": 51,
    "ORD": 52,
    "PRX": 53,
    "MED": 54,
    "DST": 55,
    "SUB": 56,
    "COO": 57,
    "APP": 58,
    "NOT": 59,
    "NEC": 60,
    "POS": 61,
    "CON": 62,
    "ROL": 63,
    "GPE": 64,
    "PER": 65,
    "LOC": 66,
    "ORG": 67,
    "ART": 68,
    "NAT": 69,
    "HAP": 70,
    "URL": 71,
    "EXS": 72,
    "ENS": 73,
    "EPS": 74,
    "EFS": 75,
    "EXG": 76,
    "ENG": 77,
    "EPG": 78,
    "EFG": 79,
    "EXT": 80,
    "ENT": 81,
    "EPT": 82,
    "EFT": 83,
    "ETG": 84,
    "ETV": 85,
    "EXV": 86,
    "NOW": 87,
    "PST": 88,
    "FUT": 89,
    "DOM": 90,
    "YOC": 91,
    "DOW": 92,
    "MOY": 93,
    "DEC": 94,
    "CLO": 95,
    "MAN": 96,  # added tag
    "RES": 97,  # added tag
    "AIM": 98,  # added tag
    "OBJ": 99,  # added tag
    "COM": 100  # added tag
}


def base_collate_fn(_samples: List[Dict[str, List[Tuple[int, float]]]]):
    joined_batch = defaultdict(list)
    for sample in _samples:
        for key, val in sample.items():
            joined_batch[key].append(torch.tensor(val))

    batch = {}

    for key, val in joined_batch.items():
        batch[key] = torch.stack(val)

    return batch


class SequentialSubsetSampler(Sampler):
    """
    Samples elements sequentially from a set of indices, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DatasetPreprocessor(object):
    def __init__(self, cfg: BabyLMConfig, tokenizer: PreTrainedTokenizerFast):
        """
        Args:
            cfg (BabyLMConfig): hydra config object
            tokenizer (PreTrainedTokenizer): instantiated tokenizer object
        """

        # data processing params
        self.include_punctuation = cfg.data_preprocessing.include_punctuation
        self.max_input_length = cfg.data_preprocessing.max_input_length
        self.join_sentences = cfg.data_preprocessing.join_sentences
        self.callback_functions = cfg.data_preprocessing.callback_functions
        self.dataset_subconfig = cfg.dataset.subconfig

        self.tokenizer = tokenizer

    ### --- Callback functions --- ###

    # NOTE: The function names of callbacks must match the names in the data preprocessing
    # callback_functions list (sepcified in the config file)

    ### --- Callback functions --- ###

# Replace the entire __call__ method (from line 191 to the end of the function) with this:
    def __call__(self, examples):
        if not self.include_punctuation:
            examples["text"] = [
                line.translate(str.maketrans("", "", string.punctuation))
                for line in examples["text"]
            ]
    
        batch = {
            "input_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "pos_tags": [],
            "filename": [], # We'll add a dummy filename
        }
    
        full_tokenized_inputs = {
            "input_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "pos_tags": [],
            "filename": [],
        }
    
        for example in range(len(examples["text"])):
            text = examples["text"][example]
            # We don't have tagged_text or filename, so we create dummies
            filename = "local_file"
    
            tokenized_inputs = self.tokenizer(
                text,
                pad_to_multiple_of=self.max_input_length
                if not self.join_sentences
                else None,
                padding="longest" if not self.join_sentences else "do_not_pad",
                max_length=self.max_input_length
                if not self.join_sentences
                else None,
                truncation=False,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
            )
    
            # Since we loaded from raw text, we don't have POS tags.
            # We will create placeholder tags (the "unknown" tag 'X').
            pos_tags = [POS_TAG_MAP["X"]] * len(
                tokenized_inputs["input_ids"]
            )
    
            if self.join_sentences:
                full_tokenized_inputs["input_ids"].extend(
                    tokenized_inputs["input_ids"]
                )
                full_tokenized_inputs["special_tokens_mask"].extend(
                    tokenized_inputs["special_tokens_mask"]
                )
                full_tokenized_inputs["attention_mask"].extend(
                    tokenized_inputs["attention_mask"]
                )
                full_tokenized_inputs["pos_tags"].extend(pos_tags)
                full_tokenized_inputs["filename"].extend(
                    [filename] * len(tokenized_inputs["input_ids"])
                )
            else:
                # Split into multiple examples if the input is too long
                for i in range(
                    0,
                    len(tokenized_inputs["input_ids"]),
                    self.max_input_length,
                ):
                    if (
                        sum(
                            tokenized_inputs["special_tokens_mask"][
                                i : i + self.max_input_length
                            ]
                        )
                        == self.max_input_length
                    ):
                        break
                    batch["input_ids"].append(
                        tokenized_inputs["input_ids"][i : i + self.max_input_length]
                    )
                    batch["special_tokens_mask"].append(
                        tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]
                    )
                    batch["attention_mask"].append(
                        tokenized_inputs["attention_mask"][i : i + self.max_input_length]
                    )
                    batch["pos_tags"].append(
                        pos_tags[i : i + self.max_input_length]
                    )
                    batch["filename"].append(filename)
                if len(batch["pos_tags"][-1]) < self.max_input_length:
                    batch["pos_tags"][-1].extend(
                        [POS_TAG_MAP["X"]]
                        * (self.max_input_length - len(batch["pos_tags"][-1]))
                    )
    
        if self.join_sentences:
            truncated_length = (
                len(full_tokenized_inputs["input_ids"])
                // self.max_input_length
            ) * self.max_input_length
    
            for i in range(0, truncated_length, self.max_input_length):
                batch["input_ids"].append(
                    full_tokenized_inputs["input_ids"][i : i + self.max_input_length]
                )
                batch["special_tokens_mask"].append(
                    full_tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]
                )
                batch["attention_mask"].append(
                    full_tokenized_inputs["attention_mask"][i : i + self.max_input_length]
                )
                batch["pos_tags"].append(
                    full_tokenized_inputs["pos_tags"][i : i + self.max_input_length]
                )
                batch["filename"].append(full_tokenized_inputs["filename"][i])
    
        if self.callback_functions:
            for callback_function in self.callback_functions:
                examples[callback_function] = getattr(self, callback_function)(
                    examples
                )
    
        return batch
