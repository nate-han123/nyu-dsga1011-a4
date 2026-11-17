import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


def custom_transform(example):
    """
    Synonym replacement transformation.

    For each word in the text, with probability 0.3, replace it with a random
    WordNet synonym (from any synset). Words with no synonyms are kept as-is.
    This transformation preserves the sentiment label: a human reading the
    transformed text would assign the same positive/negative label.  It is
    "reasonable" because real users naturally paraphrase with synonyms.
    """
    words = word_tokenize(example["text"])
    new_words = []
    detokenizer = TreebankWordDetokenizer()

    for word in words:
        if random.random() < 0.3:
            synsets = wordnet.synsets(word)
            synonyms = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace('_', ' ')
                    if candidate.lower() != word.lower():
                        synonyms.append(candidate)
            if synonyms:
                new_words.append(random.choice(synonyms))
            else:
                new_words.append(word)
        else:
            new_words.append(word)

    example["text"] = detokenizer.detokenize(new_words)
    return example
