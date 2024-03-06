import torchtext
from torchtext.datasets import WikiText2
import os

# Define the file path for the preprocessed dataset
preprocessed_file_path = '/mnt/data/wikitext-2-preprocessed.txt'

def preprocess_and_save_wikitext2(file_path):
    # Download the WikiText-2 dataset
    train_iter, val_iter, test_iter = WikiText2()

    with open(file_path, 'w', encoding='utf-8') as out_file:
        for iter in [train_iter, val_iter, test_iter]:
            prev_sentence = None
            for raw_sentence in iter:
                sentence = raw_sentence[0]  # Extract sentence from tuple
                if prev_sentence and sentence:  # Ensure there is a pair to write
                    out_file.write(f"{prev_sentence}\t{sentence}\n")
                prev_sentence = sentence

preprocess_and_save_wikitext2(preprocessed_file_path)
