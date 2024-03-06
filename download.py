from datasets import load_dataset
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Define the file path for saving the preprocessed dataset
preprocessed_file_path = './data/wikitext-2-preprocessed.txt'

def preprocess_and_save(dataset, split, base_path):
    file_path = f'{base_path}-preprocessed-{split}.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in dataset[split]:
            text = item['text']
            sentences = sent_tokenize(text)
            for i in range(len(sentences) - 1):
                # Ensure both sentences are not empty and write them separated by a tab
                if sentences[i] and sentences[i + 1]:
                    f.write(sentences[i] + '\t' + sentences[i + 1] + '\n')

# Preprocess and save the train, validation, and test splits
for split in ['train', 'validation', 'test']:
    preprocess_and_save(dataset, split, './data/wikitext-2')

print("WikiText-2 dataset has been preprocessed and saved.")
