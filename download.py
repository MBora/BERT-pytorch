from datasets import load_dataset

# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Define the file path for saving the preprocessed dataset
preprocessed_file_path = './data/wikitext-2-preprocessed.txt'

# Function to preprocess and save the dataset
def preprocess_and_save(dataset, split, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in dataset[split]:
            text = item['text']
            sentences = text.split('\n')
            for i in range(len(sentences) - 1):
                if sentences[i] and sentences[i + 1]:  # Ensure both sentences are not empty
                    f.write(sentences[i] + '\t' + sentences[i + 1] + '\n')

# Preprocess and save the train, validation, and test splits
for split in ['train', 'validation', 'test']:
    preprocess_and_save(dataset, split, f'/mnt/data/wikitext-2-preprocessed-{split}.txt')

print("WikiText-2 dataset has been preprocessed and saved.")
