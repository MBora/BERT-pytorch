import torchtext
from torchtext.datasets import WikiText2

# Define the file path where you want to save the dataset
file_path = './data/wikitext-2-train.txt'

# Download and load the WikiText-2 training dataset
train_iter = WikiText2(split='train')

# Open a file in write mode
with open(file_path, 'w', encoding='utf-8') as f:
    for line in train_iter:
        # Write each line to the file. Each line returned from WikiText2 is a tuple with a single string element.
        f.write(line[0] + '\n')  # Add a newline character to keep the original line breaks

print(f"WikiText-2 training dataset has been saved to {file_path}")
