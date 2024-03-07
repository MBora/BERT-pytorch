from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.datas = []

        with open(corpus_path, "r", encoding=encoding) as f:
            for line in f:
                if "\t" in line:
                    sentence1, sentence2 = line.strip().split("\t")  # Split line into two sentences
                    self.datas.append((sentence1, sentence2))


        self.prepare_negative_samples()

    def prepare_negative_samples(self):
        indices = list(range(len(self.datas)))
        random.shuffle(indices)
        self.negative_indices = indices

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # Select next or random sentence
        if random.random() > 0.5:
            is_next = 1
        else:
            random_index = self.negative_indices[index % len(self.negative_indices)]
            t2 = self.get_random_line(random_index)
            is_next = 0

        return t1, t2, is_next

    def get_corpus_line(self, item):
        return self.datas[item][0], self.datas[item][1]

    def get_random_line(self, index):
        return self.datas[index][1]

class BERTDatasetDual(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.datas = []

        with open(corpus_path, "r", encoding=encoding) as f:
            for line in f:
                if "\t" in line:
                    sentence1, sentence2 = line.strip().split("\t")  # Split line into two sentences
                    self.datas.append((sentence1, sentence2))

        self.prepare_negative_samples()

    def prepare_negative_samples(self):
        indices = list(range(len(self.datas)))
        random.shuffle(indices)
        self.negative_indices = indices
        
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t3, t4, is_next_label2 = self.random_sent(item)

        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        t3_random, t3_label = self.random_word(t3)
        t4_random, t4_label = self.random_word(t4)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t3 = [self.vocab.sos_index] + t3_random + [self.vocab.eos_index]
        t4 = t4_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        t3_label = [self.vocab.pad_index] + t3_label + [self.vocab.pad_index]
        t4_label = t4_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        segment_label2 = ([1 for _ in range(len(t3))] + [2 for _ in range(len(t4))])[:self.seq_len]

        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        bert_input2 = (t3 + t4)[:self.seq_len]
        bert_label2 = (t3_label + t4_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        padding2 = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input2))]
        bert_input2.extend(padding2), bert_label2.extend(padding2), segment_label2.extend(padding2)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                    "bert_input2": bert_input2,
                    "bert_label2": bert_label2,
                  "segment_label": segment_label,
                  "segment_label2": segment_label2,
                  "is_next": is_next_label,
                  "is_next2": is_next_label2
                  }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # Select next or random sentence
        if random.random() > 0.5:
            is_next = 1
        else:
            random_index = self.negative_indices[index % len(self.negative_indices)]
            t2 = self.get_random_line(random_index)
            is_next = 0

        return t1, t2, is_next

    def get_corpus_line(self, item):
        return self.datas[item][0], self.datas[item][1]

    def get_random_line(self, index):
        return self.datas[index][1]
