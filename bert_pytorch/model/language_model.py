import torch.nn as nn

from model.bert import BERT

def kl_divergence(p, q):
    # Ensure the division is stable using a small epsilon value
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    # Using PyTorch's batch-wise operation for KL divergence
    return (p * (p / q).log()).sum(dim=1)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size,
                                           embedding=self.bert.embedding.token)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)

class BERTLM_Dual(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size,
                                           embedding=self.bert.embedding.token)

    def forward(self, x, segment_label, x2, segment_label2):
        x = self.bert(x, segment_label)
        x2 = self.bert(x2, segment_label2)
        print(x)
        print("X SHAPE", x.shape)
        x_mean = x.mean(dim=1)
        x2_mean = x2.mean(dim=1)

        x_mean_positive = x_mean + abs(x_mean.min())
        x2_mean_positive = x2_mean + abs(x2_mean.min())
        print("X MEAN SHAPE", x_mean_positive.shape)

        x_mean_normalized = x_mean_positive / x_mean_positive.sum(dim=1, keepdim=True)
        x2_mean_normalized = x2_mean_positive / x2_mean_positive.sum(dim=1, keepdim=True)

        loss_kd = js_divergence(x_mean_normalized, x2_mean_normalized).mean()
        print("After X shape", x.shape)
        return self.next_sentence(x), self.mask_lm(x), self.next_sentence(x2), self.mask_lm(x2), loss_kd

class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]).tanh())


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size, embedding=None):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        if embedding is not None:
            self.linear.weight.data = embedding.weight.data
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
