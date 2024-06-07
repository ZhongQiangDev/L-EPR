import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)


class BERTModelConfig:
    batch_size = 32
    output_size = 6
    hidden_dim = 768
    # training params
    lr = 2e-6
    epochs = 100
    use_cuda = USE_CUDA
    # bert_path = './bert-base-cased'
    bert_path = './bert-base-uncased'
    save_path = 'saved/bert-un'


class BERTClassifier(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_dim):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_ids = x[:, 0, :]
        attention_masks = x[:, 1, :]
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_masks)[0][:, 0, :]
        out = self.fc(pooled)
        out = self.sigmoid(out)
        return out
