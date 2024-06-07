import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)


class UniXcoderModelConfig:
    batch_size = 32
    output_size = 6
    hidden_dim = 768
    # training params
    lr = 2e-6
    epochs = 200
    use_cuda = USE_CUDA
    unixcoder_path = './unixcoder-base'
    save_path = 'saved/unixcoder'


class UniXcoderClassifier(nn.Module):
    def __init__(self, codebertpath, hidden_dim, output_dim):
        super(UniXcoderClassifier, self).__init__()
        self.unixcoder = RobertaModel.from_pretrained(codebertpath)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_ids = x[:, 0, :]
        attention_masks = x[:, 1, :]
        pooled = self.unixcoder(input_ids=input_ids, attention_mask=attention_masks)[0][:, 0, :]
        out = self.fc(pooled)
        out = self.sigmoid(out)
        return out
