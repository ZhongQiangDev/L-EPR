import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)


class GraphCodeBERTModelConfig:
    batch_size = 32
    output_size = 6
    hidden_dim = 768
    # training params
    lr = 2e-6
    epochs = 200
    use_cuda = USE_CUDA
    graphcodebert_path = './graphcodebert-base'
    save_path = 'saved/graphcodebert'


class GraphCodeBERTClassifier(nn.Module):
    def __init__(self, graphcodebertpath, hidden_dim, output_dim):
        super(GraphCodeBERTClassifier, self).__init__()
        self.graphcodebert = RobertaModel.from_pretrained(graphcodebertpath)
        for param in self.graphcodebert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, position_idx, attn_mask):
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.graphcodebert.embeddings.word_embeddings(input_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        pooled = self.graphcodebert(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0][:, 0, :]
        out = self.fc(pooled)
        out = self.sigmoid(out)
        return out
