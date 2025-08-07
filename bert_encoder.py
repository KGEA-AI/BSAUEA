import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERTNameEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', output_dim=300):
        super(BERTNameEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, names: list[str]):
        tokens = self.tokenizer(names, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            bert_output = self.bert(**tokens).last_hidden_state[:, 0, :]  # CLS token
        return self.mlp(bert_output)
