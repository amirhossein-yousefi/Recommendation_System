from torch import nn
import torch


class MFBlock(nn.Module):
    def __init__(self, max_users, max_items, num_emb, dropout_p=0.5):
        super(MFBlock, self).__init__()
        self.max_users = max_users
        self.max_items = max_items
        self.dropout_p = dropout_p
        self.num_emb = num_emb
        self.user_embeddings = nn.Embedding(max_users, num_emb)
        self.item_embeddings = nn.Embedding(max_items, num_emb)
        self.dropout_user = nn.Dropout(dropout_p)
        self.dropout_item = nn.Dropout(dropout_p)
        self.dense_user = nn.Linear(num_emb, num_emb)
        self.dense_item = nn.Linear(num_emb, num_emb)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, users, items):
        a = self.user_embeddings(users)
        a = self.dense_user(a)
        a = self.relu_1(a)

        b = self.item_embeddings(items)
        b = self.dense_item(b)
        b = self.relu_2(b)

        predictions = self.dropout_user(a) * self.dropout_item(b)
        predictions = torch.sum(predictions, dim=1)

        return predictions
