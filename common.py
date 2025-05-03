from transformers import AutoTokenizer, AutoModel
import torch, torch.nn as nn

# ——— Shared tokenizer + encoder ———
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
muril_model = AutoModel.from_pretrained("google/muril-base-cased")
muril_model.eval()

# ——— Shared model class ———
class MultiLabelNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# ——— Shared preprocessing & embed function ———
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        return muril_model(**inputs).last_hidden_state[:,0,:]
