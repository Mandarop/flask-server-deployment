from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

# ——— FastAPI setup ———
app = FastAPI()

# ——— Load tokenizer and MuRIL encoder ———
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
muril_model = AutoModel.from_pretrained("google/muril-base-cased")
muril_model.eval()

# ——— Define your classifier ———
class MultiLabelNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# ——— Symptom labels for Anxiety (1-7) and Depression (1-9) ———
anxiety_labels = [
    "Symptom 10", "Symptom 11", "Symptom 12", "Symptom 13", 
    "Symptom 14", "Symptom 15", "Symptom 16"
]

depression_labels = [
    "Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4", 
    "Symptom 5", "Symptom 6", "Symptom 7", "Symptom 8", "Symptom 9"
]

# ——— Request schema ———
class TextRequest(BaseModel):
    text: str

# ——— Prediction logic ———
def predict_symptoms(text: str, model: nn.Module, labels: list, threshold: float = 0.3):
    # tokenize + encode
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        cls_repr = muril_model(**inputs).last_hidden_state[:, 0, :]
        logits = model(cls_repr)               # shape [1, output_dim]
        probs = torch.sigmoid(logits).squeeze(0)  # shape [output_dim]

    # apply threshold
    mask = (probs > threshold).tolist()
    predicted = [lab for lab, keep in zip(labels, mask) if keep]

    return logits.squeeze(0).tolist(), probs.tolist(), predicted

# ——— API endpoint for Anxiety Prediction ———
@app.post("/predict/anxiety/")
async def predict_anxiety(req: TextRequest):
    # Load the anxiety model inside the request function
    anxiety_model = MultiLabelNet(input_dim=768, output_dim=7)
    anxiety_model.load_state_dict(torch.load("anxiety.pth", map_location="cpu"))
    anxiety_model.eval()

    logits, probs, predicted = predict_symptoms(req.text, anxiety_model, anxiety_labels, threshold=0.3)
    
    # Release the model from memory after use
    del anxiety_model
    
    return {
        "predicted_symptoms": predicted
    }

# ——— API endpoint for Depression Prediction ———
@app.post("/predict/depression/")
async def predict_depression(req: TextRequest):
    # Load the depression model inside the request function
    depression_model = MultiLabelNet(input_dim=768, output_dim=9)
    depression_model.load_state_dict(torch.load("depression.pth", map_location="cpu"))
    depression_model.eval()

    logits, probs, predicted = predict_symptoms(req.text, depression_model, depression_labels, threshold=0.3)
    
    # Release the model from memory after use
    del depression_model
    
    return {
        "predicted_symptoms": predicted
    }

# ——— Run with Uvicorn ———
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render assigns the PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
