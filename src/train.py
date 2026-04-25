import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.config import Config
from src.model import MFCC_Model, Mel_Model, Chroma_Model

def train_one_model(model_class, json_name):
    print(f"\n--- Treinando Especialista: {json_name.upper()} ---")
    
    with open(Config.PROCESSED_DIR / f"{json_name}.json", "r") as f:
        data = json.load(f)
    
    X = torch.tensor(np.array(data["features"]), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.array(data["labels"]), dtype=torch.long)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)

    model = model_class().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(Config.EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            optimizer.zero_grad(); model(inputs); # ... (Lógica de treino padrão)
            loss = criterion(model(inputs), targets)
            loss.backward(); optimizer.step()

        model.eval()
        correct = sum(p.max(1)[1].eq(t).sum().item() for p, t in [(model(i.to(Config.DEVICE)), t.to(Config.DEVICE)) for i, t in val_loader])
        acc = 100. * correct / len(X_val)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Config.MODELS_DIR / f"{json_name}.pth")
        
        if (epoch+1) % 10 == 0: print(f"Época {epoch+1} | Acc: {acc:.2f}%")

if __name__ == "__main__":
    train_one_model(MFCC_Model, "mfcc")
    train_one_model(Mel_Model, "mel")
    train_one_model(Chroma_Model, "chroma")