import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from src.config import Config
from src.model import GenreClassifier

def plot_confusion_matrix():
    print("Carregando dados para avaliação...")
    with open(Config.PROCESSED_DATA_PATH, "r") as fp:
        data = json.load(fp)
    
    # Avaliar em todo o dataset para gerar uma matriz visual completa
    X = torch.tensor(np.array(data["mfcc"]), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.array(data["labels"]), dtype=torch.long)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenreClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []

    print("Gerando predições. Isso pode levar alguns segundos...")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    genres = ["blues", "classical", "country", "disco", "hiphop", 
              "jazz", "metal", "pop", "reggae", "rock"]
    
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.title('Matriz de Confusão - Classificador de Gêneros Musicais')
    plt.ylabel('Gênero Real')
    plt.xlabel('Gênero Predito da IA')
    plt.tight_layout()
    plt.savefig('matriz_de_confusao.png')
    print("Sucesso! O gráfico foi salvo como 'matriz_de_confusao.png' na pasta do projeto.")

if __name__ == "__main__":
    plot_confusion_matrix()