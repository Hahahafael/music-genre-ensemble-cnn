import torch
from pathlib import Path

class Config:
    # Caminhos Base
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "genres_original"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    
    # Parâmetros de Áudio
    SAMPLE_RATE = 22050
    DURATION = 30
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    
    # Parâmetros dos 3 Especialistas
    N_MFCC = 13
    N_MELS = 64
    N_CHROMA = 12
    
    # Hiperparâmetros
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mapeamento de Gêneros
    GENRES = ["blues", "classical", "country", "disco", "hiphop", 
              "jazz", "metal", "pop", "reggae", "rock"]

# Criar pastas automaticamente
Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)