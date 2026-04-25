import torch
import librosa
import numpy as np
from src.config import Config
from src.model import MFCC_Model, Mel_Model, Chroma_Model

class MusicEnsemble:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carregar Mapeamento (assumindo a ordem padrão do GTZAN)
        self.mapping = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        
        # Inicializar e carregar os 3 modelos
        self.model_mfcc = self._load_model(MFCC_Model, "mfcc.pth")
        self.model_mel = self._load_model(Mel_Model, "mel.pth")
        self.model_chroma = self._load_model(Chroma_Model, "chroma.pth")

    def _load_model(self, model_class, filename):
        model = model_class().to(self.device)
        model.load_state_dict(torch.load(f"{Config.MODELS_DIR}/{filename}", map_location=self.device))
        model.eval()
        return model

    def predict(self, audio_path):
        # 1. Carregar áudio e extrair features
        signal, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        # Pegar um trecho do meio da música (mais representativo)
        mid = len(signal) // 2
        segment = signal[mid : mid + int(Config.SAMPLE_RATE * 3)]

        # Extrair as 3 "visões"
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=Config.N_MFCC).T
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=Config.N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max).T
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=Config.N_CHROMA).T

        # Converter para Tensores
        mfcc_t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        mel_t = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        chroma_t = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. Obter probabilidades (Softmax)
        with torch.no_grad():
            prob_mfcc = torch.softmax(self.model_mfcc(mfcc_t), dim=1)
            prob_mel = torch.softmax(self.model_mel(mel_t), dim=1)
            prob_chroma = torch.softmax(self.model_chroma(chroma_t), dim=1)

        # 3. Ensemble (Média das probabilidades)
        final_prob = (prob_mfcc + prob_mel + prob_chroma) / 3
        conf, class_idx = torch.max(final_prob, dim=1)

        # MUDANÇA AQUI: Vamos retornar um dicionário com os detalhes
        detalhes = {
            "mfcc": (self.mapping[torch.argmax(prob_mfcc).item()], torch.max(prob_mfcc).item()),
            "mel": (self.mapping[torch.argmax(prob_mel).item()], torch.max(prob_mel).item()),
            "chroma": (self.mapping[torch.argmax(prob_chroma).item()], torch.max(prob_chroma).item())
        }

        return self.mapping[class_idx.item()], conf.item(), detalhes

if __name__ == "__main__":
    test_file = "test_audios/allmyloving.wav"
    ensemble = MusicEnsemble()
    genre, confidence, detalhes = ensemble.predict(test_file)

    print(f"\n" + "="*30)
    print(f"   VOTOS DOS ESPECIALISTAS")
    print(f"="*30)
    for modelo, (pred, conf_ind) in detalhes.items():
        print(f"-> {modelo.upper()}: {pred.upper()} ({conf_ind*100:.1f}%)")
    
    print(f"="*30)
    print(f"RESULTADO DO COMITÊ: {genre.upper()}")
    print(f"CONFIANÇA MÉDIA: {confidence*100:.2f}%")
    print(f"="*30)