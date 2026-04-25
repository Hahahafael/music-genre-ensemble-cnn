import os
import json
import librosa
import numpy as np
from src.config import Config

class FeatureExtractor:
    def __init__(self):
        self.data = {
            "mfcc": {"features": [], "labels": []},
            "mel": {"features": [], "labels": []},
            "chroma": {"features": [], "labels": []}
        }

    def extract(self, num_segments=10):
        samples_per_segment = int(Config.SAMPLES_PER_TRACK / num_segments)

        for i, genre in enumerate(Config.GENRES):
            genre_path = Config.DATA_PATH / genre
            if not genre_path.exists(): continue
            
            print(f"Extraindo: {genre}...")
            for f in os.listdir(genre_path):
                if not f.endswith(".wav"): continue
                
                try:
                    signal, _ = librosa.load(str(genre_path / f), sr=Config.SAMPLE_RATE)
                    for d in range(num_segments):
                        segment = signal[samples_per_segment*d : samples_per_segment*(d+1)]
                        if len(segment) < samples_per_segment: continue

                        # Extrair as 3 visões
                        self._process_mfcc(segment, i)
                        self._process_mel(segment, i)
                        self._process_chroma(segment, i)
                except Exception as e:
                    print(f"Erro no arquivo {f}: {e}")

        self._save_all()

    def _process_mfcc(self, segment, label):
        feat = librosa.feature.mfcc(y=segment, sr=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC).T
        if len(feat) == 130:
            self.data["mfcc"]["features"].append(feat.tolist())
            self.data["mfcc"]["labels"].append(label)

    def _process_mel(self, segment, label):
        mel = librosa.feature.melspectrogram(y=segment, sr=Config.SAMPLE_RATE, n_mels=Config.N_MELS)
        feat = librosa.power_to_db(mel, ref=np.max).T
        if len(feat) == 130:
            self.data["mel"]["features"].append(feat.tolist())
            self.data["mel"]["labels"].append(label)

    def _process_chroma(self, segment, label):
        feat = librosa.feature.chroma_stft(y=segment, sr=Config.SAMPLE_RATE, n_chroma=Config.N_CHROMA).T
        if len(feat) == 130:
            self.data["chroma"]["features"].append(feat.tolist())
            self.data["chroma"]["labels"].append(label)

    def _save_all(self):
        for key in self.data:
            path = Config.PROCESSED_DIR / f"{key}.json"
            with open(path, "w") as f:
                json.dump(self.data[key], f)
            print(f"Salvo: {path}")

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.extract()