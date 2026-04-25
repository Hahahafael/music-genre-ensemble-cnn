# 🎵 Music Genre Classifier: Ensemble CNN

Este projeto implementa um classificador de gêneros musicais utilizando um sistema de **Ensemble Learning** com Redes Neurais Convolucionais (CNN). 

## 🚀 Como Funciona?
O diferencial deste projeto é a utilização de **três especialistas** (modelos independentes) que analisam diferentes características do áudio para decidir o gênero final:

1.  **MFCC Specialist**: Foca no timbre e características percussivas.
2.  **Mel Spectrogram Specialist**: Foca na textura sonora e frequências (percepção humana).
3.  **Chroma Specialist**: Foca no conteúdo harmônico e notas musicais.

O resultado final é obtido através de uma média ponderada das probabilidades de cada especialista, garantindo maior robustez contra falsos positivos.

## 📊 Performance
- **Mel Specialist**: ~77% Acc
- **MFCC Specialist**: ~72% Acc
- **Chroma Specialist**: ~45% Acc
- **Ensemble Final**: Maior estabilidade em gêneros complexos (ex: Metal vs Rock).


## 🛠️ Tecnologias
- **Python 3.14**
- **PyTorch** (Deep Learning)
- **Librosa** (Processamento de Áudio)
- **Matplotlib/Seaborn** (Visualização)

## 🏃 Como Rodar
1. Instale as dependências:
   `pip install -r requirements.txt`

2. Coloque o dataset GTZAN em `data/raw/genres_original/`.

3. Extraia as features:
   `python -m src.extrator`

4. Treine os modelos:
   `python -m src.train`

5. Teste uma música:
   `python -m src.ensemble`

---
Desenvolvido por Rafael durante estudos de Deep Learning aplicados a áudio.