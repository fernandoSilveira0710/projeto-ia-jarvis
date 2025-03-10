import nltk
import nltk.data  # Adicione esta linha
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="La entrada é uma lista de strings, mas o modelo espera uma lista de listas de strings.",
)


class TextToSpeechService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Inicializa o serviço de síntese de fala.

        Args:
            device (str, opicional): O dispositivo a ser usado para o modelo, "cuda" se um GPU estiver disponível ou "cpu".
            Defaults to "cuda" se disponível, senão "cpu".
        """
        self.device = "cuda"
        self.processor = self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/pt_speaker_3"):
        """
        Sintetiza áudio a partir do texto fornecido usando o voice preset especificado.

        Args:
            text (str): O texto a ser sintetizado.
            voice_preset (str, opcional): O conjunto de voz a ser usado para a síntese.
            Defaults to "v2/pt_speaker_3".

        Returns:
            tuple: Uma tupla contendo a taxa de amostragem e o array de áudio sintetizado.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/pt_speaker_3"):
        """
       Sintetiza áudio a partir do texto fornecido usando o voice preset especificado.

        Args:
            text (str): O texto a ser sintetizado.
            voice_preset (str, opcional): O conjunto de voz a ser usado para a síntese.
            Defaults to "v2/pt_speaker_3".

        Returns:
            tuple: Uma tupla contendo a taxa de amostragem e o array de áudio sintetizado.
        """
        pieces = []
        nltk.download("punkt")

        # Corrigindo o erro forçando o idioma correto
        tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()  # Usa o tokenizer padrão
        sentences = tokenizer.tokenize(text)  # Agora não tenta carregar `punkt_tab`

        silence = np.zeros(int(1.0 * self.model.generation_config.sample_rate))

        text = " ".join(sentences)  # Junta todas as frases antes de processar
        sample_rate, audio_array = self.synthesize(text, voice_preset)
        pieces.append(audio_array)  # Adiciona tudo de uma vez

        return self.model.generation_config.sample_rate, np.concatenate(pieces)
