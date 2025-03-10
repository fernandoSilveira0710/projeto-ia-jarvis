import torch

class TextToSpeechService:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.language = 'es'  # ✅ Escolha um idioma compatível (ex: 'en' para inglês, 'es' para espanhol)
        self.model_id = 'v3_es'  # ✅ Escolha um modelo compatível com o idioma
        self.speaker = 'es_0'  # ✅ Escolha um locutor válido dentro do idioma

        # Carregar modelo do Silero TTS
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=self.language,
            speaker=self.model_id
        )

        self.model.to(self.device)
        self.sample_rate = 48000  # ✅ Definindo a taxa de amostragem padrão

    def sintetizar(self, text: str):
        print("🔊 Gerando áudio com Silero TTS...")
        audio = self.model.apply_tts(
            text=text,  # ✅ "text" e não "texts"
            speaker=self.speaker,  # ✅ Locutor válido dentro do idioma escolhido
            sample_rate=self.sample_rate  # ✅ Taxa de amostragem padrão
        )
        return self.sample_rate, audio
