import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import torch
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("medium")  # Modelo Whisper para melhor precisão
tts = TextToSpeechService(device="cuda" if torch.cuda.is_available() else "cpu")

template = """
Você é um assistente de IA chamado Jarvis.
Sempre me chame de "Meu querido" e fale apenas em português. A seguir está a pergunta.

Histórico da conversa:
{history}

Pergunta do usuário: {input}

Sua resposta:
"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

# Configuração do modelo de IA para geração de resposta
cadeia_conversacao = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistente:"),
    llm=Ollama(model="mistral", base_url="http://localhost:11434"),
)


def gravar_audio(evento_parar, fila_dados):
    """
    Captura áudio do microfone do usuário e adiciona à fila para processamento.

    Args:
        evento_parar (threading.Event): Sinaliza quando a gravação deve ser interrompida.
        fila_dados (queue.Queue): Fila onde os dados de áudio gravados serão armazenados.
    """
    def callback(indata, frames, tempo, status):
        if status:
            console.print(status)
        fila_dados.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not evento_parar.is_set():
            time.sleep(0.1)


def transcrever(audio_np: np.ndarray) -> str:
    """
    Transcreve o áudio utilizando o modelo Whisper.

    Args:
        audio_np (numpy.ndarray): Dados de áudio a serem transcritos.

    Returns:
        str: Texto transcrito.
    """
    resultado = stt.transcribe(audio_np, fp16=False, language="pt")  # Transcrição em PT-BR
    texto = resultado["text"].strip()
    return texto


def gerar_resposta_ia(texto: str) -> str:
    """
    Gera uma resposta com base no texto fornecido usando o modelo de IA.

    Args:
        texto (str): Texto de entrada.

    Returns:
        str: Resposta gerada pela IA.
    """
    resposta = cadeia_conversacao.predict(input=texto)
    if resposta.startswith("Assistente:"):
        resposta = resposta[len("Assistente:") :].strip()
    return resposta


def reproduzir_audio(taxa_amostragem, dados_audio):
    """
    Reproduz o áudio gerado pela IA.

    Args:
        taxa_amostragem (int): Taxa de amostragem do áudio.
        dados_audio (numpy.ndarray): Dados de áudio a serem reproduzidos.
    """
    sd.play(dados_audio, taxa_amostragem)
    sd.wait()


if __name__ == "__main__":
    console.print("[cyan]Assistente iniciado! Pressione Ctrl+C para sair.")

    try:
        while True:
            console.input("[cyan]Pressione Enter para começar a gravar...")

            fila_dados = Queue()  
            evento_parar = threading.Event()
            thread_gravacao = threading.Thread(target=gravar_audio, args=(evento_parar, fila_dados))
            thread_gravacao.start()

            input()  # Aguarda o usuário pressionar Enter para interromper a gravação
            evento_parar.set()
            thread_gravacao.join()

            dados_audio = b"".join(list(fila_dados.queue))
            audio_np = np.frombuffer(dados_audio, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("🔊 Transcrevendo...", spinner="earth"):
                    inicio_tempo = time.time()
                    texto = transcrever(audio_np)
                    console.print(f"Tempo para transcrever: {time.time() - inicio_tempo:.2f} segundos")
                console.print(f"[yellow]Você disse: {texto}")

                with console.status("Gerando resposta...", spinner="earth"):
                    inicio_tempo = time.time()
                    resposta = gerar_resposta_ia(texto)
                    console.print(f"Tempo para gerar resposta: {time.time() - inicio_tempo:.2f} segundos")

                with console.status("Sintetizando áudio...", spinner="earth"):
                    inicio_tempo = time.time()
                    taxa_amostragem, dados_audio = tts.sintetizar(resposta)
                    console.print(f"Tempo para gerar áudio: {time.time() - inicio_tempo:.2f} segundos")

                console.print(f"[cyan]Jarvis: {resposta}")
                reproduzir_audio(taxa_amostragem, dados_audio)
            else:
                console.print("[red]Nenhum áudio capturado. Verifique o microfone.")

    except KeyboardInterrupt:
        console.print("\n[red]Encerrando assistente...")

    console.print("[blue]Sessão finalizada.")
