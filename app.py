import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import torch
import os
import pyautogui
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("medium")  # Melhor precisão
tts = TextToSpeechService(device="cuda" if torch.cuda.is_available() else "cpu")

template = """
Você é um assistente de IA chamado Jarvis.  
Me chame de Meu querido, fale sempre em português. Segue a pergunta:

Histórico da conversa:
{history}

Pergunta do usuário: {input}

Sua resposta:
"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistente:"),
    llm=Ollama(model="mistral", base_url="http://localhost:11434"),
)

# Dicionário de comandos
COMANDOS = {
    "abrir bloco de notas": "notepad.exe",
}

def executar_comando(comando):
    """
    Executa comandos do sistema operacional.
    """
    if comando in COMANDOS:
        os.system(COMANDOS[comando])  # Abre o aplicativo correspondente
        time.sleep(1)  # Espera o bloco de notas abrir
        return True
    return False


def record_audio(stop_event, data_queue):
    """
    Captura áudio do microfone e armazena na fila para processamento.

    Args:
        stop_event (threading.Event): Evento para parar a gravação.
        data_queue (queue.Queue): Fila onde os dados do áudio são armazenados.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcreve o áudio capturado usando Whisper.

    Args:
        audio_np (numpy.ndarray): Dados de áudio.

    Returns:
        str: Texto transcrito.
    """
    result = stt.transcribe(audio_np, fp16=False, language="pt")  # Forçar PT
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Gera resposta baseada no texto fornecido.

    Args:
        text (str): Entrada do usuário.

    Returns:
        str: Resposta gerada pelo LLM.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistente:"):
        response = response[len("Assistente:") :].strip()
    return response


def play_audio(sample_rate, audio_array):
    """
    Reproduz o áudio gerado.

    Args:
        sample_rate (int): Taxa de amostragem.
        audio_array (numpy.ndarray): Dados do áudio.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":
    console.print("[cyan]Assistente iniciado! Pressione Ctrl+C para sair.")

    try:
        while True:
            console.input("[cyan]Pressione Enter para começar a gravar...")

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()  # Aguarda o usuário pressionar Enter para parar a gravação
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("🔊 Transcrevendo...", spinner="earth"):
                    start_time = time.time()
                    text = transcribe(audio_np)
                    console.print(f"Tempo para transcrever: {time.time() - start_time:.2f} segundos")
                console.print(f"[yellow]Você disse: {text}")

                # 🔹 Verifica se há um comando especial
                if any(cmd in text.lower() for cmd in COMANDOS.keys()):
                    comando_detectado = next(cmd for cmd in COMANDOS.keys() if cmd in text.lower())
                    console.print(f"[green]Executando comando: {comando_detectado}...")
                    if executar_comando(comando_detectado):
                        time.sleep(1)  # Tempo para abrir o bloco de notas
                    continue  # Pula a geração de resposta normal

                with console.status("Gerando resposta...", spinner="earth"):
                    start_time = time.time()
                    response = get_llm_response(text)
                    console.print(f"Tempo para gerar resposta: {time.time() - start_time:.2f} segundos")

                with console.status("Sintetizando áudio...", spinner="earth"):
                    start_time = time.time()
                    sample_rate, audio_array = tts.sintetizar(response)
                    console.print(f"Tempo para gerar áudio: {time.time() - start_time:.2f} segundos")

                console.print(f"[cyan]Jarvis: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print("[red]Nenhum áudio capturado. Verifique o microfone.")

    except KeyboardInterrupt:
        console.print("\n[red]Saindo...")

    console.print("[blue]Sessão encerrada.")
