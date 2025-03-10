import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import torch
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
Me chame de 'Meu querido', fale sempre em português. Segue a pergunta:

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

# Controle do sistema
parar_conversa = threading.Event()  # Para a fala
executando = True  # Controla a execução do sistema


def record_audio(stop_event, data_queue):
    """ Captura áudio do microfone e adiciona na fila. """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """ Transcreve áudio usando Whisper. """
    result = stt.transcribe(audio_np, fp16=False, language="pt")
    return result["text"].strip()


def get_llm_response(text: str) -> str:
    """ Gera resposta para um texto usando Llama-2. """
    response = chain.predict(input=text)
    if response.startswith("Assistente:"):
        response = response[len("Assistente:") :].strip()
    return response


def play_audio(sample_rate, audio_array):
    """ Toca um áudio gerado pelo TTS e escuta por comandos de parada. """
    parar_conversa.clear()
    sd.play(audio_array, sample_rate)

    def monitorar_comandos():
        """ Escuta por comandos enquanto o áudio está sendo reproduzido. """
        while sd.get_stream().active:
            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

            time.sleep(1)  # Tempo curto para capturar comandos

            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                text = transcribe(audio_np).lower()
                console.print(f"[yellow]Você disse: {text}[/yellow]")

                if "parar" in text or "encerrar" in text:
                    console.print("[red]Parando fala imediatamente...[/red]")
                    parar_conversa.set()
                    sd.stop()
                    return

    # Rodar a monitoração em uma thread paralela
    monitor_thread = threading.Thread(target=monitorar_comandos)
    monitor_thread.start()

    while sd.get_stream().active:
        if parar_conversa.is_set():
            sd.stop()
            break
        time.sleep(0.1)

    monitor_thread.join()


def iniciar_bloco_de_notas():
    """ Abre o bloco de notas e insere tudo o que for falado. """
    console.print("[green]Abrindo o bloco de notas...[/green]")
    pyautogui.hotkey("win", "r")
    time.sleep(1)
    pyautogui.write("notepad")
    pyautogui.press("enter")
    time.sleep(1)

    console.print("[yellow]Falando para o bloco de notas...[/yellow]")

    while True:
        data_queue = Queue()
        stop_event = threading.Event()
        recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
        recording_thread.start()

        time.sleep(2)  # Escuta por 2 segundos

        stop_event.set()
        recording_thread.join()

        audio_data = b"".join(list(data_queue.queue))
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if audio_np.size > 0:
            text = transcribe(audio_np)
            pyautogui.write(text + "\n")
        else:
            console.print("[red]Nenhum áudio detectado.[/red]")
            break


if __name__ == "__main__":
    console.print("[cyan]Jarvis iniciado! Ouvindo...[/cyan]")

    try:
        while executando:
            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

            tempo_espera = time.time()  # Marca o tempo de espera inicial

            while True:
                time.sleep(1)  # Continua ouvindo

                stop_event.set()
                recording_thread.join()

                audio_data = b"".join(list(data_queue.queue))
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                if audio_np.size > 0:
                    tempo_espera = time.time()  # Reinicia o tempo de espera
                    text = transcribe(audio_np).lower()
                    console.print(f"[yellow]Você disse: {text}[/yellow]")

                    if "encerrar" in text:
                        console.print("[red]Encerrando o sistema...[/red]")
                        executando = False
                        break

                    if "teste" in text:
                        response = "Posso ajudar, meu querido?"
                        console.print(f"[cyan]Jarvis: {response}[/cyan]")
                        sample_rate, audio_array = tts.sintetizar(response)
                        play_audio(sample_rate, audio_array)

                        # Iniciar nova gravação após a resposta
                        data_queue = Queue()
                        stop_event = threading.Event()
                        recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
                        recording_thread.start()

                        tempo_espera = time.time()  # Marca o tempo de espera

                        while True:
                            time.sleep(1)  # Continua ouvindo

                            stop_event.set()
                            recording_thread.join()

                            audio_data = b"".join(list(data_queue.queue))
                            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                            if audio_np.size > 0:
                                tempo_espera = time.time()  # Reinicia o tempo de espera
                                text = transcribe(audio_np).lower()
                                console.print(f"[yellow]Você disse: {text}[/yellow]")

                                if "não precisa" in text:
                                    console.print("[blue]Voltando ao estado de escuta...[/blue]")
                                    break

                                if "parar" in text or "encerrar" in text:
                                    console.print("[red]Parando conversa e voltando à escuta...[/red]")
                                    parar_conversa.set()
                                    break

                                if "bloco de notas" in text:
                                    iniciar_bloco_de_notas()
                                    break

                                response = get_llm_response(text)
                                console.print(f"[cyan]Jarvis: {response}[/cyan]")
                                sample_rate, audio_array = tts.sintetizar(response)
                                play_audio(sample_rate, audio_array)

                            else:
                                if time.time() - tempo_espera > 3:  # Se 3 segundos se passaram sem fala
                                    response = "Bem, como não tem dúvidas, irei descansar, chefe!"
                                    console.print(f"[cyan]Jarvis: {response}[/cyan]")
                                    sample_rate, audio_array = tts.sintetizar(response)
                                    play_audio(sample_rate, audio_array)
                                    break

                            stop_event = threading.Event()
                            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
                            recording_thread.start()

                        if time.time() - tempo_espera > 3:  # Se 3 segundos se passaram sem fala
                            response = "Bem, como não tem dúvidas, irei descansar, chefe!"
                            console.print(f"[cyan]Jarvis: {response}[/cyan]")
                            sample_rate, audio_array = tts.sintetizar(response)
                            play_audio(sample_rate, audio_array)
                            break

                else:
                    if time.time() - tempo_espera > 3:  # Se 3 segundos se passaram sem fala
                        response = "Bem, como não tem dúvidas, irei descansar, chefe!"
                        console.print(f"[cyan]Jarvis: {response}[/cyan]")
                        sample_rate, audio_array = tts.sintetizar(response)
                        play_audio(sample_rate, audio_array)
                        break

                stop_event = threading.Event()
                recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
                recording_thread.start()

    except KeyboardInterrupt:
        console.print("\n[red]Saindo...[/red]")

    console.print("[blue]Sessão encerrada.[/blue]")