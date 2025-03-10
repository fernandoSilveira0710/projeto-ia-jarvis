# 📖 Documentação do Assistente de Voz com Transcrição e Síntese de Fala

## 📌 Visão Geral

Este projeto é um assistente de voz que:
1. **Transcreve áudio para texto** usando **Whisper**.
2. **Gera respostas** usando **Ollama** (LLMs locais).
3. **Converte texto para fala (TTS)** usando **Silero TTS**.

O assistente roda **100% localmente**, sem necessidade de internet, garantindo **privacidade** e **baixo custo**.

---

## 📁 Estrutura do Projeto

```
📂 JARVIS
│── 📄 app.py               # Código principal do assistente de voz
│── 📄 tts.py               # Serviço de síntese de fala (TTS) com Silero
│── 📄 tts_old_by_bark.py   # Código alternativo de TTS usando Bark (não utilizado)
│── 📄 requirements.txt      # Lista de dependências do projeto
│── 📄 README.md            # Documentação do projeto
│── 📄 .gitignore           # Arquivos ignorados pelo Git
│── 📄 pyproject.toml       # Configurações do projeto (para futuras melhorias)
│── 📄 Makefile             # (Opcional) Comandos para automação de tarefas
│── 📄 LICENSE              # Licença do projeto
```

---

## 🛠️ Configuração do Ambiente

### 🔹 1. Instale o **Python**
O projeto foi testado com **Python 3.10+**, então baixe a versão mais recente no site oficial:

🔗 [Download Python](https://www.python.org/downloads/)

### 🔹 2. Instale o **Whisper**, **Ollama** e **Silero TTS**
Certifique-se de que tem o [**Ollama**](https://ollama.com/) instalado:

```bash
# Para Linux/MacOS
curl -fsSL https://ollama.com/install.sh | sh

# Para Windows (via PowerShell)
iwr https://ollama.com/install.ps1 -useb | iex
```

Depois, instale as dependências do Python:

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Usar

### 🔹 1. Ativar o ambiente virtual (se aplicável)

Se estiver usando **ambiente virtual**, ative-o antes de rodar o código:

```bash
# Windows
env\Scripts\activate

# Linux/MacOS
source env/bin/activate
```

### 🔹 2. Iniciar o Assistente

Agora, basta rodar:

```bash
python app.py
```

---

## 🏗️ Explicação dos Arquivos

### 📜 **1. `app.py` (Código principal)**
- Captura **áudio** do microfone.
- **Transcreve** usando **Whisper**.
- Envia para o **modelo de IA** (Ollama) para gerar respostas.
- **Converte a resposta** em voz com **Silero TTS**.

### 🎙️ **2. `tts.py` (Síntese de fala - TTS)**
- Utiliza o **Silero TTS** para converter **texto em áudio**.
- Usa o modelo **`multi_v2`** (mais versátil e natural).
- Permite **alterar locutor e idioma**.

### 🗣️ **3. `tts_old_by_bark.py` (Antiga implementação)**
- Usa o **Bark** como alternativa para TTS.
- **Não está em uso**, pois o **Silero** tem melhor desempenho.

### 📦 **4. `requirements.txt` (Dependências do projeto)**
Lista todas as bibliotecas usadas, incluindo:
```txt
torch
torchaudio
numpy
sounddevice
nltk
transformers
```
Instale com:
```bash
pip install -r requirements.txt
```

### 📃 **5. `README.md` (Esta documentação)**
Explica como instalar e rodar o assistente.

---

## 🔄 Alteração do Locutor ou Idioma (Silero TTS)

Se quiser **mudar a voz**, altere **`tts.py`**, mudando estas linhas:

```python
self.language = 'es'  # Troque para 'en' (inglês), 'fr' (francês), etc.
self.model_id = 'v3_es'  # Modelo compatível com o idioma
self.speaker = 'es_0'  # Locutor disponível no idioma
```

### 🔹 **Locutores disponíveis:**
Para **espanhol (`v3_es`)**, você pode escolher:
- `'es_0'`
- `'es_1'`
- `'es_2'`
- `'random'` (aleatório)

Se quiser **inglês (`v3_en`)**, use:
- `'en_0'`
- `'en_1'`
- `'en_2'`
- `'random'`

Se houver **suporte para português no futuro**, basta ajustar essas variáveis.

---

## 🔧 Problemas Comuns e Soluções

| Erro | Solução |
|------|---------|
| `ModuleNotFoundError: No module named 'omegaconf'` | Rode `pip install omegaconf` |
| `ValueError: Speaker not in the supported list` | Verifique se o locutor está correto no **`tts.py`** |
| `Torch not compiled with CUDA enabled` | Rode `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| `Connection refused on Ollama API` | Certifique-se de que o **Ollama** está rodando (`ollama run llama2`) |

---

## 📌 Próximos Passos

- Melhorar a **fluidez da voz** no Silero.
- Adicionar **suporte a português** quando disponível no Silero.
- Criar um **frontend** para facilitar o uso.

---

Caso tenha dúvidas ou sugestões, sinta-se à vontade para contribuir! 🚀