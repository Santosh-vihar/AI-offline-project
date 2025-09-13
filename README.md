# AI-offline-project

# Prism: Offline AI Voice Assistant

Offline AI assistant in Python that uses Vosk for speech-to-text, Langchain & Ollama LLM for local text search, and pyttsx3 for spoken answers.

## Features
- Offline voice recognition
- FAQ file and notes-based semantic search
- Local LLM for fallback responses
- Zero cloud dependence and private data

## Requirements
- Python 3.8+
- pyttsx3
- sounddevice
- vosk
- langchain
- langchain_community
- langchain_ollama
- chromadb

## Setup
1. Clone this repo and enter the directory:
2. Install dependencies:
3. Download Vosk model and extract to your local path. Update `BASE_MODEL_PATH` in source code.
4. Prepare your own `faq.txt` (format: question|answer), and `data_structures.txt` for notes.
5. Run:

## Usage
- Speak a question when prompted.
- Say “exit” to quit.
- Program will speak and print answer from FAQ, notes, or fallback to LLM as needed.

---

## Contributing
Submit PRs or open issues for improvements.

## License
MIT

For more details, see the official [GitHub Quickstart documentation][2][1].
