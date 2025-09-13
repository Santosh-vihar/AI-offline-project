import pyttsx3
import queue
import sounddevice as sd
import vosk
import json
from pathlib import Path
import sys

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

# ----------------------------
# Text to Speech (TTS)
# ----------------------------
def speak(text: str):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ----------------------------
# Speech-to-Text (STT) Initialization
# ----------------------------
def find_model_path(base_path: str) -> Path:
    base = Path(base_path)
    if not base.exists():
        print(f"❌ Path does not exist: {base}")
        sys.exit(1)

    required = ["am", "conf", "graph"]
    if all((base / r).exists() for r in required):
        return base

    for sub in base.iterdir():
        if sub.is_dir() and all((sub / r).exists() for r in required):
            return sub

    print("❌ Could not find valid Vosk model folder inside " + str(base))
    sys.exit(1)

BASE_MODEL_PATH = r"C:\New folder\assistant\vosk1\vosk-model-small-en-in-0.4"
model_path = find_model_path(BASE_MODEL_PATH)
print(f"✅ Using Vosk model from: {model_path}")

model = vosk.Model(str(model_path))

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

def listen() -> str:
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, 16000)
        print("🎤 Speak now... (say 'exit' to quit)")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result["text"]:
                    print("🗣 You said:", result["text"])
                    return result["text"].lower()

# ----------------------------
# Load FAQ File
# ----------------------------
def load_faq(filepath="faq.txt"):
    faq_dict = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            if "|" in line:
                question, answer = line.strip().split("|", 1)
                faq_dict[question.lower()] = answer
    return faq_dict

# ----------------------------
# Setup Notes QA System
# ----------------------------
def setup_notes_qa(notes_file="data_structures.txt"):
    with open(notes_file, "r", encoding="utf-8") as f:
        notes = f.read()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(notes)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="gemma:2b"),
        retriever=retriever
    )
    return qa

# ----------------------------
# Get Answer Logic
# ----------------------------
def get_answer(question, qa, faq_dict):
    question_key = question.lower().strip()

    # Check FAQ
    if question_key in faq_dict:
        return {"answer": faq_dict[question_key], "source": "FAQ file"}

    # Semantic search in notes
    retriever = qa.retriever
    relevant_docs = retriever.get_relevant_documents(question)

    if relevant_docs and any(doc.page_content.strip() for doc in relevant_docs):
        result = qa.run(question)

        # If unhelpful response, fallback
        if "does not provide any information" in result.lower():
            direct_llm = OllamaLLM(model="gemma:2b")
            result = direct_llm.predict(question)
            return {"answer": result, "source": "Direct LLM (fallback)"}
        else:
            return {"answer": result, "source": "Notes retrieval + LLM"}

    # Fallback to direct LLM
    direct_llm = OllamaLLM(model="gemma:2b")
    result = direct_llm.predict(question)
    return {"answer": result, "source": "Direct LLM (fallback)"}

# ----------------------------
# Main Loop
# ----------------------------
def main():
    faq_dict = load_faq("faq.txt")
    qa = setup_notes_qa()

    speak("Hello, I am ready. Ask me something from your notes, or say exit to quit.")

    while True:
        user_question = listen()

        if "exit" in user_question:
            print("👋 Exiting...")
            speak("Goodbye!")
            break

        if user_question.strip() == "":
            continue

        try:
            print("🤖 Thinking...")
            speak("I'm thinking, please wait for a while.")

            response = get_answer(user_question, qa, faq_dict)

            answer = response["answer"]
            source = response["source"]

        except Exception as e:
            answer = f"⚠ Error contacting AI model: {e}"
            source = "Error"

        print(f"🤖 AI (from {source}): {answer}")
        speak(answer)

if _name_ == "_main_":
    main()
