from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import spacy

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = FastAPI(
    title="AI Spam Detection API",
    description="spaCy + BiLSTM + FastAPI",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load spaCy
nlp = spacy.load("en_core_web_sm")


# Load model
model = load_model("spam_model.keras")


# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


MAX_LENGTH = 100


# -------------------------
# spaCy preprocessing
# -------------------------

def preprocess(text):
    doc=nlp(text)
    tokens=[]
    for token in doc:
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        if token.is_space:
            continue
        tokens.append(token.lemma_.lower())
    return " ".join(tokens)


# -------------------------
# Request model
# -------------------------

class Message(BaseModel):
    message: str


# -------------------------
# Home
# -------------------------

@app.get("/")
def home():

    return {
        "message": "AI Spam Detection API",
        "technology": [
            "spaCy",
            "NLP",
            "BiLSTM",
            "FastAPI"
        ]
    }


# -------------------------
# Prediction
# -------------------------

@app.post("/predict")
def predict(data: Message):

    original = data.message

    # spaCy NLP
    processed = preprocess(original)

    # Tokenize
    sequence = tokenizer.texts_to_sequences(
        [processed]
    )

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    # RNN prediction
    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    probability = float(prediction)

    if probability >= 0.5:

        result = "SPAM"
        confidence = probability

    else:

        result = "HAM"
        confidence = 1 - probability


    return {
        "original_message": original,
        "processed_message": processed,
        "prediction": result,
        "confidence": round(confidence, 4)
    }