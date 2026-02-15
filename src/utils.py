import os
import re
import joblib
import torch
import numpy as np
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT.parent / "models"

DISTILBERT_DIR = MODELS_DIR / "distilbert"
FALLBACK_PATH = MODELS_DIR / "model_fallback.joblib"
META_PATH = MODELS_DIR / "model_metadata.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTECTED_SLURS = {"nigga", "nigger"}
INSULTS = {"whore", "slut", "asshat", "twat"}
PROFANITY = {"fuck", "shit", "bitch", "dick", "asshole", "bastard", "cunt", "motherfucker", "douche"}
SEXUAL_TERMS = {"boobies", "boobs", "tits", "pussy", "cock", "penis", "vagina", "blowjob"}
VIOLENCE = {"kill", "die", "hurt", "stab", "shoot", "beat", "destroy"}

NEGATION_RE = re.compile(r"\b(no|not|never|don't|dont|won't|cannot|can't|cant)\b", flags=re.I)

DISTILBERT_MODEL = None
DISTILBERT_TOKENIZER = None
DISTILBERT_AVAILABLE = False
FALLBACK_PIPE = None

ID2LABEL = {
    0: "hate_speech",
    1: "offensive",
    2: "safe"
}

def text_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"&[#a-z0-9]+;", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"[^\w'\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if STOPWORDS:
        s = " ".join(w for w in s.split() if w not in STOPWORDS and len(w) > 1)
    return s.strip()

def _contains_wordset(text: str, wordset: set) -> bool:
    if not text:
        return False
    t = text.lower()
    for w in wordset:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True
    return False

def explicit_threat(text: str) -> bool:
    if not _contains_wordset(text, VIOLENCE):
        return False
    t = text.lower()
    for m in re.finditer(r"\b(" + "|".join(map(re.escape, VIOLENCE)) + r")\b", t):
        start = max(0, m.start() - 40)
        context = t[start:m.start()]
        if NEGATION_RE.search(context):
            continue
        window = t[max(0, m.start() - 30): m.end() + 30]
        if re.search(r"\b(you|your|i will|i'll|im going to|i am going to)\b", window):
            return True
    return False

def advisory_label_upgrade(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    if _contains_wordset(t, PROTECTED_SLURS):
        return "hate_speech"
    if explicit_threat(t):
        return "hate_speech"
    if _contains_wordset(t, INSULTS):
        return "offensive"
    if _contains_wordset(t, SEXUAL_TERMS):
        return "offensive"
    if _contains_wordset(t, PROFANITY):
        return "offensive"
    return ""


def load_distilbert():
    global DISTILBERT_MODEL, DISTILBERT_TOKENIZER, DISTILBERT_AVAILABLE
    if DISTILBERT_DIR.exists():
        try:
            DISTILBERT_TOKENIZER = DistilBertTokenizer.from_pretrained(str(DISTILBERT_DIR))
            DISTILBERT_MODEL = DistilBertForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
            DISTILBERT_MODEL.to(DEVICE)
            DISTILBERT_MODEL.eval()
            DISTILBERT_AVAILABLE = True
            print("[utils] Loaded DistilBERT")
        except Exception as e:
            print("[utils] Failed to load DistilBERT:", e)
            DISTILBERT_AVAILABLE = False
    else:
        DISTILBERT_AVAILABLE = False

def load_fallback():
    global FALLBACK_PIPE
    if FALLBACK_PATH.exists():
        try:
            FALLBACK_PIPE = joblib.load(str(FALLBACK_PATH))
            print("[utils] Loaded fallback model")
        except Exception as e:
            print("[utils] Failed to load fallback:", e)
            FALLBACK_PIPE = None

load_distilbert()
load_fallback()

def distilbert_predict(text: str):
    encoding = DISTILBERT_TOKENIZER(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = DISTILBERT_MODEL(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    best_idx = int(np.argmax(probs))
    best_label = ID2LABEL.get(best_idx, "safe")
    best_score = float(probs[best_idx])
    probs_dict = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}
    return best_label, best_score, probs_dict

def predict_text(text: str, threshold: float = 0.65):
    if not isinstance(text, str) or not text.strip():
        return {"label": "safe", "confidence": 0.0, "probs": {}, "model": "none"}

    adv_immediate = advisory_label_upgrade(text)
    if adv_immediate == "hate_speech":
        return {"label": adv_immediate, "confidence": 0.9999, "probs": {}, "model": "advisory-immediate"}

    if DISTILBERT_AVAILABLE:
        try:
            label, conf, probs = distilbert_predict(text)
            if conf < threshold:
                adv = advisory_label_upgrade(text)
                if adv:
                    return {"label": adv, "confidence": 0.9999, "probs": probs, "model": "distilbert(advisory)"}
                return {"label": "safe", "confidence": conf, "probs": probs, "model": "distilbert(low_conf)"}
            adv = advisory_label_upgrade(text)
            if label == "safe" and adv:
                return {"label": adv, "confidence": 0.9999, "probs": probs, "model": "distilbert(advisory)"}
            return {"label": label, "confidence": conf, "probs": probs, "model": "distilbert"}
        except Exception as e:
            print("[utils] DistilBERT prediction error:", e)

    if FALLBACK_PIPE is not None:
        try:
            cleaned = text_clean(text)
            probs = FALLBACK_PIPE.predict_proba([cleaned])[0]
            classes = [str(c) for c in FALLBACK_PIPE.classes_]
            idx = int(np.argmax(probs))
            label = classes[idx]
            conf = float(probs[idx])
            probs_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
            return {"label": label, "confidence": conf, "probs": probs_dict, "model": "fallback"}
        except Exception as e:
            print("[utils] Fallback prediction error:", e)

    adv = advisory_label_upgrade(text)
    if adv:
        return {"label": adv, "confidence": 0.9999, "probs": {}, "model": "advisory-only"}

    return {"label": "unknown", "confidence": 0.0, "probs": {}, "model": "none"}

def save_pipeline(pipeline_obj, path: str = str(FALLBACK_PATH), metadata: dict = None):
    try:
        joblib.dump(pipeline_obj, path)
        if metadata:
            joblib.dump(metadata, META_PATH)
        return True
    except Exception as e:
        print("[utils] Failed to save pipeline:", e)
        return False
