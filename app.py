import os
import glob
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer

# ------------------------------------------------------------------
# PATHS (all relative to the repo root, same folder as app.py)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # <-- folder where app.py lives

MODEL_PATH = os.path.join(BASE_DIR, "medical_transformer.keras")  # <-- rebuilt at runtime from the parts
PARTS_DIR = os.path.join(BASE_DIR, "model_parts")                 # <-- chunks folder (part_000, part_001, ...)
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer")              # <-- tokenizer folder
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")               # <-- config file


# ------------------------------------------------------------------
# REBUILD MODEL FROM PARTS (runs only once if the file doesn't exist)
# ------------------------------------------------------------------
def rebuild_model():
    if os.path.exists(MODEL_PATH):
        return
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "part_*")))
    if not parts:
        raise FileNotFoundError(f"No model parts found in {PARTS_DIR}")
    with open(MODEL_PATH, "wb") as out:
        for part in parts:
            with open(part, "rb") as p:
                out.write(p.read())


# ------------------------------------------------------------------
# CUSTOM LAYERS
# Paste your custom classes from the notebook here
# (PositionalEncoding, TransformerBlock, masks, etc.)
# ------------------------------------------------------------------
# class PositionalEncoding(tf.keras.layers.Layer): ...
# class TransformerBlock(tf.keras.layers.Layer): ...
# CUSTOM_OBJECTS = {"PositionalEncoding": PositionalEncoding, ...}
CUSTOM_OBJECTS = {}


@st.cache_resource
def load_all():
    rebuild_model()
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    return cfg, tokenizer, model


with st.spinner("Loading model (first run may take a few minutes)..."):
    cfg, tokenizer, model = load_all()
MAX_LEN = cfg["MAX_LENGTH"]


def generate(question: str) -> str:
    # NOTE: must match the preprocessing/decoding used in your notebook
    # (start/end tokens, padding, greedy decoding, ...)
    enc = tokenizer(
        question,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    enc_in = enc["input_ids"]

    start_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    end_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0
    dec = [start_id]

    for _ in range(MAX_LEN - 1):
        dec_in = np.array([dec + [pad_id] * (MAX_LEN - len(dec))])
        preds = model.predict([enc_in, dec_in], verbose=0)
        next_id = int(np.argmax(preds[0, len(dec) - 1]))
        if next_id == end_id:
            break
        dec.append(next_id)

    return tokenizer.decode(dec[1:], skip_special_tokens=True)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Medical Chatbot")
st.caption("⚠️ This chatbot is not a substitute for a doctor. For educational use only.")

with st.sidebar:
    st.header("About")
    st.write("A transformer-based chatbot trained on medical Q&A data.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
