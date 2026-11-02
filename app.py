import os
import glob
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")  # must be the first Streamlit call

# ------------------------------------------------------------------
# PATHS (all relative to the repo root, same folder as app.py)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def find_path(parent: str, name: str) -> str:
    """Case-insensitive lookup (Linux is case-sensitive: Tokenizer vs tokenizer)."""
    exact = os.path.join(parent, name)
    if os.path.exists(exact):
        return exact
    if os.path.isdir(parent):
        for item in os.listdir(parent):
            if item.lower() == name.lower():
                return os.path.join(parent, item)
    return exact


MODEL_PATH = os.path.join(ARTIFACTS_DIR, "medical_transformer.keras")  # rebuilt from parts
PARTS_DIR = find_path(ARTIFACTS_DIR, "model_parts")                     # part_000, part_001, ...
TOKENIZER_PATH = find_path(ARTIFACTS_DIR, "Tokenizer")                  # works with Tokenizer/tokenizer
CONFIG_PATH = find_path(ARTIFACTS_DIR, "config.json")

# ------------------------------------------------------------------
# CONFIG (needed before the layers so they get the right defaults)
# ------------------------------------------------------------------
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = json.load(f)

MAX_LENGTH = int(CFG.get("MAX_LENGTH", 256))
CONTEXT = MAX_LENGTH - 1                      # model context = 255
D_MODEL = int(CFG.get("DMODEL", 512))
NUM_HEADS = int(CFG.get("NUM_HEADS", 8))
FF_DIM = int(CFG.get("FF_DIM", 2048))
NUM_LAYERS = int(CFG.get("NUM_LAYERS", 4))
DROPOUT = float(CFG.get("DROPOUT", 0.1))
VOCAB_SIZE = 50257


# ------------------------------------------------------------------
# REBUILD MODEL FROM PARTS (runs only once if the file doesn't exist)
# ------------------------------------------------------------------
def rebuild_model():
    if os.path.exists(MODEL_PATH):
        return
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "part_*")))
    if not parts:
        raise FileNotFoundError(f"No model parts found in {PARTS_DIR}")
    tmp_path = MODEL_PATH + ".tmp"
    with open(tmp_path, "wb") as out:
        for part in parts:
            with open(part, "rb") as p:
                while True:
                    chunk = p.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    os.replace(tmp_path, MODEL_PATH)


# ------------------------------------------------------------------
# CUSTOM LAYERS (decoder-only Transformer, same as the notebook)
# ------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="chatdoc")
class AddPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len=CONTEXT, d_model=D_MODEL, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(angle[None, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


@tf.keras.utils.register_keras_serializable(package="chatdoc")
class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def _split(self, x, batch):
        x = tf.reshape(x, (batch, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch = tf.shape(x)[0]
        seq = tf.shape(x)[1]
        q = self._split(self.wq(x), batch)
        k = self._split(self.wk(x), batch)
        v = self._split(self.wv(x), batch)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        mask = tf.linalg.band_part(tf.ones((seq, seq)), -1, 0)  # causal mask
        scores += (1.0 - mask) * -1e9
        weights = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(weights, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch, seq, self.d_model))
        return self.dense(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg


@tf.keras.utils.register_keras_serializable(package="chatdoc")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.att = CausalSelfAttention(d_model, num_heads)
        self.ffn1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(d_model)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        a = self.drop1(self.att(x), training=training)
        x = self.ln1(x + a)
        f = self.drop2(self.ffn2(self.ffn1(x)), training=training)
        return self.ln2(x + f)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "ff_dim": self.ff_dim, "dropout": self.dropout_rate,
        })
        return cfg


CUSTOM_OBJECTS = {
    "AddPositionalEncoding": AddPositionalEncoding,
    "CausalSelfAttention": CausalSelfAttention,
    "TransformerBlock": TransformerBlock,
}


def build_model():
    inp = tf.keras.Input(shape=(CONTEXT,), dtype="int32")
    x = tf.keras.layers.Embedding(VOCAB_SIZE, D_MODEL)(inp)
    x = AddPositionalEncoding(CONTEXT, D_MODEL)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    for _ in range(NUM_LAYERS):
        x = TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM, DROPOUT)(x)
    out = tf.keras.layers.Dense(VOCAB_SIZE)(x)
    return tf.keras.Model(inp, out)


@st.cache_resource(show_spinner=False)
def load_all():
    rebuild_model()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    except Exception:
        # Fallback: rebuild the architecture and load only the weights
        model = build_model()
        model.load_weights(MODEL_PATH)
    return tokenizer, model


with st.spinner("Loading model (first run may take a few minutes)..."):
    tokenizer, model = load_all()

EOS_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256


@tf.function(reduce_retracing=True)
def forward(x):
    return model(x, training=False)


# ------------------------------------------------------------------
# GENERATION (decoder-only, single input, prompt format from the notebook)
# ------------------------------------------------------------------
def generate(question, max_new_tokens=100, temperature=0.8, top_k=40):
    prompt = f"<|patient|> {question.strip()} <|doctor|>"
    ids = tokenizer.encode(prompt)[-CONTEXT:]
    generated = []

    for _ in range(max_new_tokens):
        window = ids[-CONTEXT:]
        n = len(window)
        x = np.full((1, CONTEXT), EOS_ID, dtype=np.int32)   # pad with EOS (as in training)
        x[0, :n] = window
        logits = forward(tf.constant(x)).numpy()[0, n - 1].astype(np.float64)

        if temperature <= 0:
            next_id = int(np.argmax(logits))
        else:
            logits = logits / temperature
            if top_k and top_k < logits.shape[0]:
                top_idx = np.argpartition(logits, -top_k)[-top_k:]
                top_logits = logits[top_idx]
                probs = np.exp(top_logits - top_logits.max())
                probs /= probs.sum()
                next_id = int(np.random.choice(top_idx, p=probs))
            else:
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))

        if next_id == EOS_ID:
            break
        ids.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🩺 Medical Chatbot")
st.caption("⚠️ This chatbot is not a substitute for a doctor. For educational use only.")

with st.sidebar:
    st.header("About")
    st.write("A transformer-based chatbot trained from scratch on medical Q&A data.")
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.1)
    top_k = st.slider("Top-k", 1, 100, 40)
    max_new = st.slider("Max new tokens", 20, 200, 100, 10)
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
            answer = generate(prompt, max_new, temperature, top_k) or "…"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
