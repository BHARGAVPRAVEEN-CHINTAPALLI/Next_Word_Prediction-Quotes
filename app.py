import time
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Config
# ------------------------------
MODEL_PATH = "final_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 34   # should match X.shape[1] from training (padded_input_sequences.shape[1] - 1)

st.set_page_config(page_title="LSTM Text Generator", page_icon="📝")

# ------------------------------
# Load model and tokenizer (cached so it only loads once)
# ------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# Build reverse word index once (index -> word) for fast lookup
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# ------------------------------
# Text generation function (step-by-step generator)
# ------------------------------
def generate_text_steps(seed_text, num_words, max_len=MAX_LEN):
    """
    Yields one dict per loop iteration, mirroring the original notebook logic:
      1. tokenize the current text
      2. pad it to max_len
      3. predict the next word's index (argmax over softmax output)
      4. look up the word for that index and append it
    """
    text = seed_text
    for step in range(1, num_words + 1):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len, padding='pre')

        probs = model.predict(padded_token_text, verbose=0)[0]
        pos = int(np.argmax(probs))
        confidence = float(probs[pos])
        word = index_to_word.get(pos)

        if word is None:
            break

        text = text + " " + word

        yield {
            "step": step,
            "predicted_word": word,
            "predicted_index": pos,
            "confidence": confidence,
            "text_so_far": text,
        }

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📝 LSTM Next-Word Text Generator")
st.write("Enter a starting word (or phrase) and choose how many words you'd like generated.")

seed_text = st.text_input("Starting word / phrase", value="Embrace")
num_words = st.slider("Number of words to generate", min_value=1, max_value=50, value=15)

if st.button("Generate"):
    if not seed_text.strip():
        st.warning("Please enter a starting word.")
    else:
        st.subheader("Generated Text")
        output_placeholder = st.empty()

        steps = []
        current_text = seed_text.strip()
        output_placeholder.info(current_text)

        # Stream word-by-word, updating the same placeholder each loop
        for step_info in generate_text_steps(current_text, num_words):
            steps.append(step_info)
            output_placeholder.success(step_info["text_so_far"])
            time.sleep(0.4)  # small delay so each new word is visible appearing

        # ------------------------------
        # How the machine works (explanation)
        # ------------------------------
        st.markdown("---")
        st.subheader("⚙️ How the Machine Generates Each Word")
        st.markdown(
            """
This model predicts text **one word at a time**, feeding its own output back in as input for the next
prediction. For every loop, four things happen:

1. **Tokenize** — the current sentence so far is converted into numbers using the same `Tokenizer`
   that was fit on the training data. Each word maps to a fixed integer ID.
2. **Pad** — the number sequence is padded with zeros at the front so it always matches the fixed
   input length (`max_len`) the LSTM was trained on.
3. **Predict** — the padded sequence is passed through the trained LSTM, which outputs a probability
   for every word in the vocabulary (a softmax over ~1199 words in this case).
4. **Pick the next word** — the word with the **highest probability** (`argmax`) is chosen and
   appended to the sentence. That new, longer sentence becomes the input for the next loop.

This repeats for however many words you asked for, so the sentence grows one word at a time,
each new word conditioned on everything generated so far.
            """
        )

        # ------------------------------
        # Loop-by-loop breakdown (like the original notebook prints)
        # ------------------------------
        st.subheader("🔁 Loop-by-Loop Breakdown")
        for step_info in steps:
            st.write(
                f"**Loop {step_info['step']}** → predicted word: "
                f"`{step_info['predicted_word']}` (index {step_info['predicted_index']}, "
                f"confidence {step_info['confidence']:.2%})  \n"
                f"Sentence so far: *{step_info['text_so_far']}*"
            )