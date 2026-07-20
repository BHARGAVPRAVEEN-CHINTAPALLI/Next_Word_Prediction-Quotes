# Next Word Prediction - Quotes ✨

A next-word text generator trained on a dataset of inspirational quotes, using an LSTM (Long Short-Term Memory) neural network built with TensorFlow/Keras. Given a starting word or phrase, the model predicts and generates the next several words, one at a time, in the style of the quotes it was trained on.

An interactive **Streamlit** app is included so you can generate text directly from your browser.

---

## 🧠 How It Works

The model is trained on a dataset of quotes and learns to predict the most probable next word given a sequence of preceding words.

At generation time, the process repeats in a loop:

1. **Tokenize** — the current sentence is converted into integer tokens using a `Tokenizer` fit on the training vocabulary.
2. **Pad** — the token sequence is padded with zeros at the front to match the fixed input length the LSTM expects.
3. **Predict** — the padded sequence is passed through the trained LSTM, which outputs a probability distribution over the entire vocabulary (softmax).
4. **Select next word** — the word with the highest probability (`argmax`) is chosen and appended to the sentence.

This repeats word-by-word, with each new word conditioned on everything generated so far, until the requested number of words has been generated.

---

## 🏗️ Model Architecture

- **Embedding layer** — maps each word index to a dense vector representation
- **LSTM layer(s)** — learn sequential/contextual patterns in the quote text
- **Dense (softmax) output layer** — predicts the probability of each word in the vocabulary being the next word

Trained using categorical cross-entropy loss and the Adam optimizer.

---

## 📈 How This Scales to Large Language Models

The core mechanism in this project — tokenize → pad → predict → pick next word → repeat — is the same fundamental idea used by large-scale language models (like GPT-style models). Only the scale and the machinery around it change.

### The underlying question never changes
Every next-word predictor, small or huge, is estimating the same thing:

> Given the words seen so far, what's the probability distribution over every possible next word?

This project learns that distribution from a few hundred quotes. Large-scale models learn it from massive text corpora — the difference is scale, not concept.

### What changes as the dataset grows

| Aspect | This Project | Large-Scale Models |
|---|---|---|
| **Vocabulary** | ~1,199 whole words | 50,000+ subword tokens |
| **Tokenization** | Whole-word (`Tokenizer`) | Subword (e.g. Byte-Pair Encoding), so even unseen words can be represented |
| **Architecture** | Stacked LSTMs (sequential) | Transformers (self-attention, parallelizable) |
| **Context window** | Fixed ~34 tokens | Thousands to hundreds of thousands of tokens |
| **Training data** | Hundreds of quotes | Hundreds of billions to trillions of tokens |
| **Compute** | Minutes on CPU | Thousands of GPUs/TPUs over weeks or months |

### Why word-level LSTMs don't scale well
If this architecture were trained on something like all of Wikipedia instead of quotes:
- Training time would explode, since LSTMs process sequences one step at a time and don't parallelize well over long sequences
- The word-level vocabulary would balloon and become sparse (rare words, typos, etc. all need their own token)
- LSTMs suffer from **vanishing gradients** over long sequences, so they struggle to retain context from far earlier in a passage — more data wouldn't fix this on its own

This is exactly why large-scale language models moved to the **Transformer** architecture: it isn't just about raw performance, but about making training on huge datasets computationally feasible at all.

### Generation logic is the same everywhere
Even in state-of-the-art models, the generation loop looks like this:

```
repeat:
    1. take the current sequence
    2. run it through the model
    3. get a probability distribution over the vocabulary
    4. select the next token
    5. append the token to the sequence
```

The main practical difference is **step 4 — how the next token is chosen**. This project uses greedy `argmax` selection (always pick the single most probable word), which is deterministic but can produce repetitive text. Large-scale models commonly use smarter sampling instead:

- **Temperature sampling** — softens or sharpens the probability distribution to control randomness/creativity
- **Top-k sampling** — only samples from the k most probable next tokens
- **Top-p (nucleus) sampling** — samples from the smallest set of tokens whose combined probability exceeds a threshold p

These produce more natural, varied output instead of always generating the same continuation for the same input.

---

## 📂 Repository Structure

| File | Description |
|---|---|
| `Untitled.ipynb` | Notebook containing data preprocessing, model building, and training code |
| `app.py` | Streamlit app for interactive text generation |
| `final_model.keras` | Trained LSTM model, saved in Keras native format |
| `tokenizer.pkl` | Fitted tokenizer (pickled) used to convert text to sequences |
| `requirements.txt` | Python dependencies needed to run the app |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/BHARGAVPRAVEEN-CHINTAPALLI/Next_Word_Prediction-Quotes.git
cd Next_Word_Prediction-Quotes
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

This will open the app in your browser, where you can:
- Enter a starting word or phrase
- Choose how many words to generate
- Watch the sentence build word-by-word
- View a loop-by-loop breakdown of each prediction, including the model's confidence for each word

---

## 📊 Example

**Input:** `Embrace`
**Generated (15 words):**
> Embrace the power of a warm smile it can light up even the gloomiest days it

---

## 🔧 Retraining the Model

If you'd like to retrain the model on your own dataset of quotes:

1. Open `Untitled.ipynb`
2. Replace the dataset path with your own `.csv` file containing a `Quotes` column
3. Re-run the notebook cells to preprocess the text, build input sequences, and train the LSTM
4. Save the new model and tokenizer:

```python
model.save("final_model.keras")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

5. Replace the old `final_model.keras` and `tokenizer.pkl` files in this repo with the new ones

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy

---

## 📌 Notes

- The model uses **greedy decoding** (`argmax`), so the same starting word will always generate the same output. For more varied/creative output, temperature-based sampling could be added instead of always picking the highest-probability word.
- Generation quality depends heavily on the size and diversity of the training dataset used.

---

## 📄 License

This project currently has no license specified. Add one (e.g. MIT) if you'd like others to be able to reuse this code.
