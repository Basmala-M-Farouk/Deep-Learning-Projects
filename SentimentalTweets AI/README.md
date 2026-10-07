# 🐦 TweetPulse: Tweet Sentiment Analysis with RNN vs. Bidirectional LSTM

Classifies tweets as **positive or negative** and compares two recurrent architectures on **1.6 million real tweets**. The bidirectional LSTM reaches **82.8% test accuracy**, about 6 points better than a simple RNN.

**Stack:** TensorFlow / Keras · GloVe word embeddings · scikit-learn · pandas · Google Colab

---

## 📊 Results

Both models were evaluated on a held-out test set of 160,000 tweets that wasn't used for training or tuning.

| Model | Test accuracy | Test loss | Embeddings |
|-------|:------------:|:---------:|------------|
| SimpleRNN (64 units) | 76.5% | 0.491 | GloVe 100d, frozen |
| **Bidirectional LSTM (128 units)** | **82.8%** | **0.383** | GloVe 100d, fine-tuned |

**What made the difference:** the bidirectional LSTM reads each tweet in both directions and keeps long-range context (for example "not ... good"), and letting the GloVe embeddings fine-tune adapts them to Twitter slang. The trade-off is training time: about 1.7× longer per epoch than the RNN (≈136 s vs. ≈81 s on a Colab GPU).

---

## 📂 Dataset

**[Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140)**: 1.6 million tweets labelled negative (0) or positive (4, mapped to 1).

| Split | Share | Tweets |
|-------|:-----:|-------:|
| Train | 80% | 1,280,000 |
| Validation | 10% | 160,000 |
| Test | 10% | 160,000 |

Splits are stratified so each one keeps the same positive/negative balance.

> The dataset and GloVe vectors aren't included in this repo. Download them from Kaggle and the [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/) (`glove.6B.100d.txt`).

---

## 🔄 Pipeline

1. **Cleaning:** remove URLs, @mentions, `#` symbols and special characters; lowercase everything.
2. **Tokenization:** keep the 20,000 most frequent words (unknown words map to `<OOV>`) and pad or truncate each tweet to 100 tokens.
3. **Embeddings:** initialise the embedding layer with pre-trained **GloVe 100-dimensional** vectors.
4. **Class weights:** computed with `class_weight='balanced'` and applied during training.
5. **Models:**
   - **RNN:** frozen GloVe → `SimpleRNN(64)` → dropout → dense → sigmoid, trained 6 epochs with Adam (lr 1e-4).
   - **BiLSTM:** fine-tuned GloVe → `Bidirectional(LSTM(128))` → dropout → dense → sigmoid, trained 8 epochs.
6. **Evaluation:** accuracy and loss on the untouched test set, plus training/validation curves for both models.
7. **Try it:** a `predict()` function classifies any sentence you type with both models.

```
Enter tweet to test: I love meat
RNN Prediction:  Positive 😊
LSTM Prediction: Positive 😊
```

---

## 🚀 Run it

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

1. Open `SentimentalTweets AI Project.ipynb` in Google Colab.
2. Put the Sentiment140 CSV at `MyDrive/Twitter_Sentiment_Analysis/training.csv` and the GloVe file at `MyDrive/glove.6B/glove.6B.100d.txt` (or edit the paths in the notebook).
3. Run all cells. Training the BiLSTM takes about 18 minutes on a Colab GPU.

---

## 📁 Files

| File | Purpose |
|------|---------|
| `SentimentalTweets AI Project.ipynb` | Full pipeline: cleaning, tokenization, GloVe embeddings, both models, evaluation, plots and live prediction |

## 💡 What I learned

- Pre-trained embeddings give a strong start; fine-tuning them helps on informal text like tweets.
- Bidirectional LSTMs capture context and negation much better than simple RNNs.
- Keep a test set the models never see, so the comparison is fair.

## 📧 Contact

[LinkedIn](https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/)
