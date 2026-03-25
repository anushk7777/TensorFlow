import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    "I love this video, very helpful!",
    "You are an idiot and should shut up",
    "Great explanation, thanks for sharing",
    "What a stupid person, go away",
    "This is such a wonderful day",
    "I hate you, you are worthless",
    "Beautiful sunset today!",
    "Kill yourself, nobody likes you",
    "Thanks for the kind words",
    "You are garbage and deserve nothing",
]

labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

vocab_size = 1000
max_len = 20
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(np.array(padded_sequences), labels, epochs=20, verbose=1)


def predict_toxicity(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "TOXIC" if score > 0.5 else "NOT_TOXIC"

    return {
        "text": text,
        "prediction": label,
        "score": score
    }


if __name__ == "__main__":
    test_samples = [
        "This tutorial is amazing!",
        "You are a terrible human being",
        "Have a great weekend",
        "I want to hurt you"
    ]

    for sample in test_samples:
        res = predict_toxicity(sample)
        print(f"[{res['prediction']:<9}] (Score: {res['score']:.4f}) {res['text']}")