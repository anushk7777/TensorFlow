import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    "All people of that religion are evil and should be removed",
    "You are so stupid and ugly",
    "The weather is lovely today",
    "That ethnic group is inferior and dangerous",
    "What an idiot, I can't believe he said that",
    "I really enjoyed the movie last night",
    "Those immigrants are ruining the country",
    "Shut up, nobody asked for your opinion",
    "Thanks for the help, really appreciate it",
    "Women should not be allowed to work",
    "You are the worst person I've ever met",
    "Good morning, hope you have a great day",
]

labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

vocab_size = 1000
max_len = 20

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.fit(np.array(padded_sequences), labels, epochs=30, verbose=1)

class_names = ["HATE_SPEECH", "OFFENSIVE", "NEITHER"]


def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    pred_class = np.argmax(probs)

    return {
        "text": text,
        "prediction": class_names[pred_class],
        "probs": {class_names[i]: float(probs[i]) for i in range(3)}
    }


if __name__ == "__main__":
    test_samples = [
        "Have a wonderful day!",
        "You are such a moron",
        "That group of people are subhuman",
        "I disagree with your opinion"
    ]

    for sample in test_samples:
        res = predict_text(sample)
        print(f"[{res['prediction']}] {res['text']}")