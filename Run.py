import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import pickle
from tensorflow.keras.preprocessing.text import tokenizer_from_json
# Load the model
model = load_model('fake_news_detection_model.h5')

# Load the label encoder for y
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Load the tokenizer values from the JSON file
with open('tokenizer_config.json', 'r') as json_file:
    loaded_tokenizer_config = json.load(json_file)
tokenizer = tokenizer_from_json(loaded_tokenizer_config)

# Example input text for prediction
new_texts = ["Michelle Obama DNC speech: ‘I wake up every morning in a house built by slaves’", "Iran, Saudi Arabia to exchange diplomatic visits: Iranian foreign minister"]

# Tokenize and pad the input text
sequences = tokenizer.texts_to_sequences(new_texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=30)

# Make predictions
predictions = model.predict(padded_sequences)
binary_predictions = (predictions > 0.5).astype(int)

# Convert binary predictions back to original labels
predicted_labels = label_encoder.inverse_transform(binary_predictions.flatten())

# Display the results
for text, label in zip(new_texts, predicted_labels):
    print(f"Text: {text}")
    print(f"Predicted Label: {label}")
    print()
