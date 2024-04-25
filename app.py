#Imports
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
import json
import nltk
from nltk.corpus import stopwords
# Download NLTK data
nltk.download('stopwords')
#Preprocessing text function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    custom_stop_words = ['from', 'subject', 're', 'edu', 'use']
    stop_words.update(custom_stop_words)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Filter out stopwords and short tokens
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words and len(token) > 3]

    return filtered_tokens

# Load the trained model
model = load_model("my_model2.h5")


# Set up Flask
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():

    # Get data from JSON request
    data = request.get_json(force=True)
    text = data['text']

    # Preprocess the text
    text_processed = preprocess(text)
    
    #Loading the tokenizer
    tokenizer_path = 'okenizer.pkl'
    with open(tokenizer_path, 'rb') as f:
        loaded_tokenizer = pickle.load(f)

    #Tokenization
    text_tokenized = loaded_tokenizer.texts_to_sequences([text_processed])  # Note: texts_to_sequences expects a list of strings

    #Padding
    text_padded = pad_sequences(text_tokenized, maxlen=42, padding='post')

    # Use the model to predict
    predictions = model.predict(text_padded)

    # Get the predicted priority
    predicted_priority = int(np.argmax(predictions))
    
    # Return the predicted priority as JSON response
    response = {'predicted_priority': predicted_priority}
    return jsonify(response)


   
    

if __name__ == '__main__':
    app.run(debug=True)
