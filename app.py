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


#Preprocessing text function
def preprocess(text):
    custom_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

    # Tokenize the text
    tokens = word_tokenize(text)

    # Filter out custom stopwords and short tokens
    filtered_tokens = [token for token in tokens if token.lower() not in custom_stop_words and len(token) > 3]

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
