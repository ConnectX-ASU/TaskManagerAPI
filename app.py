import pickle
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

def preprocess(text):
    custom_stop_words = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
        'should', 'now'
    ])
    
    words = re.findall(r'\w+', text.lower())
    filtered_tokens = [word for word in words if word not in custom_stop_words and len(word) > 3]
    
    return filtered_tokens

# Load the SavedModel
model_config_name = 'model_config.json'
model_file_name = 'best_model.h5'

# Load model architecture from JSON
with open(model_config_name, "r") as json_file:
    json_string = json_file.read()
model = tf.keras.models.model_from_json(json_string)

# Load model weights from HDF5
model.load_weights(model_file_name)

tokenizer = pickle.load(open('okenizer.pkl', 'rb'))

@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data['text']
        
        # Preprocess text
        text_processed = preprocess(text)
        text_tokenized = tokenizer.texts_to_sequences([text_processed])
        text_padded = pad_sequences(text_tokenized, maxlen=42, padding='post')
        
        # Perform inference using the loaded model
        predictions = model.predict(text_padded)
        predicted_priority = int(np.argmax(predictions))  # Convert predictions to NumPy array
        
        response = {'predicted_priority': predicted_priority}
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
