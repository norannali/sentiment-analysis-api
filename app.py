from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer on startup
print("Loading model and tokenizer")
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer_config = pickle.load(f)

word_index = tokenizer_config['word_index']
MAX_WORDS = tokenizer_config['max_words']
MAX_LEN = tokenizer_config['max_len']

print("Model and tokenizer loaded successfully!")


def preprocess_text(text):
    """Convert text to sequence of integers and pad it."""
    # Convert text to lowercase and split into words
    words = text.lower().split()
    
    # Convert words to indices (add 3 to match IMDb dataset offset)
    sequence = []
    for word in words:
        if word in word_index:
            idx = word_index[word] + 3  # IMDb uses offset of 3
            if idx < MAX_WORDS:
                sequence.append(idx)
        # Unknown words are skipped
    
    # Pad sequence
    padded = pad_sequences([sequence], maxlen=MAX_LEN)
    return padded


@app.route('/health', methods=['GET'])
def health_check():
    """Check if API is running."""
    return jsonify({
        'status': 'healthy',
        'message': 'Sentiment Analysis API is running',
        'model_loaded': model is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment of given text."""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request body'
            }), 400
        
        text = data['text']
        
        # Check if text is empty
        if not text or not text.strip():
            return jsonify({
                'error': 'Text field cannot be empty'
            }), 400
        
        # Preprocess and predict
        processed_text = preprocess_text(text)
        prediction = model.predict(processed_text, verbose=0)[0][0]
        
        # Determine label and confidence
        if prediction >= 0.5:
            label = 'positive'
            confidence = float(prediction)
        else:
            label = 'negative'
            confidence = float(1 - prediction)
        
        # Return response
        return jsonify({
            'text': text,
            'predicted_label': label,
            'confidence': round(confidence, 4)
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict sentiment for multiple texts."""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing "texts" field in request body'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': '"texts" must be a list'
            }), 400
        
        results = []
        for text in texts:
            if text and text.strip():
                processed_text = preprocess_text(text)
                prediction = model.predict(processed_text, verbose=0)[0][0]
                
                if prediction >= 0.5:
                    label = 'positive'
                    confidence = float(prediction)
                else:
                    label = 'negative'
                    confidence = float(1 - prediction)
                
                results.append({
                    'text': text,
                    'predicted_label': label,
                    'confidence': round(confidence, 4)
                })
            else:
                results.append({
                    'text': text,
                    'error': 'Empty text'
                })
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


@app.route('/', methods=['GET'])
def home():
    """API documentation."""
    return jsonify({
        'message': 'Sentiment Analysis API',
        'endpoints': {
            'GET /health': 'Check API health status',
            'POST /predict': 'Predict sentiment for a single text',
            'POST /predict/batch': 'Predict sentiment for multiple texts'
        },
        'example_usage': {
            '/predict': {
                'input': {'text': 'I really loved this movie!'},
                'output': {'predicted_label': 'positive', 'confidence': 0.93}
            }
        }
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)