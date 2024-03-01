from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS from flask_cors
from pretrain import Model



app = Flask(__name__)
# Allow requests from 'http://localhost:3000' to the '/make_recommendations' route
CORS(app, origins=["*"])

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/test', methods=['POST'])
def skipGram():
    try:
        data = request.get_json()
        
        model_instance = Model(data['input1'], data['input2'])

        print(f'result:{model_instance.cosine_similarity_scratch()}')

        result = model_instance.cosine_similarity_scratch()
        return result
    except KeyError as e:
        return jsonify({'error': f"KeyError: {str(e)}"}), 404
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)