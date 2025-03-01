from flask import Flask, request, render_template, jsonify
import os
from audio_handling.convert_to_wav import main_transcribe
from infering.infer import adjust_transcription
from sentiments.sentimetal import analyze_sentiment_and_suggest
from sentiments.short_sentiments import adjust_short_transcription

app = Flask(__name__)

# Directory setup
UPLOAD_FOLDER = "recordings"
TRANSCRIPT_FOLDER = "Transcripts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

def read_transcript(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    return text

def run(path_to_audio, input_file, output_file):
    main_transcribe(path_to_audio)
    adjust_transcription(input_file, output_file)
    label, suggestion = analyze_sentiment_and_suggest(read_transcript(output_file))
    return label, suggestion

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save uploaded audio file
    audio_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(audio_path)
    
    # Define transcript paths
    input_file = os.path.join(TRANSCRIPT_FOLDER, "transcription.txt")
    output_file = os.path.join(TRANSCRIPT_FOLDER, "output.txt")
    
    # Process the audio and get results
    label, suggestion = run(audio_path, input_file, output_file)
    keywords = adjust_short_transcription(read_transcript(output_file))
    
    # Prepare response
    result = {
        "label": label.strip(),
        "keywords": keywords.strip(),  # Remove padding spaces
        "suggestion": suggestion.strip()
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)