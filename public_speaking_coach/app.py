import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pytube import YouTube
from pydub import AudioSegment
import nltk
from pydub.silence import split_on_silence

app = Flask(__name__)

# Ensure NLTK 'punkt' data is available
nltk.download('punkt')

# Configuration for file uploads
UPLOAD_FOLDER = '/home/koyilada-keerti/Desktop/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Excel data
data_path = '/home/koyilada-keerti/Downloads/top_15_views.xlsx'
df = pd.read_excel(data_path)

# Calculate average values to use as thresholds
clarity_threshold = df['Articulation Rate'].mean()
emotional_threshold = df['Pitch'].mean()
# Assuming head_turn_angles is a string of comma-separated values
hand_gestures_threshold = df['head_turn_angles'].apply(lambda x: len(x.split(','))).mean()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_speech(audio_path):
    # Load audio file
    audio = AudioSegment.from_file(audio_path)

    # Split audio on silence for segment analysis
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    feedback_list = []
    start_time = 0
    for i, chunk in enumerate(chunks):
        end_time = start_time + len(chunk)  # end_time in milliseconds

        # Calculate the timestamp for each chunk in minutes and seconds
        start_minutes = start_time // 60000
        start_seconds = (start_time % 60000) // 1000
        end_minutes = end_time // 60000
        end_seconds = (end_time % 60000) // 1000
        timestamp = "{:02d}:{:02d} - {:02d}:{:02d}".format(start_minutes, start_seconds, end_minutes, end_seconds)

        # Placeholder for segment text extraction
        text = "This is placeholder text for segment {}".format(i+1)

        # Tokenize the text for analysis
        tokens = nltk.word_tokenize(text)

        # Example feedback generation (simplified)
        segment_feedback = {
            'timestamp': timestamp,
            'speech_clarity': "Segment {}: {}".format(i+1, df.columns[12]),  # Articulation Rate
            'emotional_expression': "Segment {}: {}".format(i+1, df.columns[10]),  # Pitch
            'hand_gestures': "Segment {}: {}".format(i+1, df.columns[8])  # Head Turn Angles
        }

        # Compare against the dataset
        if len(tokens) > clarity_threshold:
            segment_feedback['speech_clarity'] = "Segment {}: Good speech clarity.".format(i+1)
        else:
            segment_feedback['speech_clarity'] = "Segment {}: Speech clarity needs improvement.".format(i+1)

        # Placeholder analysis values
        analyzed_emotional_expression = 50  # Replace with actual analysis
        analyzed_hand_gestures = 30  # Replace with actual analysis

        if analyzed_emotional_expression > emotional_threshold:
            segment_feedback['emotional_expression'] = "Segment {}: Good emotional expression.".format(i+1)
        else:
            segment_feedback['emotional_expression'] = "Segment {}: Emotional expression needs improvement.".format(i+1)

        if analyzed_hand_gestures > hand_gestures_threshold:
            segment_feedback['hand_gestures'] = "Segment {}: Good use of hand gestures.".format(i+1)
        else:
            segment_feedback['hand_gestures'] = "Segment {}: Hand gestures need improvement.".format(i+1)

        feedback_list.append(segment_feedback)
        start_time = end_time  # Update the start_time for the next chunk

    return feedback_list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    url = request.form['youtube_url']
    if not url:
        return 'No URL provided'
    
    try:
        # Download the YouTube video
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        video_path = stream.download(output_path=app.config['UPLOAD_FOLDER'])

        # Convert video to audio
        audio_clip = AudioSegment.from_file(video_path)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
        audio_clip.export(audio_path, format='wav')

        feedback = analyze_speech(audio_path)
        
        return render_template('index.html', feedback=feedback)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
