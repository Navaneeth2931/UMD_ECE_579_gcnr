from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS

cnn_model_d = load_model('./models/cnn_model_d')
cnn_model_d.load_weights('./models/cnn_model_d_weights.h5')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the folder where speech output files will be stored
SPEECH_FOLDER = 'speech_output'
app.config['SPEECH_FOLDER'] = SPEECH_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(SPEECH_FOLDER):
    os.makedirs(SPEECH_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to serve speech output file
@app.route('/speech/<filename>')
def speech_file(filename):
    return send_from_directory(app.config['SPEECH_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        predicted_caption = generate_caption(filepath)
        speech_output = convert_to_speech(predicted_caption)
        
        return render_template('result.html', image=filename, caption=predicted_caption, speech=speech_output)
    else:
        return "File not allowed"

def generate_caption(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predicted_output = cnn_model_d.predict(img_array)
    
    output_classes = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
    predicted_class_index = np.argmax(predicted_output)
    predicted_class_label = output_classes[predicted_class_index]
    return [predicted_class_label]

def convert_to_speech(text):
    tts = gTTS(text=text[0], lang='en')  # text is now a list, so access the first element
    filename = secure_filename("output.mp3")  # Ensure a unique filename
    file_path = os.path.join(app.config['SPEECH_FOLDER'], filename)
    tts.save(file_path)
    return filename

if __name__ == '__main__':
    app.run(debug=True)