from flask import Flask, render_template, request, send_file
from PIL import Image, ImageFilter, ImageEnhance
import base64
import io
from io import BytesIO
import cv2
import numpy as np

app = Flask(__name__)

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/edit', methods=['POST'])
def edit():
    img_data = request.files['image']
    img = Image.open(img_data)
    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Apply filters based on user inputs
    if 'rotate' in request.form:
        angle = int(request.form['rotate_angle'])
        img = img.rotate(angle)

    if 'blur' in request.form:
        radius = float(request.form['blur_radius'])
        img = img.filter(ImageFilter.GaussianBlur(radius))

    if 'contrast' in request.form:
        factor = float(request.form['contrast_factor'])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)

    if 'crop' in request.form:
        if 'crop_coordinates' in request.form:
            coordinates = list(map(int, request.form['crop_coordinates'].split(',')))
            # Ensure we have all four coordinates
            if len(coordinates) == 4:
                img = img.crop(coordinates)

    if 'grayscale' in request.form:
        img = img.convert('L')

    
    
    if 'face_detection' in request.form:
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
        opencv_img = detect_faces(opencv_img)
        img = Image.fromarray(opencv_img)
    

    # Convert edited image to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('edit.html', img_data=img_str)

# Route for downloading the final image
@app.route('/download')
def download():
    global edited_image
    img_io = io.BytesIO()
    edited_image.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, as_attachment=True, download_name='edited_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)
