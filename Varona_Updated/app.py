from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to detect brown-colored Varroa mites
def detect_varroa_mites(image_path):
    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to HSV (Hue, Saturation, Value) to focus on brown areas
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper range for brown colors
    lower_brown = np.array([5, 50, 50])  # Adjust these values as needed
    upper_brown = np.array([20, 255, 255])

    # Create a mask that highlights only the brown areas (mites)
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Perform morphological operations to clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours on the brown regions
    contours_brown, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of varroa mites
    num_mites_brown = len(contours_brown)

    # Save the result image with contours drawn
    result_image_path = os.path.join(UPLOAD_FOLDER, "varroa_mites_detected.png")
    contour_image_brown = image.copy()
    cv2.drawContours(contour_image_brown, contours_brown, -1, (0, 255, 0), 2)
    cv2.imwrite(result_image_path, contour_image_brown)

    # Print for debugging instead of showing image
    print("Brown Mask Shape:", brown_mask.shape)
    print("Number of Brown Contours:", num_mites_brown)

    return num_mites_brown, result_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call the detection function
        num_mites, result_image_path = detect_varroa_mites(file_path)

        # Render the result page with the prediction
        return render_template('index.html', prediction=f"Detected Varroa Mites: {num_mites}", filename="varroa_mites_detected.png")

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)