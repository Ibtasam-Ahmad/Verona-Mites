# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import tensorflow as tf
# import cv2
# import os

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the model
# def load_model():
#     model_path = 'varona_model.h5'
#     if os.path.exists(model_path):
#         model = tf.keras.models.load_model(model_path)
#     else:
#         return None
#     return model

# model = load_model()

# def preprocess_image(file):
#     image = cv2.imread(file)
#     image = cv2.resize(image, (50, 50))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     img_arr = tf.keras.preprocessing.image.img_to_array(image)
#     img_arr = img_arr / 255.0
#     np_image = np.expand_dims(img_arr, axis=0)
#     return np_image

# @app.route('/predict', methods=['POST'])

# def predict():
#     if not model:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     try:
#         # Save the uploaded file temporarily
#         temp_path = 'temp_image.png'
#         file.save(temp_path)
        
#         # Preprocess the image and make prediction
#         processed_image = preprocess_image(temp_path)
#         pred_value = model.predict(processed_image)
        
#         # Delete the temporary file
#         os.remove(temp_path)
        
#         # Return prediction result
#         prediction = 'varona' if pred_value < 0.5 else 'no varona'
#         return jsonify({'prediction': prediction})
    
#     except Exception as e:
#         app.logger.error(f"Exception occurred: {str(e)}")
#         return jsonify({'error': 'Internal Server Error'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import cv2
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
def load_model():
    model_path = 'varona_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        return None
    return model

model = load_model()

def preprocess_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = img_arr / 255.0
    np_image = np.expand_dims(img_arr, axis=0)
    return np_image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Save the uploaded file temporarily
        temp_path = 'temp_image.png'
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Preprocess the image and make prediction
        processed_image = preprocess_image(temp_path)
        pred_value = model.predict(processed_image)
        
        # Delete the temporary file
        os.remove(temp_path)
        
        # Return prediction result
        prediction = 'varona' if pred_value < 0.5 else 'no varona'
        print(prediction)
        return JSONResponse(content={'prediction': prediction})
    
    except Exception as e:
        return JSONResponse(content={'error': 'Internal Server Error'}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
