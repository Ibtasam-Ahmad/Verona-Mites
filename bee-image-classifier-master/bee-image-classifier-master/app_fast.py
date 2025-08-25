from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

class ImageRequest(BaseModel):
    image: str

@app.post("/detect")
async def detect(image_request: ImageRequest):
    image_data = base64.b64decode(image_request.image.split(",")[1])
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    height, width, channels = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "bee":  # Assuming you added "bee" in coco.names
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    result = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            result.append({"x": x, "y": y, "width": w, "height": h})

    return {"boxes": result}


if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
# To run the FastAPI app, use:
# uvicorn app:app --reload
