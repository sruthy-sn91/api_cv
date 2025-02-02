from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import FileResponse
from typing import List
import io
import cv2
import numpy as np
import os

app = FastAPI()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur_image = cv2.GaussianBlur(grey_image, (7,7), 0)
    
    # Edge detection
    edge = cv2.Canny(blur_image, 100, 200)

    # Save the processed image to a temporary file
    temp_file_path = "temp_processed_image.png" 
    cv2.imwrite(temp_file_path, edge)

    # Return the processed image as a file response
    return FileResponse(temp_file_path, media_type="image/png", filename="processed_image.png")
