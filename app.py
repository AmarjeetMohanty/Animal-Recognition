from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from fastapi.responses import FileResponse

app = FastAPI()



# Your existing code up to the model definition

# Load the trained model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(712, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.summary()

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Cat and Dog Image Classifier</title>
        </head>
        <body>
            <h1>Cat and Dog Image Classifier</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input type="file" name="file">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        result = "Dog"
    else:
        result = "Cat"
    
    return {"prediction": result}

# Add a route to serve the visualization image grid
# @app.get("/visualization/")
# def visualization():
#     # Your code to create and save the visualization grid
#     return FileResponse("path_to_visualization_image_grid.png", media_type="image/png")
