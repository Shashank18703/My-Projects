import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np


mean = 0.1307
std = 0.3081

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 10, stride = 1, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, stride = 1, kernel_size = 5)
        self.fc = nn.Linear(in_features = 320, out_features = 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc(x)

        return x
    
model = CNN_MNIST()
model.load_state_dict(torch.load("CNN_MNIST_model.pth", map_location=torch.device('cpu')))
model.eval()

metadata = joblib.load("model_metadata.pkl")

def preprocess_digit(image) -> torch.Tensor:
    img = image.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img[np.newaxis, np.newaxis, :, :] # shape should be (1, 1, 28, 28)
    return torch.tensor(img, dtype = torch.float32)

def predict_digit_from_canvas_b64(b64_string: str):
    img_bytes = base64.b64decode(b64_string.split(",")[1])
    img = Image.open(BytesIO(img_bytes)).convert("L") # convert to grayscale
    img = img.resize((28, 28), Image.LANCZOS)
    img_arr = np.array(img)

    x = preprocess_digit(img_arr)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim = 1).numpy()[0]
        pred_digit = int(probs.argmax())
    return pred_digit, probs.tolist()


# Building the Flask Backend
from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    data = request.get_json()
    digit, probs = predict_digit_from_canvas_b64(data['image'])
    return jsonify({
        "predicted_digit": digit,
        "probabilities": probs
    })

if __name__ == "__main__":
    app.run(debug = True)
# End of app.py