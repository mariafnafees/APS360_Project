from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from math import floor
import os

app = Flask(__name__)

### Model + Class Definition ###
CLASSES = ["apple", "bee", "cat", "eyeglasses", "fish", "flower", "house", "pencil", "pizza"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

class Alexnet_Classifier(nn.Module):
    def __init__(self):
        super(Alexnet_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.x = floor((6 - 3 + 1) / 2)
        self.FC_input = 512 * self.x * self.x
        self.fc1 = nn.Linear(self.FC_input, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.FC_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# load feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = models.alexnet(pretrained=True).to(device)
alexnet.eval()

# load trained classifier
classifier = Alexnet_Classifier().to(device)
classifier.load_state_dict(torch.load(os.path.join("data", "alexnet_classifier_trained.pth"), map_location=device))
classifier.eval()

### Flask Routes ####
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "drawing" not in data:
        return jsonify({"error": "No image data provided."})

    # extract base64 image data
    drawing_data = data["drawing"].split(",")[1]
    image_bytes = base64.b64decode(drawing_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # preprocess and predict
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = alexnet.features(image_tensor)
        output = classifier(features)
        pred_idx = output.argmax(dim=1).item()
        pred_label = CLASSES[pred_idx]

    return jsonify({"prediction": pred_label})


### Run Flask App ###
if __name__ == "__main__":
    app.run(debug=True)
