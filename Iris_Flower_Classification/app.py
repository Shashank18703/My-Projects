from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import joblib

# ----- Model definition (same as in notebook) -----
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# ----- Load model and mapping -----
model = IrisClassifier()
model.load_state_dict(torch.load("iris_model.pth", map_location="cpu"))
model.eval()

species_map = joblib.load("species_map.pkl")

app = Flask(__name__)

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    with torch.no_grad():
        x_single = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)
        output = model(x_single)
        _, pred = torch.max(output.data, 1)
        label_idx = int(pred.item())
        return species_map[label_idx]

# ----- Routes -----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sl = float(data["sepal_length"])
    sw = float(data["sepal_width"])
    pl = float(data["petal_length"])
    pw = float(data["petal_width"])

    species_name = predict_iris(sl, sw, pl, pw)
    return jsonify({"species": species_name})

if __name__ == "__main__":
    app.run(debug=True)
