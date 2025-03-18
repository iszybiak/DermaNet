import keras
import numpy as np
from flask import Flask, request, jsonify


app = Flask(__name__)

# Loading a saved model
model = keras.saving.load_model("model/skin_disease_model.keras")

# Mapping labels
labels = ["melanoma", "nevus", "seborrheic keratosis", "eczema", "psoriasis", "lichen planus", "rosacea"]

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = keras.utils.load_img(file, target_size=(224, 224))
    x = keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)
    predicted_label = labels[np.argmax(prediction)]
    return jsonify({"Disease": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)


