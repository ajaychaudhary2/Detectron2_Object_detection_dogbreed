import os
import torch
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

setup_logger()

# Flask app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Detectron2 Model
cfg = get_cfg()
cfg.merge_from_file(r"E:\Data_Science _master\DL\CNN\Object detection\detectron2\Detectron2_Object_detection_dogbreed_web\Detectron2_Object_detection_dogbreed\config\config.yaml")

cfg.MODEL.WEIGHTS = r"E:\Data_Science _master\DL\CNN\Object detection\detectron2\Detectron2_Object_detection_dogbreed_web\Detectron2_Object_detection_dogbreed\model\model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Confidence threshold
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Check if dataset is registered
dataset_name = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else None

if dataset_name and dataset_name in MetadataCatalog.list():
    print(f"✅ Dataset '{dataset_name}' is registered!")
    metadata = MetadataCatalog.get(dataset_name)
else:
    print("⚠️ Warning: Dataset is not registered!")
    metadata = None

# Ensure class names are set
if metadata and hasattr(metadata, "thing_classes") and metadata.thing_classes:
    class_names = metadata.thing_classes
else:
    print("⚠️ Warning: Metadata thing_classes is empty! Setting manually.")
    class_names = ["dogs", "African_hunting_dog", "Appenzeller", "Bernese_mountain_dog",
                   "Brabancon_griffon", "Cardigan", "Doberman", "EntleBucher", "Eskimo_dog",
                   "French_bulldog", "Great_Dane", "Great_Pyrenees", "Greater_Swiss_Mountain_dog",
                   "Leonberg", "Mexican_hairless", "Newfoundland", "Pembroke", "Pomeranian",
                   "Saint_Bernard", "Samoyed", "Siberian_husky", "Tibetan_mastiff", "affenpinscher",
                   "basenji", "boxer", "bull_mastiff", "chow", "dhole", "dingo", "keeshond",
                   "malamute", "miniature_pinscher", "miniature_poodle", "pug", "standard_poodle",
                   "toy_poodle"]

predictor = DefaultPredictor(cfg)

def get_highest_conf_bbox(image_path):
    image = cv2.imread(image_path)
    outputs = predictor(image)

    if "instances" in outputs and outputs["instances"].has("scores"):
        scores = outputs["instances"].scores.tolist()
        boxes = outputs["instances"].pred_boxes.tensor.tolist()
        classes = outputs["instances"].pred_classes.tolist()

        if scores:
            max_idx = scores.index(max(scores))  # Get highest confidence bbox
            bbox = boxes[max_idx]
            confidence = scores[max_idx]
            class_id = classes[max_idx]

            print(f"Predicted Class ID: {class_id}")

            if 0 <= class_id < len(class_names):
                class_label = class_names[class_id]
            else:
                print(f"⚠️ Warning: Class ID {class_id} is out of range!")
                class_label = "Unknown"

            # Convert bounding box to integer format
            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bbox
            cv2.putText(image, f"{class_label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save and return path of processed image
            output_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
            cv2.imwrite(output_path, image)

            return {"bbox": bbox, "confidence": confidence, "class": class_label, "output_image": output_path}

    return {"error": "No detections found"}

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected!"})

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_path)

    # Get prediction
    result = get_highest_conf_bbox(image_path)
    return jsonify(result)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
