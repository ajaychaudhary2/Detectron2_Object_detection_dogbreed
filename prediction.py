import torch
import cv2
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

setup_logger()

# Load model configuration and weights
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
    print(f"..")
    metadata = None


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
    """_summary_
    """                   "toy_poodle"]

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
            print(f"Total Classes Available: {len(class_names)}")

            # Validate class ID
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

            # Display the image
            cv2.imshow("Prediction", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return {"bbox": bbox, "confidence": confidence, "class": class_label}

    return {"error": "No detections found"}

if __name__ == "__main__":
    image_path = r"E:\Data_Science _master\DL\CNN\Object detection\detectron2\Detectron2_Object_detection_dogbreed_web\Detectron2_Object_detection_dogbreed\data\images\test\n02116738_607_jpg.rf.dda7d624cafd6979a922165131015a72.jpg"

    result = get_highest_conf_bbox(image_path)
    print(result)
