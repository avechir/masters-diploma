import os
import glob
import cv2
from parsing import parse_filename
import json

def predict_images(model, images, save_json="predictions.json", conf_model=0.25):
    results_list = []
    if isinstance(images, str):  
        image_paths = sorted(glob.glob(os.path.join(images, "*.jpg")))
    elif isinstance(images, list):  
        image_paths = sorted(images)
    else:
        raise ValueError("source must be a string or a list")

    # print(len(image_paths))
    for img_path in image_paths:
        img = cv2.imread(img_path)
        results = model.predict(img, imgsz=640, verbose=False, conf=conf_model)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()  
            classes = r.boxes.cls.cpu().numpy().tolist() 
            # print(boxes, confs, classes)

            fname = os.path.basename(img_path)
            timestampsec, timestampnanosec = parse_filename(fname)

            for box, conf, cls in zip(boxes, confs, classes):
                results_list.append({
                    "file": fname,
                    "timestampsec": timestampsec,
                    "timestampnsec": timestampnanosec,
                    "class": int(cls),
                    "confidence": float(conf),
                    "bbox": box
                })

    with open(save_json, "w") as f:
        json.dump(results_list, f, indent=2)

    return results_list