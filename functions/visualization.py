import cv2
import matplotlib.pyplot as plt
import os

def draw_box(img, bbox, label, color, conf=None):
    if bbox is None:
        return img
    
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    text = label
    if conf is not None:
        text += f" {conf:.2f}"
    
    cv2.putText(img, text, (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def show_fp_images(matches, images_path, class_names, max_show=10):
    fps = [m for m in matches if m["type"] == "FP"]

    for i, fp in enumerate(fps[:max_show]):
        img_path = os.path.join(images_path, fp["file"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_box(img, fp["pred_bbox"], 
                       f"Pred: {class_names[fp['pred_class']]}", 
                       (255,0,0), fp["confidence"])

        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"FP: {fp['file']}")
        plt.show()


def show_fn_images(matches, images_path, class_names, max_show=10):
    fns = [m for m in matches if m["type"] == "FN"]

    for i, fn in enumerate(fns[:max_show]):
        img_path = os.path.join(images_path, fn["file"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = draw_box(img, fn["gt_bbox"], 
                       f"GT: {class_names[fn['gt_class']]}", 
                       (0,255,0))

        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"FN: {fn['file']}")
        plt.show()
