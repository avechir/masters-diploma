import os

def parse_filename(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    timestampsec = int(parts[-2])
    timestampnanosec = int(parts[-1])
    return timestampsec, timestampnanosec

def parse_yolo_label(label_path, img_width, img_height):
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                objects.append({
                    'class': class_id,
                    'bbox': [x1, y1, x2, y2],
                    'matched': False
                })
    return objects

def parse_bbox_from_csv(bbox_str):
    if isinstance(bbox_str, str):
        # print(bbox_str)
        bbox_str = bbox_str.strip('[]')
        bbox_list = [float(x.strip()) for x in bbox_str.split(',')]
        return bbox_list
    elif isinstance(bbox_str, list):
        return [float(x) for x in bbox_str]
    else:
        return bbox_str