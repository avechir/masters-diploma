import torch
from ultralytics.utils.ops import scale_boxes

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0

    overlap_rgb = inter_area / box1_area * 100
    # overlap_lwir = inter_area / box2_area * 100
    # print(f"RGB overlap: {overlap_rgb:.1f}%")
    # print(f"LWIR overlap: {overlap_lwir:.1f}%")
    return inter_area / union_area, overlap_rgb

def is_in_common_view(rgb_box, rgb_img_size, lwir_img_size):
    rgb_w, rgb_h = rgb_img_size
    lwir_w, lwir_h = lwir_img_size

    scale = rgb_h / lwir_h
    lwir_w_scaled = lwir_w * scale

    x_min = (rgb_w - lwir_w_scaled) / 2
    x_max = rgb_w - x_min
    y_min = 0
    y_max = rgb_h

    x1, y1, x2, y2 = rgb_box
    if x2 < x_min or x1 > x_max or y2 < y_min or y1 > y_max:
        return False
    return True


def results_fusion(rgb_lowconfidence, lwir_results, pair_dict, 
                   rgb_img_size, lwir_img_size, iou_threshold=0.5, rgb_overlap_threshold=0.8, 
                   threshold_rgb=0.55, threshold_lwir=0.0):
    notanobjectlist = []

    for rgb_obj in rgb_lowconfidence:
        if rgb_obj['confidence'] >= threshold_rgb:
            continue
        # print('----------------------------------------------------------------')
        rgb_key = (rgb_obj["timestampsec"], rgb_obj["timestampnsec"])
        rgb_box = rgb_obj["bbox"]

        if not is_in_common_view(rgb_box, rgb_img_size, lwir_img_size):
            # print("box is not in view", rgb_box)
            continue

        # Якщо немає LWIR-пари для цього RGB-об'єкта
        if rgb_key not in pair_dict:
            notanobjectlist.append(rgb_obj)
            continue

        # Всі LWIR-об'єкти для цього файлу
        lwir_filename = pair_dict[rgb_key]
        lwir_objects_for_pair = [res for res in lwir_results if res["file"] == lwir_filename]

        if not lwir_objects_for_pair:
            notanobjectlist.append(rgb_obj)
            continue


        has_good_match = False

        for lwir_obj in lwir_objects_for_pair:
            # scale LWIR boxes to RGB boxes coordinates
            lwir_box_tensor = torch.tensor([lwir_obj["bbox"]])
            lwir_box_scaled = scale_boxes(
                img1_shape=lwir_img_size,
                boxes=lwir_box_tensor,
                img0_shape=rgb_img_size
            )[0].tolist()

            iou, rgb_overlap = calculate_iou(rgb_box, lwir_box_scaled)
            # print('rgb_obj: ',rgb_obj)
            # print('lwir_obj: ',lwir_obj)
            # print(iou, rgb_overlap)
            if iou >= iou_threshold or rgb_overlap >= rgb_overlap_threshold and lwir_obj['confidence'] > threshold_lwir:
                has_good_match = True
                # print('good match! - do not delete')
                break

        if not has_good_match:
            # print('no good match - delete')
            # visualize_checked_objects_on_rgb(rgb_obj, lwir_objects_for_pair, pair_dict, rgb_images_path, lwir_images_path)
            notanobjectlist.append(rgb_obj)
    
    return notanobjectlist