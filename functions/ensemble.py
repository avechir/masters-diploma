import torch
from ultralytics.utils.ops import scale_boxes
import numpy as np
from collections import defaultdict

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

    # overlap_rgb = inter_area / box1_area * 100
    # overlap_lwir = inter_area / box2_area * 100
    # print(f"RGB overlap: {overlap_rgb:.1f}%")
    # print(f"LWIR overlap: {overlap_lwir:.1f}%")
    return inter_area / union_area

def is_in_common_view(rgb_box, rgb_img_size, lwir_img_size):
    rgb_h, rgb_w = rgb_img_size
    lwir_h, lwir_w = lwir_img_size

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


def calculate_normalized_distance(box1, box2, img_size):
    img_h, img_w = img_size
    
    x1, y1, x2, y2 = box1
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    norm_cx1, norm_cy1 = cx1 / img_w, cy1 / img_h

    x1, y1, x2, y2 = box2
    cx2 = (x1 + x2) / 2
    cy2 = (y1 + y2) / 2
    norm_cx2, norm_cy2 = cx2 / img_w, cy2 / img_h

    distance = np.sqrt((norm_cx1 - norm_cx2)**2 + (norm_cy1 - norm_cy2)**2)
    
    return distance


def prepare_lwir_data(lwir_results, lwir_img_size, rgb_img_size):
    # Групує LWIR-детекції за файлом і масштабує під розмір RGB-зображення.
    lwir_by_file = defaultdict(list)

    for r in lwir_results:
        lwir_by_file[r['file']].append(r)

    for fname, objs in lwir_by_file.items():
        if not objs:
            continue
        boxes = torch.tensor([o['bbox'] for o in objs], dtype=torch.float32)
        scaled = scale_boxes(img1_shape=lwir_img_size, boxes=boxes, img0_shape=rgb_img_size)
        for o, s in zip(objs, scaled):
            o['_bbox_scaled'] = s.tolist()

    return lwir_by_file


def index_rgb_by_frame(rgb_detections):
    # Створює індекс RGB-детекцій за (timestampsec, timestampnsec).
    rgb_by_key = defaultdict(list)
    for obj in rgb_detections:
        key = (obj['timestampsec'], obj['timestampnsec'])
        rgb_by_key[key].append(obj)
    return rgb_by_key


def results_fusion_wr(rgb_lowconfidence, lwir_results, pair_dict, rgb_img_size, lwir_img_size, 
                      iou_threshold=0.5, threshold_rgb=0.426, distance_threshold=0.3, conf_fov_threshold=0.35
):
    
    # Підготовка LWIR та RGB індексів
    lwir_by_file = prepare_lwir_data(lwir_results, lwir_img_size, rgb_img_size)
    rgb_by_key = index_rgb_by_frame(rgb_lowconfidence)
    # print(lwir_by_file)
    # print(rgb_by_key)
    # print('-----------------------------')
    notanobjectlist = []

    # Обробка RGB-детекцій з низькою впевненістю
    for rgb_obj in rgb_lowconfidence:
      if rgb_obj.get('confidence', 0.0) < threshold_rgb:
        rgb_key = (rgb_obj["timestampsec"], rgb_obj["timestampnsec"])
        rgb_box = rgb_obj["bbox"]
        # print('')
        # print(f'rgb_obj {rgb_obj}')

        # Спільне поле зору
        if not is_in_common_view(rgb_box, rgb_img_size, lwir_img_size):
            if rgb_obj['confidence'] < conf_fov_threshold:
                # print('not fov, but low-conf')
                notanobjectlist.append(rgb_obj)
            # else: print('not fov, but is good')
            continue

        # Перевірка пари LWIR
        if rgb_key not in pair_dict:
            # print('no pair')
            notanobjectlist.append(rgb_obj)
            continue

        lwir_filename = pair_dict[rgb_key]
        lwir_objects = lwir_by_file.get(lwir_filename, [])
        # print(len(lwir_objects))
        if not lwir_objects:
            # print('no lwir detections')
            notanobjectlist.append(rgb_obj)
            continue

        # Пошук збігу з LWIR
        has_good_match = False
        for lwir_obj in lwir_objects:
            lwir_box_scaled = lwir_obj.get('_bbox_scaled')
            if lwir_box_scaled is None:
                # print('!!!no scaled box')
                continue

            iou = calculate_iou(rgb_box, lwir_box_scaled)
            distance = calculate_normalized_distance(rgb_box, lwir_box_scaled, rgb_img_size)
            # print(f'iou {iou}')
            if not (iou >= iou_threshold or distance <= distance_threshold):
                # print(f'not match by iou+ditance ({lwir_obj})')
                continue
            # print(f'match by iou+ditance ({lwir_obj})')
            # print('match! check if claimed...')

            # Перевірка, чи не зайнятий цей об'єкт LWIR більш впевненим RGB об'єктом
            claimed = False
            for neighbor in rgb_by_key[rgb_key]:
                
                if neighbor.get('confidence', 0.0) < threshold_rgb:
                    continue  
                # print(f'high-conf neighbor {neighbor}')
                n_iou = calculate_iou(neighbor['bbox'], lwir_box_scaled)
                n_dist = calculate_normalized_distance(neighbor['bbox'], lwir_box_scaled, rgb_img_size)

                if (n_dist < distance) or (n_iou > iou):
                    claimed = True
                    # print('it is not match...')
                    break
                # print('that neighbor is ok')

            if claimed:
                # print('claimed')
                continue  
            else:
                # print('no neighbors')
                # print(distance)
                # print('MATCHED!')
                has_good_match = True
                break

        if not has_good_match:
            # print('TO DELETE')
            notanobjectlist.append(rgb_obj)

    return notanobjectlist