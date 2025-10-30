from collections import defaultdict
from parsing import parse_yolo_label, parse_bbox_from_csv
from ensemble import calculate_iou
import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def match_predictions_to_gt(predictions, ground_truth_dir, img_size, iou_threshold=0.5):
    matches = []
    pred_by_file = defaultdict(list)
    for pred in predictions:
        pred_by_file[pred['file']].append(pred.copy())

    all_gt_files = [
        f.replace('.txt', '.jpg')
        for f in os.listdir(ground_truth_dir)
        if f.endswith('.txt')
    ]
    for filename in all_gt_files:
        # ground truth by file
        label_path = os.path.join(ground_truth_dir, filename.replace('.jpg', '.txt'))
        gt_objects = parse_yolo_label(label_path, img_size[1], img_size[0])
        for gt in gt_objects:
            gt['matched'] = False

        file_preds = pred_by_file.get(filename, [])

        # GT matching
        for pred in file_preds:
            best_match = None
            best_iou = 0

            for i, gt in enumerate(gt_objects):
                if gt['class'] != pred['class'] or gt['matched']:
                    continue

                iou, _ = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = i

            if best_match is not None:
                # True Positive
                gt_objects[best_match]['matched'] = True
                matches.append({
                    'file': filename,
                    'pred_class': pred['class'],
                    'gt_class': gt_objects[best_match]['class'],
                    'confidence': pred['confidence'],
                    'iou': best_iou,
                    'type': 'TP',
                    'pred_bbox': pred['bbox'],
                    'gt_bbox': gt_objects[best_match]['bbox']
                })
            else:
                # False Positive
                matches.append({
                    'file': filename,
                    'pred_class': pred['class'],
                    'gt_class': -1,
                    'confidence': pred['confidence'],
                    'iou': 0,
                    'type': 'FP',
                    'pred_bbox': pred['bbox'],
                    'gt_bbox': None
                })

        # False Negatives (усі unmatched GT)
        for gt in gt_objects:
            if not gt['matched']:
                matches.append({
                    'file': filename,
                    'pred_class': -1,
                    'gt_class': gt['class'],
                    'confidence': 0,
                    'iou': 0,
                    'type': 'FN',
                    'pred_bbox': None,
                    'gt_bbox': gt['bbox']
                })

    return matches

def calculate_metrics(matches, class_names=None):
    all_classes = set()
    for match in matches:
        if match['pred_class'] != -1:
            all_classes.add(match['pred_class'])
        if match['gt_class'] != -1:
            all_classes.add(match['gt_class'])

    all_classes = sorted(list(all_classes))

    if class_names is None:
        class_names = [f'Class_{i}' for i in all_classes]

    tp_per_class = {cls: 0 for cls in all_classes}
    fp_per_class = {cls: 0 for cls in all_classes}
    fn_per_class = {cls: 0 for cls in all_classes}

    for match in matches:
        if match['type'] == 'TP':
            tp_per_class[match['pred_class']] += 1
        elif match['type'] == 'FP':
            fp_per_class[match['pred_class']] += 1
        elif match['type'] == 'FN':
            fn_per_class[match['gt_class']] += 1

    metrics_per_class = {}

    for cls in all_classes:
        tp = tp_per_class[cls]
        fp = fp_per_class[cls]
        fn = fn_per_class[cls]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_per_class[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            # 'support': tp + fn  
        }


    # Overall metrics
    total_tp = sum(tp_per_class.values())
    total_fp = sum(fp_per_class.values())
    total_fn = sum(fn_per_class.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    total_predictions = total_tp + total_fp + total_fn
    accuracy = total_tp / total_predictions if total_predictions > 0 else 0
    return {
        'per_class': metrics_per_class,
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'accuracy': accuracy,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        },
        'classes': all_classes,
        'class_names': class_names
    }

def print_evaluation_results(metrics):
    overall = metrics['overall']
    print(f"\nmetrics:")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1-Score:  {overall['f1']:.4f}")
    print(f"Accuracy:  {overall['accuracy']:.4f}")
    print(f"\nCounts:")
    print(f"True Positives:  {overall['total_tp']}")
    print(f"False Positives: {overall['total_fp']}")
    print(f"False Negatives: {overall['total_fn']}")

    print("\nPer-class metrics:")
    for cls in metrics['classes']:
        cls_metrics = metrics['per_class'][cls]
        class_name = metrics['class_names'][cls] if cls < len(metrics['class_names']) else f'Class_{cls}'
        print(f"Class: {class_name}")
        print(f"   Precision: {cls_metrics['precision']:.4f}")
        print(f"   Recall: {cls_metrics['recall']:.4f}")
        print(f"   TP: {cls_metrics['tp']}")
        print(f"   FN: {cls_metrics['fn']}")
        print(f"   FP: {cls_metrics['fp']}")

def evaluate_results(predictions_file, labels_dir, img_size, iou_threshold=0.5, class_names=None):
    if predictions_file.endswith('.json'):
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
    elif predictions_file.endswith('.csv'):
        df = pd.read_csv(predictions_file)
        predictions = df.to_dict('records')
        for pred in predictions:
            pred['bbox'] = parse_bbox_from_csv(pred['bbox'])

    matches = match_predictions_to_gt(predictions, labels_dir, img_size, iou_threshold)
    metrics = calculate_metrics(matches, class_names)
    print_evaluation_results(metrics)

    return metrics, matches
