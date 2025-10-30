from single_model_predict import predict_images
from parsing import parse_filename
import os

def find_empty_rgb_files(filtered_rgb_results, rgb_images_path):
    all_rgb_files = set(os.listdir(rgb_images_path))
    rgb_files_with_objects = set(res["file"] for res in filtered_rgb_results)
    empty_rgb_files = list(all_rgb_files - rgb_files_with_objects)
    print(f"Empty RGB detections: {len(empty_rgb_files)}")
    return empty_rgb_files


def detect_on_lwir_for_empty_rgb(empty_rgb_files, pair_dict, lwir_images_path, model_lwir, conf_for_empty=0.45):
    emptyimgs_lwir_paths = []
    for rgb_file in empty_rgb_files:
        ts_sec, ts_nsec = parse_filename(rgb_file)
        key = (ts_sec, ts_nsec)
        if key in pair_dict:
            lwir_file = pair_dict[key]
            emptyimgs_lwir_paths.append(os.path.join(lwir_images_path, lwir_file))
    if not emptyimgs_lwir_paths:
        return []
    results = predict_images(model_lwir, emptyimgs_lwir_paths, 'predictions_foremptyimgs_lwir.json', conf_model=conf_for_empty)
    print(f"Objects detected on LWIR: {len(results)}")
    return results


def convert_and_merge_results(filtered_rgb_results, lwir_results, pairs_df):
    lwir_converted = []
    for obj in lwir_results:
        match = pairs_df[pairs_df['lwir_filename'] == obj['file']]
        if not match.empty:
            rgb_file = match.iloc[0]['rgb_filename']
            obj_copy = obj.copy()
            obj_copy['file'] = rgb_file
            lwir_converted.append(obj_copy)
    final_results = filtered_rgb_results + lwir_converted
    print(f"Final total detections: {len(final_results)}")
    return final_results

def verify_empty_rgb_with_lwir(filtered_rgb_results, rgb_images_path, lwir_images_path, pair_dict, pairs_df, model_lwir, conf_for_empty=0.45):
    empty_rgb_files = find_empty_rgb_files(filtered_rgb_results, rgb_images_path)
    lwir_results = detect_on_lwir_for_empty_rgb(empty_rgb_files, pair_dict, lwir_images_path, model_lwir, conf_for_empty)
    final_results = convert_and_merge_results(filtered_rgb_results, lwir_results, pairs_df)
    return final_results
