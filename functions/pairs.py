import os
import pandas as pd
from parsing import parse_filename

def form_pairs(rgb_images, lwir_images, output_csv):
    rgb_files = sorted([f for f in os.listdir(rgb_images) if f.endswith(".jpg")])
    lwir_files = sorted([f for f in os.listdir(lwir_images) if f.endswith(".jpg")])

    lwir_dict = {}
    for lwir_f in lwir_files:
        sec_lwir, nsec_lwir = parse_filename(lwir_f)
        if sec_lwir not in lwir_dict:
            lwir_dict[sec_lwir] = []
        lwir_dict[sec_lwir].append((nsec_lwir, lwir_f))
    pairs = []
    for rgb_f in rgb_files:
        sec_rgb, nsec_rgb = parse_filename(rgb_f)
        if sec_rgb not in lwir_dict:
          print(f'no timestampsec {sec_rgb} in lwir')

        lwir_candidates = lwir_dict[sec_rgb]
        closest = min(lwir_candidates, key=lambda x: abs(x[0] - nsec_rgb))

        pairs.append({
                'timestampsec': sec_rgb,
                'rgb_filename': rgb_f,
                'rgb_timestampnanosec': nsec_rgb,
                'lwir_filename': closest[1],
                # 'lwir_timestampsec': sec_rgb,
                'lwir_timestampnanosec': closest[0]
            })

    df = pd.DataFrame(pairs)
    df.to_csv(output_csv, index=False)
    print(f"pairs count: {len(pairs)}")
    return df

def pairs_to_dict(pairs_df) -> dict:
    return {
        (row["timestampsec"], row["rgb_timestampnanosec"]): row["lwir_filename"]
        for _, row in pairs_df.iterrows()
    }

def get_lwir_paths(rgb_lowconfidence, pair_dict, lwir_images_path: str) -> list[str]:
    selected_lwir_files = {
        pair_dict[key]
        for r in rgb_lowconfidence
        if (key := (r["timestampsec"], r["timestampnsec"])) in pair_dict
    }
    return [os.path.join(lwir_images_path, f) for f in selected_lwir_files]