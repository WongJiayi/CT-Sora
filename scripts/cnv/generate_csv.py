import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import cv2


def get_video_info(path: str):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        aspect_ratio = height / width if width > 0 else np.nan
        resolution = height * width

        return {
            "path": path,
            "height": height,
            "width": width,
            "fps": fps,
            "num_frames": frames,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }

    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {
            "path": path,
            "height": np.nan,
            "width": np.nan,
            "fps": np.nan,
            "num_frames": np.nan,
            "aspect_ratio": np.nan,
            "resolution": np.nan,
        }


def process_row(path):
    return get_video_info(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path-only CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    paths = df["path"].tolist()

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_row, paths), total=len(paths)))

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)

    print(f"Saved CSV → {args.output} ({len(out_df)} items)")


if __name__ == "__main__":
    main()
