import mmcv
import json
import argparse
from tqdm import tqdm
from glob import glob


def scale_bboxes(path_to_annotation, bbox_scaler, width=720, height=404):

    # Open annotation
    with open(path_to_annotation, "rb") as handle:
        annotation_dict = json.load(handle)

    x_min_lim = 0
    y_min_lim = 0

    x_max_lim = width
    y_max_lim = height

    for ann in annotation_dict["annotations"]:
        for d in ann["detections"]:
            xmin, ymin, xmax, ymax = (
                d["bbox"][0],
                d["bbox"][1],
                d["bbox"][2],
                d["bbox"][3],
            )

            xmin -= bbox_scaler * (xmax - xmin)
            xmax += bbox_scaler * (xmax - xmin)
            ymin -= bbox_scaler * (ymax - ymin)
            ymax += bbox_scaler * (ymax - ymin)

            xmin = x_min_lim if (xmin < x_min_lim) else xmin
            ymin = y_min_lim if (ymin < y_min_lim) else ymin

            xmax = x_max_lim if (xmax > x_max_lim) else xmax
            ymax = y_max_lim if (ymax > y_max_lim) else ymax

            d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3] = (
                xmin,
                ymin,
                xmax,
                ymax,
            )
    return annotation_dict


def match_files(a, v):
    if a.split("/")[-1].split(".")[0] == v.split("/")[-1].split(".")[0]:
        return True
    return False


def scale_annotations(annotation_path, video_path, outpath, bbox_scaler):
    annotations = glob(f"{annotation_path}/**/*.json", recursive=True)
    videos = glob(f"{video_path}/**/*.mp4", recursive=True)
    for a in tqdm(annotations):
        for v in videos:
            if match_files(a, v):
                # Extract file name (without path)
                name = a.split("/")[-1].split(".")[0]
                video = mmcv.VideoReader(v)
                width, height = video.width, video.height

                # Scale all bbboxes
                scaled_annotation = scale_bboxes(
                    path_to_annotation=a,
                    bbox_scaler=bbox_scaler,
                    width=width,
                    height=height,
                )

                with open(f"{outpath}/{name}.json", "w") as handle:
                    json.dump(scaled_annotation, handle, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    scale_annotations(
        annotation_path=args.annotation_path,
        video_path=args.video_path,
        outpath=args.outpath,
        bbox_scaler=args.bbox_scaler,
    )


def parse_args():

    # /home/dl18206/Desktop/phd/data/panaf/translation/core_video_translation.csv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_path", type=str, required=True, help="Path to annotations dir"
    )
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--outpath", type=str)
    parser.add_argument("--bbox_scaler", type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
