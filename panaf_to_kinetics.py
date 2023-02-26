import mmcv
import torch
import torchvision
import argparse
from tqdm import tqdm
from glob import glob
from einops import rearrange
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from panaf.datasets import SSLKineticsProcessing


def tensor2video(x, video_name, ape_id, frame_idx, outpath, fps: int = 24):

    outfile = f"{video_name}_f{frame_idx}_ape_{ape_id}.mp4"

    # Input tensor shape: BxTxCxWxH
    x = x.permute(0, 2, 1, 3, 4)[0].detach().cpu()
    min_val = x.min()
    max_val = x.max()
    x = ((x - min_val) * 255 / (max_val - min_val)).to(torch.uint8)
    x = torch.permute(x, (1, 2, 3, 0))

    # Write tensor to video
    torchvision.io.write_video(filename=f"{outpath}/{outfile}", video_array=x, fps=fps)


def get_dataset(
    data_path: str = None,
    ann_path: str = None,
    sequence_len: int = 96,
    sample_itvl: int = 1,
    stride: int = 96,
    behaviour_threshold: int = 96,
    resize: tuple = (256, 320),
):

    # Define basic transforms with short side scaling
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(resize)]  # 256, 320
    )

    dataset = SSLKineticsProcessing(
        data_dir=data_path,
        ann_dir=ann_path,
        sequence_len=sequence_len,
        sample_itvl=sample_itvl,
        stride=stride,
        spatial_transform=transform,
        temporal_transform=transform,
        behaviour_threshold=behaviour_threshold,
        type="r",
    )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data dir")
    parser.add_argument("--ann_path", type=str, required=True, help="Path to ann dir")
    parser.add_argument(
        "--outdir", type=str, required=True, help="Where to write cropped videos to"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset = get_dataset(data_path=args.data_path, ann_path=args.ann_path)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    # Generate cropped train videos
    for x, video_name, ape_id, frame_idx in tqdm(loader):
        tensor2video(
            x,
            next(iter(video_name)),
            ape_id.item(),
            frame_idx.item(),
            outpath=args.outdir,
        )


if __name__ == "__main__":
    main()
