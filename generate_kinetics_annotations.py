import argparse
import pandas as pd
from glob import glob



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_videos", type=str)
    parser.add_argument("--data_fraction", type=float)
    parser.add_argument("--outfile", type=str)
    args = parser.parse_args()
    return args

def generate_kinetics_annotation_file(path_to_videos, data_fraction, outfile):
    videos = glob(f'{path_to_videos}/**/*.mp4', recursive=True)
    df = pd.DataFrame(videos, columns=['videos'])
    df['label_index'] = 9 # Dummy label for SS training
    index = round(len(df) * data_fraction)
    df = df.sample(index)
    df.to_csv(outfile, sep=' ', header=None, index=False)

def main():

    args = argparser()
    generate_kinetics_annotation_file(
        path_to_videos=args.path_to_videos, 
        data_fraction=args.data_fraction,
        outfile=args.outfile
    )
    
if __name__ == "__main__":
    main()