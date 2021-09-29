#!/usr/bin/env python3
"""
This script extracts AUTSL videos into frames.

Dataset: http://cvml.ankara.edu.tr/datasets/

Note that the dataset image size is 512x512, may require preprocessing to resize images prior to
consumption or batching.

Run: python autsl_frames.py -i path/to/autsl -o path/to/autsl/frames [-d 'train' or 'val' or 'test']
e.g., python autsl_farmes.py -i "~/datasets/autsl" -o "~/datasets/autsl/frames" -d "test"

Help: python autsl_frames.py -h
"""
# Standard
from enum import Enum
from pathlib import Path, PurePath

import argparse
import glob
import os
import subprocess

# Packages
from tqdm import tqdm

import pandas as pd

################################################################################
GLOB_PATTERN_RGB = '*_color.mp4'
GLOB_PATTERN_DEPTH = '*_depth.mp4'


class CameraMode(Enum):
    RGB = 0
    DEPTH = 1


CAMERA_MODE_MAP = {
    CameraMode.RGB: 'rgb',
    CameraMode.DEPTH: 'depth'
}


def construct_frames_dataset(input_path: str,
                             output_path: str,
                             dataset_cat: str,
                             fps: str,
                             mode: str):
    """
    Constructs rgb frame dataset from videos in input_path/dataset_cat. E.g., input_path/train.

    Output path will save frames in output_path/rgb/dataset_cat/{word}/. E.g., output_path/rgb/train/{word}/
    Each {word} folder will contain frames related to {word}, with video name consistent.

    Args:
        input_path: Dataset root path containing train, val, test folders.
        output_path: Output path to save frames.
        dataset_cat: Dataset category to convert to frames.
        fps: FPS setting for ffmpeg.
        mode: Camera mode of videos, either RGB or Depth
    """
    # Get all videos from input_path/dataset_cat
    data_path = PurePath(input_path, dataset_cat)
    if mode == CameraMode.DEPTH:
        video_paths = glob.glob(f"{PurePath(data_path, GLOB_PATTERN_DEPTH)}")
        out_path = PurePath(output_path, 'depth', dataset_cat)
    else:
        video_paths = glob.glob(f"{PurePath(data_path, GLOB_PATTERN_RGB)}")
        out_path = PurePath(output_path, 'rgb', dataset_cat)
    video_filenames = [os.path.split(p)[1] for p in video_paths]
    sample_names = ["_".join(f.split("_")[:2]) for f in video_filenames]
    class_labels = pd.read_csv(PurePath(input_path, f"{dataset_cat}_labels_en.csv"))

    print(f"Found {len(video_paths)} videos in {data_path}")

    # Setup paths
    if not Path(out_path).exists():
        os.makedirs(out_path)

    # Extract frames
    pbar = tqdm(video_paths)
    for idx, item in enumerate(pbar):
        pbar.set_description(f"Processing: {item}")
        print()

        # Setup class label and path
        class_label = class_labels.loc[class_labels['Sample'] == sample_names[idx]]['EN'].item()
        class_label_path = PurePath(out_path, class_label)
        if not Path(class_label_path).exists():
            os.makedirs(class_label_path)

        # Setup filename
        frame_pattern = f"{sample_names[idx]}_%03d.jpg"
        frame_path = PurePath(class_label_path, frame_pattern)

        # Call ffmpeg subprocess to make frames
        subprocess.call([
            'ffmpeg',
            '-i', item,
            '-qscale:v', '2',
            '-vf', f'fps={fps}',
            '-loglevel', '0',
            frame_path
        ], shell=True)


def validate_args(args):
    if not Path(args.input).exists():
        msg = f"Input {args.input} does not exist"
        raise ValueError(msg)
    if not Path(args.input).is_dir():
        msg = f"Input {args.input} is not a folder."
        raise ValueError(msg)
    if not Path(args.output).exists():
        msg = f"Output {args.output} does not exist"
        raise ValueError(msg)
    if not Path(args.output).is_dir():
        msg = f"Output {args.output} is not a folder."
        raise ValueError(msg)
    if args.dataset not in ['train', 'val', 'test']:
        msg = f"Dataset {args.dataset} not valid, use 'train', 'val', or 'test'"
        raise ValueError(msg)


def main():
    arg_parser = argparse.ArgumentParser(description="AUTSL Frames Data Creation")
    arg_parser.add_argument('-i',
                            '--input',
                            help='Dataset input root path, e.g., path/to/autsl',
                            required=True)
    arg_parser.add_argument('-o',
                            '--output',
                            help="Dataset output root path, e.g., path/to/autsl/frames",
                            required=True)
    arg_parser.add_argument('-d',
                            '--dataset',
                            help='[Optional] Dataset to use, eg., train or val or test, default=train',
                            default='train')
    arg_parser.add_argument('-f',
                            '--fps',
                            help='[Optional] FPS to extract at, default=30',
                            default='30')
    arg_parser.add_argument('-m',
                            '--mode',
                            help='[Optional] Camera mode of videos. rgb=0, depth=1, default=0',
                            default='0')
    args = arg_parser.parse_args()

    validate_args(args)
    construct_frames_dataset(args.input, args.output, args.dataset, args.fps, args.mode)


if __name__ == '__main__':
    main()
