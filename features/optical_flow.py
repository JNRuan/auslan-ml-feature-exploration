#!/usr/bin/env python3
"""
Generates optical flow for all class labels and outputs to a new data sub folder.

Run: python optical_flow.py -i path/to/data -o path/to/output [-d 'train' or 'val' or 'test']
e.g., python optical_flow.py -i "~/datasets/data" -o "~/datasets/data_optical_flow" -d "test"

This will create optical flow sequences for all class labels under datasets/data/test,
assuming labels are folders (eg., datasets/autsl/test/word)

Output becomes datasets/data_optical_flow/test/word

Adapted from: https://learnopencv.com/optical-flow-in-opencv/#dense-optical-flow
"""
from pathlib import Path
import argparse
import os

from tqdm import tqdm
import cv2
import numpy as np

################################################################################


def create_optical_flow(input: str, output: str, dataset: str, mode: str, resize: int, width: int, height: int):
    dataset_path = Path(input, dataset)
    output_path = Path(output, dataset)
    if not output_path.exists():
        os.makedirs(output_path)
    label_folders = [folder for folder in dataset_path.iterdir() if folder.is_dir()]
    print(f"Found: {len(label_folders)} class labels to process.")
    print(f"Optical flow mode: {mode}")

    for label_path in label_folders:
        print(f"Processing: {label_path.name}")
        process_optical_flow(output_path, label_path, label_path.name, str(mode), int(resize), int(width), int(height))


def process_optical_flow(output: Path, label_path: Path, label: str, mode: str, resize: int, w: int, h: int):
    image_out_path = Path(output, label)
    if not image_out_path.exists():
        os.makedirs(image_out_path)
    print(f"Label to {image_out_path}")
    image_list = [img for img in Path(label_path).glob('*.jpg')]
    pbar = tqdm(image_list)
    print(f"Found: {len(image_list)} images.")

    # Setup algorithm
    old_frame = cv2.imread(str(image_list[0]))
    if resize:
        old_frame = cv2.resize(old_frame, (w, h), interpolation=cv2.INTER_AREA)
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255
    if mode == 'farneback':
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Process sequence optical flow
    for img in pbar:
        new_frame = cv2.imread(str(img))
        if resize:
            new_frame = cv2.resize(new_frame, (w, h), interpolation=cv2.INTER_AREA)
        if mode == 'rlof':
            # Apply RLOF - does not need grayscale.
            flow = cv2.optflow.calcOpticalFlowDenseRLOF(old_frame, new_frame, None, *[])
        else:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            # Apply Farneback with default params
            flow = cv2.calcOpticalFlowFarneback(old_frame, new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Polar Coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Encode with HSV
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # HSV to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Save current frame and continue
        image_file_path = Path(image_out_path, img.name)
        cv2.imwrite(str(image_file_path), rgb)
        old_frame = new_frame


def validate_args(args):
    msg = ""
    if not Path(args.input).exists():
        msg = f"Input {args.input} does not exist"
    if not Path(args.input).is_dir():
        msg = f"Input {args.input} is not a folder."
    if not Path(args.output).exists():
        msg = f"Output {args.output} does not exist"
    if not Path(args.output).is_dir():
        msg = f"Output {args.output} is not a folder."
    if args.dataset not in ['train', 'val', 'test']:
        msg = f"Dataset {args.dataset} not valid, use 'train', 'val', or 'test'"
    if args.mode not in ['farneback', 'rlof']:
        msg = f"Mode {args.mode} not valid, only 'farneback' or 'rlof' is supported."
    if msg:
        raise ValueError(msg)


def main():
    arg_parser = argparse.ArgumentParser(description="Hand Landmark feature creation with media pipe")
    arg_parser.add_argument('-i',
                            '--input',
                            help='Dataset input root path, e.g., path/to/data',
                            required=True)
    arg_parser.add_argument('-o',
                            '--output',
                            help="Dataset output root path, e.g., path/to/output",
                            required=True)
    arg_parser.add_argument('-d',
                            '--dataset',
                            help='[Optional] Dataset to use, eg., train or val or test, default=train',
                            default='train')
    arg_parser.add_argument('-m',
                            '--mode',
                            help="Set optical flow mode, farneback || rlof",
                            default='farneback')
    arg_parser.add_argument('-r',
                            '--resize',
                            help="Set 1 if resize of images desired, good for inconsistent datasets.",
                            default=0)
    arg_parser.add_argument('-w',
                            '--width',
                            help="Set width for resize",
                            default=512)
    arg_parser.add_argument('-h',
                            '--height',
                            help="Set height for resize",
                            default=512)
    args = arg_parser.parse_args()
    validate_args(args)
    create_optical_flow(args.input, args.output, args.dataset, args.mode, args.resize, args.width, args.height)


if __name__ == '__main__':
    main()
