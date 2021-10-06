#!/usr/bin/env python3
"""
Generates hand landmarks for all class labels and outputs to a new data sub folder.

Uses media pipe: https://github.com/google/mediapipe

Run: python hand_landmarks.py -i path/to/data -o path/to/output [-d 'train' or 'val' or 'test']
e.g., python autsl_farmes.py -i "~/datasets/autsl" -o "~/datasets/autsl_hand_landmarks" -d "test"

This will create landmarks for all class labels under datasets/autsl/test,
assuming labels are folders (eg., datasets/autsl/test/word)

Output becomes datasets/autsl_hand_landmarks/test/word

Help: python autsl_frames.py -h
"""
from pathlib import Path
import argparse

################################################################################


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
    args = arg_parser.parse_args()
    validate_args(args)


if __name__ == '__main__':
    main()
