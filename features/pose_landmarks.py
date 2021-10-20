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

Code adapted from: https://google.github.io/mediapipe/solutions/hands.html
"""
from pathlib import Path, PurePath
import argparse
import csv
import os

from tqdm import tqdm
import cv2
import mediapipe as mp
import numpy as np

################################################################################

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_drawing_styles._RADIUS = 2
mp_holistic = mp.solutions.holistic

# CSV_HEADER = [
#     'label',
#     'sample',
#     'image',
#     'hand_index',
#     'hand_name'
# ]
#
# # Add all hand landmark names as headers.
# for landmark in mp_hands.HandLandmark:
#     CSV_HEADER.append(landmark.name)


def create_landmarks(input: str, output: str, dataset: str, mode: int):
    dataset_path = Path(input, dataset)
    output_path = Path(output, dataset)
    # csv_path = Path(output_path, 'hand_landmarks.csv')
    if not output_path.exists():
        os.makedirs(output_path)

    # if not csv_path.exists():
    #     with open(str(csv_path), 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(CSV_HEADER)

    label_folders = [folder for folder in dataset_path.iterdir() if folder.is_dir()]
    print(f"Found: {len(label_folders)} class labels to process.")

    for label_path in label_folders:
        print(f"Processing: {label_path.name}")
        process_landmarks(str(output_path), str(label_path), label_path.name, mode)


def process_landmarks(output: str, label_path: str, label_name: str, mode: int):
    image_list = [img for img in Path(label_path).glob('*.jpg')]
    image_out_path = Path(output, label_name)
    if not image_out_path.exists():
        os.makedirs(image_out_path)
    pbar = tqdm(image_list)
    print(f"Found: {len(image_list)} images.")

    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2) as holistic:
        for file in pbar:
            image_file_path = Path(image_out_path, file.name)
            # Mediapipe needs image flipped for processing
            image = cv2.imread(str(file))
            image_height, image_width, _ = image.shape
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks \
                    and not results.left_hand_landmarks \
                    and not results.right_hand_landmarks:
                # Write OG image anyway.
                if mode:
                    placeholder = np.zeros(image.shape)
                    cv2.imwrite(str(image_file_path), placeholder)
                else:
                    cv2.imwrite(str(image_file_path), cv2.flip(image, 1))
                continue

            if mode:
                annotated_image = np.zeros(image.shape)
            else:
                annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            cv2.imwrite(str(image_file_path), annotated_image)


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
    arg_parser.add_argument('-m',
                            '--mode',
                            help='Default = 0, mode 0 to include image, mode 1 for only landmarks on image.',
                            default=0)
    args = arg_parser.parse_args()
    validate_args(args)
    create_landmarks(args.input, args.output, args.dataset, args.mode)


if __name__ == '__main__':
    main()
