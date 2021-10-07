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

################################################################################

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_drawing_styles._RADIUS = 2

CSV_HEADER = [
    'label',
    'sample',
    'image',
    'hand_index',
    'hand_name'
]

# Add all hand landmark names as headers.
for landmark in mp_hands.HandLandmark:
    CSV_HEADER.append(landmark.name)


def create_hand_landmarks(input: str, output: str, dataset: str):
    dataset_path = Path(input, dataset)
    output_path = Path(output, dataset)
    csv_path = Path(output_path, 'hand_landmarks.csv')
    if not output_path.exists():
        os.makedirs(output_path)

    if not csv_path.exists():
        with open(str(csv_path), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    label_folders = [folder for folder in dataset_path.iterdir() if folder.is_dir()]
    print(f"Found: {len(label_folders)} class labels to process.")

    for label_path in label_folders:
        print(f"Processing: {label_path.name}")
        process_hand_landmarks(str(output_path), str(label_path), label_path.name, str(csv_path))


def process_hand_landmarks(output: str, label_path: str, label_name: str, csv_path: str):
    image_list = [img for img in Path(label_path).glob('*.jpg')]
    pbar = tqdm(image_list)
    print(f"Found: {len(image_list)} images.")

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for file in pbar:
            # Mediapipe needs image flipped for processing
            image = cv2.flip(cv2.imread(str(file)), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue
            annotated_image = image.copy()

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Create image with landmark and draw new image
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                image_out_path = Path(output, label_name)
                if not image_out_path.exists():
                    os.makedirs(image_out_path)
                image_file_path = Path(image_out_path, file.name)
                cv2.imwrite(str(image_file_path), cv2.flip(annotated_image, 1))

                # Handle landmarks raw csv data
                landmarks_row = [label_name, "_".join(file.name.split('_')[:2]), file.name]
                hand_index, hand_label = get_hand_label(idx, results)
                landmarks_row.append(hand_index)
                landmarks_row.append(hand_label)
                for landmark_type in mp_hands.HandLandmark:
                    x = hand_landmarks.landmark[landmark_type].x
                    y = hand_landmarks.landmark[landmark_type].y
                    z = hand_landmarks.landmark[landmark_type].z
                    landmarks_row.append(f"{x},{y},{z}")

                with open(csv_path, 'a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks_row)


def get_hand_label(hand_index: int, results):
    for classification in results.multi_handedness:
        if classification.classification[0].index == hand_index:
            return classification.classification[0].index, classification.classification[0].label
    return -1, 'hand'

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
    create_hand_landmarks(args.input, args.output, args.dataset)


if __name__ == '__main__':
    main()
