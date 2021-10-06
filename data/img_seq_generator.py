#!/usr/bin/env python3
"""
Generates batches of frames keeping temporal ordering in sequence.
"""
from pathlib import Path, PurePath
from typing import Tuple
from timeit import default_timer as timer

# Lib
from tensorflow.keras.preprocessing import image as tfimage
import numpy as np
import pandas as pd
import tensorflow as tf

# Module Imports
from data_utils import get_image_files_recursively, max_sequence

################################################################################


class ImageSequenceDataGenerator(tf.keras.utils.Sequence):
    """
    Generates batches of image sequences representing video data.
    Preserves temporal order of frames.
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 input_path: str,
                 max_seq_len: int = 50,
                 batch_size: int = 32,
                 input_size: Tuple[int, int, int] = (224, 224, 3),
                 shuffle: bool = True):
        """
        Data:
        Expect dataframe to have columns: Sample, EN for sample id, and English class label
        """
        self.input_path = input_path
        self.df = dataframe.copy()
        self.num_samples = len(df)
        self.class_labels = df['EN'].unique()
        self.class_labels.sort()
        self.num_labels = len(self.class_labels)

        # Parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.max_sequence_size = max_seq_len
        self.shuffle = shuffle

    def on_epoch_end(self):
        """
        On epoch end shuffle dataset if shuffle flag is True.

        Shuffles dataframe containing sample names and class labels so that each
        batch will be different per epoch.
        """
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def _encode_class_labels(self, batch: pd.DataFrame) -> np.ndarray:
        """
        Creates categorical one hot encoded label matrix for batch.

        Args:
            batch: (pd.DataFrame) Current batch of data.

        Returns:
            A binary matrix representation of the input.
            The classes axis is placed last. (shape=[batch, num_classes]
        """
        labels_indices = [np.where(self.class_labels == label)[0] for label in batch['EN']]
        return tf.keras.utils.to_categorical(labels_indices, num_classes=self.num_labels)

    def _load_image_sequence(self, sample: str, label: str) -> np.ndarray:
        """
        Load an image sequence for a sample.

        This function will also resize to a target size based on self.input_size.

        Preprocessing:
        - Normalise by img / 255.
        - Pad sequence with zeroes up to self.max_seq_len

        Args:
            sample: (str) Sample name to collect sequence of images for.
            label: (str) Class label for this sample.

        Returns:
            (ndarray) Sequence of images with shape [self.max_seq_len, *self.input_size]
        """
        image_label_path = PurePath(self.input_path, label)
        image_paths = [str(x) for x in Path(image_label_path).glob('*.jpg') if f'{sample}_' in x.name]
        # Load images
        sequence = []
        target_size = self.input_size[:2]
        for img in image_paths:
            with tfimage.load_img(img, target_size=target_size) as image:
                sequence.append(tfimage.img_to_array(image))
        # Pre-process images
        sequence = np.asarray(sequence) / 255.

        # Padding
        padding = np.zeros((self.max_sequence_size - sequence.shape[0], *self.input_size), dtype='float32')
        sequence = np.concatenate((sequence, padding))
        return sequence

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """
        Get batch[index]. With shape: [batch, n, input_size]

        Labels are one hot encoded with shape [batch, n_labels]

        Args:
            index: i'th batch to retrieve.

        Returns:
            (X, Y): Tuple of numpy arrays of (Sequences, Labels)
        """
        # Slice df for current batch
        batch_start = index * self.batch_size
        batch_end = (index + 1) * self.batch_size
        batch = self.df[batch_start:batch_end]

        # Process data for this batch to get X data, Y labels
        X = []
        for x in batch['Sample']:
            label = batch.loc[batch['Sample'] == x]['EN']
            X.append(self._load_image_sequence(x, label.item()))
        Y = self._encode_class_labels(batch)

        return np.asarray(X), np.asarray(Y)
