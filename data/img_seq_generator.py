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


################################################################################


class ImageSequenceDataGenerator(tf.keras.utils.Sequence):
    """
    Generates batches of image sequences representing video data.
    Preserves temporal order of frames.
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 input_path: str,
                 batch_size: int = 32,
                 input_size: Tuple[int, int, int] = (224, 224, 3),
                 shuffle: bool = True,
                 rescale=None):
        """
        Data:
        Expect dataframe to have columns: Sample, EN for sample id, and English class label
        """
        self.input_path = input_path
        self.df = dataframe.copy()
        self.num_samples = len(self.df)
        self.class_labels = self.df['EN'].unique()
        self.class_labels.sort()
        self.num_labels = len(self.class_labels)

        # Parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.rescale = rescale

        # Shuffle initial
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def on_epoch_end(self):
        """
        On epoch end shuffle dataset if shuffle flag is True.

        Shuffles dataframe containing sample names and class labels so that each
        batch will be different per epoch.
        """
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def shuffle_dataset_from_parent(self, df):
        """Simply replaces current dataframe with shuffled dataframe"""
        self.df = df.copy()

    def _encode_class_labels(self, batch: pd.DataFrame) -> np.ndarray:
        """
        Creates categorical one hot encoded label matrix for batch.

        Args:
            batch: (pd.DataFrame) Current batch of data.

        Returns:
            A binary matrix representation of the input.
            The classes axis is placed last. (shape=[batch, num_classes]
        """
        # labels_indices: Index in class_labels for this batch eleemnts labels
        labels_indices = [np.where(self.class_labels == label)[0] for label in batch['EN']]
        # One hot encoding, for example if batch[0] == garden, garden is idx 6 in class_labels
        # Then we get encoded [[0, 0, 0, 0, 0, 0, 1, 0, ..., 0], [...], ..., [...]]
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
        image_label_path = Path(self.input_path, label)
        image_paths = [str(x) for x in image_label_path.glob('*.jpg') if f'{sample}_' in x.name]
        # Load images
        sequence = []
        target_size = self.input_size[:2]
        for img in image_paths:
            with tfimage.load_img(img, target_size=target_size) as image:
                sequence.append(tfimage.img_to_array(image))
        # Pre-process images
        sequence = np.asarray(sequence)
        if self.rescale:
            sequence /= self.rescale

        # Padding
        # padding = np.zeros((self.max_sequence_size - sequence.shape[0], *self.input_size), dtype='float32')
        # sequence = np.concatenate((sequence, padding))
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
            img_seq = self._load_image_sequence(x, label.item())
            X.append(img_seq)

        # Ensure batch has the same length of timesteps with padding
        X_padded = []
        max_sequence = max(X, key=lambda x: x.shape[0])
        max_padding = max_sequence.shape[0]
        for seq in X:
            padding = np.zeros((max_padding - seq.shape[0], *self.input_size), dtype='float32')
            # if len(padding.shape) != len(seq.shape):
            #     print("Error seq and padding:")
            #     print(seq.shape)
            #     print(padding.shape)
            #     print(f"Input path: {self.input_path}")
            if len(seq.shape) != len(padding.shape):
                # Hotfix for no landmarks found!! So just make zeros for now. TODO: Remake data
                new_seq = np.zeros((max_padding, *self.input_size), dtype='float32')
            else:
                new_seq = np.concatenate((seq, padding))
            X_padded.append(new_seq)

        # Sign gloss/translations
        Y = self._encode_class_labels(batch)

        return np.asarray(X_padded), np.asarray(Y)

    def getitem(self, index):
        return self.__getitem__(index)


class MultiSequenceGenerator(tf.keras.utils.Sequence):
    """
    Combines multiple sequence generators.

    Note that all generators must have shuffle set to False as parent will control shuffle.
    """
    def __init__(self, df, generators, shuffle=True):
        self.df = df.copy()
        self.generators = generators
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_datasets()

    def shuffle_datasets(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        for gen in self.generators:
            gen.shuffle_dataset_from_parent(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_datasets()

    def __len__(self):
        """Length of all datasets should match first generator and be consistent"""
        return len(self.generators[0])

    def __getitem__(self, index: int):
        """
        Returns np.array of all generator X values, and the associated Y value for all.

        ie., [genx1, genx2, ...], geny1
        Args:
            index: index of batch

        Returns:
            [gen1_X, gen2_X, ..., genN_X], Y
        """
        # [(X1, Y1), (X2, Y2), ...]
        batch = [gen.getitem(index) for gen in self.generators]
        X = [X for (X, Y) in batch]
        # Ensure sequence lengths match, first X is always RGB sequence.
        max_seq_len = X[0].shape[1]
        for i, batched_seq in enumerate(X):
            if batched_seq.shape[1] < max_seq_len:
                # This batch needs padding
                new_batch = []
                for seq in batched_seq:
                    padding = np.zeros(
                        (max_seq_len - seq.shape[0], *self.generators[0].input_size),
                        dtype='float32')
                    new_seq = np.concatenate((seq, padding))
                    new_batch.append(new_seq)
                X[i] = np.asarray(new_batch)
        Y = batch[0][1]
        return X, Y


# # Bad testing :)
# PATH = r'path\data\autsl\frames_rgb_20\val'
# DEPTH_PATH = r'path\autsl\frames_depth_20\val'
# LANDMARKS_PATH = r'path\autsl\frames_landmarks_2_20\val'
# OPTICAL_RLOF_PATH = r'path\autsl\frames_optical_rlof_20\val'
# df = pd.read_csv(r'path\autsl\val_labels_20classes.csv')
# generator1 = ImageSequenceDataGenerator(df, PATH, shuffle=False)
# generator2 = ImageSequenceDataGenerator(df, DEPTH_PATH, shuffle=False)
# generator3 = ImageSequenceDataGenerator(df, OPTICAL_RLOF_PATH, shuffle=False)
# generator4 = ImageSequenceDataGenerator(df, LANDMARKS_PATH, shuffle=False)
# multi_gen = MultiSequenceGenerator(df, [generator1, generator2, generator3, generator4], shuffle=True)
# batch0 = multi_gen[0]
# # batch1 = multi_gen[1]
# # batchN = multi_gen[len(multi_gen)]
# import matplotlib.pyplot as plt
# seq_len = 5  # First 5
# fig, ax = plt.subplots(4, seq_len, figsize=(16, 9))
# scale = 255.
# offset = 0
# # batch0[0=X] [0-3] [batch item] [seq item] [channels...]
# for i in range(offset, seq_len+offset):
#     ax[0, i-offset].imshow(batch0[0][0][7][i]/scale)
#     ax[1, i-offset].imshow(batch0[0][1][7][i]/scale)
#     ax[2, i-offset].imshow(batch0[0][2][7][i]/scale)
#     ax[3, i-offset].imshow(batch0[0][3][7][i]/scale)
#
# for i in range(4):
#     for j in range(seq_len):
#         ax[i][j].get_yaxis().set_ticks([])
#         ax[i][j].get_xaxis().set_ticks([])
# plt.show()
# fig.savefig('fig2.png', dpi=300)
#
# print()
