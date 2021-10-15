################################################################################
# Model building utils
################################################################################
from pathlib import Path
import tensorflow as tf

from data.img_seq_generator import ImageSequenceDataGenerator, MultiSequenceGenerator
################################################################################


def set_csv_callback(output_path, name: str):
    csv_path = Path(output_path, name)
    return tf.keras.callbacks.CSVLogger(csv_path)


def set_early_stop_callback(patience, monitor='val_loss'):
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)


def get_generator(df,
                  input_path: Path,
                  input_shape=(224, 224, 3),
                  batch_size=32,
                  shuffle=False):
    generator = ImageSequenceDataGenerator(df,
                                           str(input_path),
                                           batch_size=batch_size,
                                           input_size=input_shape,
                                           shuffle=shuffle)
    return generator


def get_multi_generators(df, input_paths, input_shape, batch_size, shuffle=True):
    generators = []
    for path in input_paths:
        gen = get_generator(df, path, input_shape, batch_size, shuffle=False)
        generators.append(gen)
    multi_gen = MultiSequenceGenerator(df, generators, shuffle=shuffle)
    return multi_gen
