################################################################################
# Train a single feature for n trials.
#
################################################################################
from pathlib import Path
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, \
    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6,EfficientNetB7
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, GlobalAveragePooling2D, \
    Input, LSTM, TimeDistributed
from tensorflow.keras import Model

import argparse
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ROOT_DIR = os.path.dirname(__file__)
import pandas as pd
import tensorflow as tf

from data.img_seq_generator import ImageSequenceDataGenerator

################################################################################


def get_data(df,
             input_path: Path,
             input_shape=(224, 224, 3),
             time_dist_shape=(None, 224, 224, 3),
             batch_size=32,
             shuffle=True):
    generator = ImageSequenceDataGenerator(df,
                                           str(input_path),
                                           batch_size=batch_size,
                                           input_size=input_shape,
                                           shuffle=shuffle)

    # Did not seem to improve.
    # def get_generator():
    #     for i in range(len(generator)):
    #         yield generator.getitem(i)
    #
    # dataset = tf.data.Dataset.from_generator(
    #     get_generator,
    #     output_signature=(
    #         tf.TensorSpec(shape=(None, *time_dist_shape), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None, 20), dtype=tf.float32)
    #     )
    # )
    return generator


def get_efficientnet(model_num=0, input_size=(224, 224, 3), finetune=False, tune_layers=3):
    model = None
    if model_num == 0:
        model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 1:
        model = EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 2:
        model = EfficientNetB2(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 3:
        model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 4:
        model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 5:
        model = EfficientNetB5(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 6:
        model = EfficientNetB6(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 7:
        model = EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_size)
    else:
        print("Model not found, ensure model number is in range [0, 7].")

    if not finetune:
        model.trainable = False
    else:
        for layer in model.layers[:-tune_layers]:
            layer.trainable = False
    return model


def single_feature_model(num_classes,
                         model_num=0,
                         input_shape=(224, 224, 3),
                         time_dist_shape=(None, 224, 224, 3),
                         dropout=0.5,
                         dense_n=128,
                         num_lstm=1,
                         lstm_n=256,
                         finetune=False,
                         tune_layers=3):
    efficient_net = get_efficientnet(model_num=0,
                                     input_size=input_shape,
                                     finetune=finetune,
                                     tune_layers=tune_layers)

    # Transfer Layers
    input_tensor = Input(shape=time_dist_shape)
    efficient_layer = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}")(input_tensor)
    # Use pooling layer to reduce number of parameters by 12x versus Flatten
    efficient_out = TimeDistributed(GlobalAveragePooling2D())(efficient_layer)

    # LSTM Sequence Layer
    if num_lstm == 2:
        lstm_start = LSTM(lstm_n, return_sequences=True)(efficient_out)
        lstm_out = LSTM(lstm_n)(lstm_start)
    else:
        lstm_out = LSTM(lstm_n)(efficient_out)
    fc = Dense(dense_n, activation='relu')(lstm_out)
    fc_out = Dropout(dropout)(fc)
    output = Dense(num_classes, activation='softmax')(fc_out)

    # Compile with Adam
    model = Model(input_tensor, output)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def set_csv_callback(output_path, name: str):
    csv_path = Path(output_path, name)
    return tf.keras.callbacks.CSVLogger(csv_path)


def set_early_stop_callback(patience, monitor='val_loss'):
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)


def run_trials(input_path: Path, output_path: Path, num_trials: int, trial_start: int, ef_net_model: int, config):
    output_runs_path = Path(output_path, "runs")
    input_train = Path(input_path, "train")
    input_val = Path(input_path, "val")
    labels_train = Path(input_path, config['trainLabels'])
    labels_val = Path(input_path, config['valLabels'])
    train_df = pd.read_csv(labels_train)
    val_df = pd.read_csv(labels_val)

    input_dim = config['inputDim']
    input_shape = (input_dim, input_dim, 3)
    time_dist_shape = (None, input_dim, input_dim, 3)
    num_classes = config['numClasses']
    batch_size = config['batchSize']
    enet_model = ef_net_model
    num_lstm = config['numLstm']
    lstm_units = config['lstm']
    dense_units = config['dense']
    dropout_value = config['dropout']
    max_epochs = config['maxEpochs']
    patience = config['patience']

    train_dataset = get_data(train_df, input_train, input_shape, time_dist_shape, batch_size, shuffle=True)
    val_dataset = get_data(val_df, input_val, input_shape, time_dist_shape, batch_size, shuffle=False)

    for i in range(num_trials):
        trial_num = i + trial_start
        print(f"Running trial {trial_num}, {i+1}/{num_trials}.")
        model = single_feature_model(num_classes,
                                     model_num=enet_model,
                                     input_shape=input_shape,
                                     time_dist_shape=time_dist_shape,
                                     dropout=dropout_value,
                                     dense_n=dense_units,
                                     num_lstm=num_lstm,
                                     lstm_n=lstm_units)

        run_log = f'training_rgb_{trial_num:03}.csv'
        run_model_summary = f'training_rgb_{trial_num:03}_model_summary.txt'
        model_checkpoint = f'training_rgb_{trial_num:03}.best.hdf5'

        with open(Path(output_runs_path, run_model_summary), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        early_stop_callback = set_early_stop_callback(patience)
        csv_callback = set_csv_callback(output_runs_path, run_log)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=Path(output_runs_path, model_checkpoint),
                                                        monitor='val_loss',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        mode='min')

        # Train
        rgb_train_history = model.fit(
            x=train_dataset,
            validation_data=val_dataset,
            epochs=max_epochs,
            callbacks=[early_stop_callback, csv_callback, checkpoint]
            # workers=8,
            # use_multiprocessing=False
        )
        print(f"Completed trial {trial_num}.")

    print(f"Completed {num_trials} trials.")


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
    if msg:
        raise ValueError(msg)


def main():
    arg_parser = argparse.ArgumentParser(description="Single feature trainer")
    arg_parser.add_argument('-i',
                            '--input',
                            help='Dataset input root path, e.g., path/to/data',
                            required=True)
    arg_parser.add_argument('-o',
                            '--output',
                            help="Dataset output root path, e.g., path/to/output",
                            required=True)
    arg_parser.add_argument('-n',
                            '--ntrials',
                            help="Number of trials to run",
                            default=5)
    arg_parser.add_argument('-t',
                            '--trial_start',
                            help="Starting number of trial filename, e.g., 1",
                            default=1)
    arg_parser.add_argument('-en',
                            '--ef_net',
                            help="Efficient net version, e.g. 0 fo B0, range [0, 7]",
                            default=0)
    arg_parser.add_argument('-c',
                            '--config',
                            help="Config file, default=config.json",
                            default="config.json")
    args = arg_parser.parse_args()

    validate_args(args)

    config_path = Path(ROOT_DIR, str(args.config))
    print(f"Config Path: {config_path}")
    with open(config_path) as json_file:
        config = json.load(json_file)
        print(f"Loaded config: {config}")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Set memory growth for GPU.")

    print(f"Running {args.ntrials} trials.")
    run_trials(Path(str(args.input)),
               Path(str(args.output)),
               int(args.ntrials),
               int(args.trial_start),
               int(args.ef_net),
               config
               )


if __name__ == '__main__':
    main()
