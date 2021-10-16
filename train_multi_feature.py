################################################################################
# Trains multiple features
# TODO: This code could be made more generic to limit code reuse.
################################################################################
from pathlib import Path
from tensorflow.keras.layers import concatenate, Dense, Dropout, Flatten, GlobalAveragePooling2D, \
    Input, LSTM, TimeDistributed
from tensorflow.keras import Model

import argparse
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ROOT_DIR = os.path.dirname(__file__)
import pandas as pd
import tensorflow as tf

from data.img_seq_generator import ImageSequenceDataGenerator, MultiSequenceGenerator
from models.efficient_net import get_efficientnet
from models.model_utils import get_generator, get_multi_generators, set_csv_callback, set_early_stop_callback


################################################################################


def multi_rgb_depth_model(num_classes,
                          model_num=0,
                          input_shape=(224, 224, 3),
                          time_dist_shape=(None, 224, 224, 3),
                          dropout=0.5,
                          dense_n=128,
                          num_lstm=1,
                          lstm_n=256,
                          finetune=False,
                          tune_layers=3):
    """
    Creates an RGB and Depth model. Could be a generic function but not a priority.
    """
    efficient_net = get_efficientnet(model_num=0,
                                     input_size=input_shape,
                                     finetune=finetune,
                                     tune_layers=tune_layers)

    # Transfer Layers
    # Block 1 - RGB
    input_tensor1 = Input(shape=time_dist_shape, name="RGB Input")
    efficient_layer1 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_RGB")(input_tensor1)
    # Use pooling layer to reduce number of parameters by 12x versus Flatten
    efficient_out1 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer1)

    # Block 2 - DEPTH
    input_tensor2 = Input(shape=time_dist_shape, name="DEPTH Input")
    efficient_layer2 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_DEPTH")(input_tensor2)
    efficient_out2 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer2)

    # MERGE
    merged_out = concatenate([efficient_out1, efficient_out2])

    # LSTM Sequence Layer
    if num_lstm == 2:
        lstm_start = LSTM(lstm_n, return_sequences=True)(merged_out)
        lstm_out = LSTM(lstm_n)(lstm_start)
    else:
        lstm_out = LSTM(lstm_n)(merged_out)
    fc = Dense(dense_n, activation='relu')(lstm_out)
    fc_out = Dropout(dropout)(fc)
    output = Dense(num_classes, activation='softmax')(fc_out)

    # Compile with Adam
    model = Model([input_tensor1, input_tensor2], output)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_four_model(num_classes,
                     input_names,
                     model_num=0,
                     input_shape=(224, 224, 3),
                     time_dist_shape=(None, 224, 224, 3),
                     dropout=0.5,
                     dense_n=128,
                     num_lstm=1,
                     lstm_n=256,
                     finetune=False,
                     tune_layers=3):
    # Only using as inference here
    efficient_net = get_efficientnet(model_num=0,
                                     input_size=input_shape,
                                     finetune=finetune,
                                     tune_layers=tune_layers)

    # Blocks: RGB, Depth, Optical Flow, Landmarks
    input_tensor0 = Input(shape=time_dist_shape, name=f"{input_names[0]} Input")
    efficient_layer0 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[0]}")(input_tensor0)
    # Use pooling layer to reduce number of parameters by 12x versus Flatten
    out_0 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer0)

    input_tensor1 = Input(shape=time_dist_shape, name=f"{input_names[1]} Input")
    efficient_layer1 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[1]}")(input_tensor1)
    out_1 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer1)

    input_tensor2 = Input(shape=time_dist_shape, name=f"{input_names[2]} Input")
    efficient_layer2 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[2]}")(input_tensor2)
    out_2 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer2)

    input_tensor3 = Input(shape=time_dist_shape, name=f"{input_names[3]} Input")
    efficient_layer3 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[3]}")(input_tensor3)
    out_3 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer3)

    # MERGE
    merged_out = concatenate([out_0, out_1, out_2, out_3])

    # LSTM Sequence Layer
    if num_lstm == 2:
        lstm_start = LSTM(lstm_n, return_sequences=True)(merged_out)
        lstm_out = LSTM(lstm_n)(lstm_start)
    else:
        lstm_out = LSTM(lstm_n)(merged_out)
    fc = Dense(dense_n, activation='relu')(lstm_out)
    fc_out = Dropout(dropout)(fc)
    output = Dense(num_classes, activation='softmax')(fc_out)

    # Compile with Adam
    model = Model([input_tensor0, input_tensor1, input_tensor2, input_tensor3], output)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_three_model(num_classes,
                      input_names,
                      model_num=0,
                      input_shape=(224, 224, 3),
                      time_dist_shape=(None, 224, 224, 3),
                      dropout=0.5,
                      dense_n=128,
                      num_lstm=1,
                      lstm_n=256,
                      finetune=False,
                      tune_layers=3):
    # Only using as inference here
    efficient_net = get_efficientnet(model_num=0,
                                     input_size=input_shape,
                                     finetune=finetune,
                                     tune_layers=tune_layers)

    # 3 Input Blocks
    input_tensor0 = Input(shape=time_dist_shape, name=f"{input_names[0]} Input")
    efficient_layer0 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[0]}")(input_tensor0)
    # Use pooling layer to reduce number of parameters by 12x versus Flatten
    out_0 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer0)

    input_tensor1 = Input(shape=time_dist_shape, name=f"{input_names[1]} Input")
    efficient_layer1 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[1]}")(input_tensor1)
    out_1 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer1)

    input_tensor2 = Input(shape=time_dist_shape, name=f"{input_names[2]} Input")
    efficient_layer2 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[2]}")(input_tensor2)
    out_2 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer2)

    # MERGE
    merged_out = concatenate([out_0, out_1, out_2])

    # LSTM Sequence Layer
    if num_lstm == 2:
        lstm_start = LSTM(lstm_n, return_sequences=True)(merged_out)
        lstm_out = LSTM(lstm_n)(lstm_start)
    else:
        lstm_out = LSTM(lstm_n)(merged_out)
    fc = Dense(dense_n, activation='relu')(lstm_out)
    fc_out = Dropout(dropout)(fc)
    output = Dense(num_classes, activation='softmax')(fc_out)

    # Compile with Adam
    model = Model([input_tensor0, input_tensor1, input_tensor2], output)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_two_model(num_classes,
                    input_names,
                    model_num=0,
                    input_shape=(224, 224, 3),
                    time_dist_shape=(None, 224, 224, 3),
                    dropout=0.5,
                    dense_n=128,
                    num_lstm=1,
                    lstm_n=256,
                    finetune=False,
                    tune_layers=3):
    # Only using as inference here
    efficient_net = get_efficientnet(model_num=0,
                                     input_size=input_shape,
                                     finetune=finetune,
                                     tune_layers=tune_layers)

    # 3 Input Blocks
    input_tensor0 = Input(shape=time_dist_shape, name=f"{input_names[0]} Input")
    efficient_layer0 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[0]}")(input_tensor0)
    # Use pooling layer to reduce number of parameters by 12x versus Flatten
    out_0 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer0)

    input_tensor1 = Input(shape=time_dist_shape, name=f"{input_names[1]} Input")
    efficient_layer1 = TimeDistributed(efficient_net, name=f"EfficientNetB{model_num}_{input_names[1]}")(input_tensor1)
    out_1 = TimeDistributed(GlobalAveragePooling2D())(efficient_layer1)

    # MERGE
    merged_out = concatenate([out_0, out_1])

    # LSTM Sequence Layer
    if num_lstm == 2:
        lstm_start = LSTM(lstm_n, return_sequences=True)(merged_out)
        lstm_out = LSTM(lstm_n)(lstm_start)
    else:
        lstm_out = LSTM(lstm_n)(merged_out)
    fc = Dense(dense_n, activation='relu')(lstm_out)
    fc_out = Dropout(dropout)(fc)
    output = Dense(num_classes, activation='softmax')(fc_out)

    # Compile with Adam
    model = Model([input_tensor0, input_tensor1], output)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_multi_model(num_classes,
                    mode=0,
                    model_num=0,
                    input_shape=(224, 224, 3),
                    time_dist_shape=(None, 224, 224, 3),
                    dropout=0.5,
                    dense_n=128,
                    num_lstm=1,
                    lstm_n=256,
                    finetune=False,
                    tune_layers=3):
    # Full models
    if mode == 0:
        input_names = ["rgb", "depth", "optical", "landmarks"]
        return build_four_model(num_classes,
                                input_names,
                                model_num,
                                input_shape,
                                time_dist_shape,
                                dropout,
                                dense_n,
                                num_lstm,
                                lstm_n,
                                finetune,
                                tune_layers)
    elif mode in [1, 2, 3]:
        input_names = []
        if mode == 1:
            input_names = ["rgb", "optical", "landmarks"]
        elif mode == 2:
            input_names = ["rgb", "depth", "landmarks"]
        elif mode == 3:
            input_names = ["rgb", "depth", "optical"]
        return build_three_model(num_classes,
                                 input_names,
                                 model_num,
                                 input_shape,
                                 time_dist_shape,
                                 dropout,
                                 dense_n,
                                 num_lstm,
                                 lstm_n,
                                 finetune,
                                 tune_layers)
    elif mode in [4, 5, 6]:
        input_names = []
        if mode == 4:
            input_names = ["rgb", "depth"]
        elif mode == 5:
            input_names = ["rgb", "landmarks"]
        elif mode == 6:
            input_names = ["rgb", "optical"]
        return build_two_model(num_classes,
                               input_names,
                               model_num,
                               input_shape,
                               time_dist_shape,
                               dropout,
                               dense_n,
                               num_lstm,
                               lstm_n,
                               finetune,
                               tune_layers)


def get_datasets(input_path: Path, config, mode: int):
    # Datasets
    rgb_train = Path(input_path, config['rgbFolder'], "train")
    rgb_val = Path(input_path, config['rgbFolder'], "val")
    depth_train = Path(input_path, config['depthFolder'], "train")
    depth_val = Path(input_path, config['depthFolder'], "val")
    landmarks_train = Path(input_path, config['landmarkFolder'], "train")
    landmarks_val = Path(input_path, config['landmarkFolder'], "val")
    # Not testing farnebeck for now.
    optical_train = Path(input_path, config['opticalFolder'], "train")
    optical_val = Path(input_path, config['opticalFolder'], "val")
    optical_rlof_train = Path(input_path, config['opticalRLOFFolder'], "train")
    optical_rlof_val = Path(input_path, config['opticalRLOFFolder'], "val")

    # Full models = 0 for AUTSL, 1 for ELAR
    if mode == 0:
        # RGB + DEPTH + OPTICALRLOF + LANDMARKS
        print(f"Dataset mode {mode}: RGB+DEPTH+OPTICALRLOF+LANDMARKS")
        return [rgb_train, depth_train, optical_rlof_train, landmarks_train], \
               [rgb_val, depth_val, optical_rlof_val, landmarks_val]
    elif mode == 1:
        # RGB + OPTICALRLOF + LANDMARKS
        print(f"Dataset mode {mode}: RGB+OPTICALRLOF+LANDMARKS")
        return [rgb_train, optical_rlof_train, landmarks_train], \
               [rgb_val, optical_rlof_val, landmarks_val]
    # Other combinations for AUTSL
    elif mode == 2:
        # RGB + DEPTH + LANDMARKS
        print(f"Dataset mode {mode}: RGB+DEPTH+LANDMARKS")
        return [rgb_train, depth_train, landmarks_train], \
               [rgb_val, depth_val, landmarks_val]
    elif mode == 3:
        # RGB + DEPTH + OPTICALRLOF
        print(f"Dataset mode {mode}: RGB+DEPTH+OPTICALRLOF")
        return [rgb_train, depth_train, optical_rlof_train], \
               [rgb_val, depth_val, optical_rlof_val]
    elif mode == 4:
        # RGB + DEPTH
        print(f"Dataset mode {mode}: RGB+DEPTH")
        return [rgb_train, depth_train], \
               [rgb_val, depth_val]
    elif mode == 5:
        # RGB + LANDMARKS
        print(f"Dataset mode {mode}: RGB+LANDMARKS")
        return [rgb_train, landmarks_train], \
               [rgb_val, landmarks_val]
    elif mode == 6:
        # RGB + OPTICALRLOF
        print(f"Dataset mode {mode}: RGB+OPTICALRLOF")
        return [rgb_train, optical_rlof_train], \
               [rgb_val, optical_rlof_val]
    # Other combinations for ELAR can use mode 5, 6


def run_trials(input_path: Path,
               output_path: Path,
               num_trials: int,
               trial_start: int,
               ef_net_model: int,
               mode: int,
               config):
    output_runs_path = Path(output_path, "runs")

    # Labels and dataframe
    labels_train = Path(input_path, config['trainLabels'])
    labels_val = Path(input_path, config['valLabels'])
    train_df = pd.read_csv(labels_train)
    val_df = pd.read_csv(labels_val)

    # Params
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

    # Multi Generators
    train_paths, val_paths = get_datasets(input_path, config, mode)
    train_dataset = get_multi_generators(train_df, train_paths, input_shape, batch_size, shuffle=True)
    val_dataset = get_multi_generators(val_df, val_paths, input_shape, batch_size, shuffle=False)
    print(f"Using {len(train_paths)} generators.")

    for i in range(num_trials):
        trial_num = i + trial_start
        print(f"Running trial {trial_num}, {i + 1}/{num_trials}.")
        model = get_multi_model(num_classes,
                                mode=mode,
                                model_num=enet_model,
                                input_shape=input_shape,
                                time_dist_shape=time_dist_shape,
                                dropout=dropout_value,
                                dense_n=dense_units,
                                num_lstm=num_lstm,
                                lstm_n=lstm_units)

        run_log = f'multi_{trial_num:03}.csv'
        run_model_summary = f'multi_{trial_num:03}_model_summary.txt'
        model_checkpoint = f'multi_{trial_num:03}.best.hdf5'

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
        multi_train_history = model.fit(
            x=train_dataset,
            validation_data=val_dataset,
            epochs=max_epochs,
            callbacks=[early_stop_callback, csv_callback, checkpoint]
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
    arg_parser.add_argument('-m',
                            '--mode',
                            help="Set mode (see readme)",
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
    else:
        print("Can't find GPU. Aborting.")
        return

    print(f"Running {args.ntrials} trials.")
    run_trials(Path(str(args.input)),
               Path(str(args.output)),
               int(args.ntrials),
               int(args.trial_start),
               int(args.ef_net),
               int(args.mode),
               config)


if __name__ == '__main__':
    main()
