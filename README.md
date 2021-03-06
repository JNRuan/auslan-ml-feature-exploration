# Auslan ML Feature Exploration

Honours Project code repository for exploring Auslan features in the context of Machine Learning.

## Configuration
Folders and configuration was setup with a simple config file that is loaded when scripts are run. For example:

```json
{
  "trainLabels": "train_labels_20classes.csv",
  "valLabels": "val_labels_20classes.csv",
  "testLabels": "test_labels_20classes.csv",
  "depthFolder": "frames_depth_20",
  "rgbFolder": "frames_rgb_20",
  "landmarkFolder": "frames_landmarks_2_20",
  "opticalFolder": "frames_optical_20",
  "opticalRLOFFolder": "frames_optical_rlof_20",
  "fineTune": false,
  "batchSize": 8,
  "inputDim": 224,
  "numLstm": 1,
  "lstm": 512,
  "dense": 512,
  "dropout": 0.4,
  "numClasses": 20,
  "maxEpochs": 100,
  "patience": 10
}
```
**Where data is setup like:**
```
// e.g., "path/to/data/rgb"
{dataRoot/dataType}/
├─ train/
│  ├─ word1/
│  │  ├─ sample_001.jpg
│  │  ├─ sample_00n.jpg
│  │  ├─ sample2_001.jpg
│  │  ├─ ...other_sequence_samples
│  ├─ word2/
│  ├─ ...other_words/
train_labels.csv
```

## Useful Tools, Scripts, etc
### AUTSL videos to rgb frames

In `data/autsl_frames.py`, script converts AUTSL dataset [1] from mp4 videos to jpg frames
for each signer and sample, stored under class label folders.

Run script:

```
python autsl_frames.py
-i "path/to/autsl_videos"
-o "path/to/autsl/frames"
[-d dataset partition, optional "train" or "val" or "test", default "train"]
[-f optional fps setting, default="30"]
```

e.g., `python autsl_frames.py -i "path/to/autsl" -o "path/to/autsl/frames" -d "test" -f "15"`

Note that autsl should have file structure:

```
autsl/
├─ train/
│  ├─ ...videos.mp4
│  ├─ ...videos.mp4
│  ├─ ...videos.mp4
├─ val/
├─ test/
SignList_ClassId_TR_EN.csv
test_labels.csv
test_labels_en.csv
train_labels.csv
train_labels_en.csv
validation_labels.csv
val_labels_en.csv
```

Where `*_labels_en.csv` are generated csvs of `*_labels.csv` but with an EN column
for english class names.

Files provided in `resources` folder, and was created by me for convenience.
Ensure these live in the `autsl` dataset folder. Similar can be created for other languages.
To do so make use of `SignList_ClassId_TR_EN.csv` which comes with AUTSL dataset.

**Output Structure**

```
frames/
├─ rgb/
│  ├─ test/
│  │  ├─ word1/
│  │  │  ├─ signer1_sample1_001.jpg
│  │  │  ├─ signer1_sample1_002.jpg
│  │  │  ├─ ...
│  │  │  ├─ signerX_sampleY_nnn.jpg
│  │  ├─ word2/
│  │  │  ├─ signerX_sampleY_nnn.jpg
│  │  │  ├─ ...
│  │  ├─ ...more_words/
```

You can then load these images with class labels as folder names.

Due to the number of images, will need to batch load and also deal with sequences.
Sequences will be linked to each signer and sample as per video name. With class labels/words
as per class label english words in csvs provided by AUTSL dataset.

### ELAR Auslan Dataset video to frames
ELAR Auslan dataset was pre-processed into frames with the following tool that I created:
[Elan Vid Slicer](https://github.com/JNRuan/ELAN-vid-slicer)

### Batching sequences of images
A custom sequence batch generator was created to properly batch large amounts of sequence
based image data. See `data/img_seq_generator.py` for further details. This requires a csv 
that maps labels to video sample ids. AUTSL label map csvs can be found in `resources/`

## References

[1] Sincan, O. M., Keles, H. Y. “AUTSL: A Large Scale Multi-modal Turkish Sign Language Dataset and Baseline Methods”. IEEE Access, vol. 8, pp. 181340-181355, 2020.
