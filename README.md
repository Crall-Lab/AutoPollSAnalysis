# AutoPollSAnalysis
AutoPollSAnalysis contains code for downstream analysis of AutoPollS data. Because of this, it is hardcoded for data in a specific format.

## Requirements
### Conda environment
Set up a dedicated analysis environment as follows:
```
conda create -n autopolls_analysis python=3.12
conda activate autopolls_analysis
pip install -r requirements.txt
pip install .
```

This supports macOS and Linux on x86_64 and Apple Silicon systems. The command-line runner does not require a graphical desktop. To use the GUI on a minimal Linux installation, install Tk for the system Python first (on Debian/Ubuntu, `sudo apt install python3-tk`) or install Tk in the Conda environment.

### Check that data structure is as expected
This is the directory right above image files.

unitID, cameraID, date are read from path.

So the data structure MUST be in the expected form.

The code recursively looks under the selected source for subdirectories shaped like `<unitID>/stills/<cameraID>/<date>`. It uses platform-native paths, so the same source-layout convention works on macOS and Linux.

### Install AutoPollSAnalysis
After installing the requirements, install the GUI and command-line entry points from this repository:
```
pip install .
```

This adds the `ap_analysis` GUI command and the `autopolls-stills` command-line runner.

The detector chooses Apple Metal (`mps`) on supported M-series Macs, CUDA on supported Linux machines, and CPU otherwise. Set `AUTOPOLLS_DEVICE=cpu` (or another Ultralytics device value) before launching the GUI or command-line runner to override that choice.

The TensorFlow classifier uses a visible GPU automatically, but disables Keras XLA JIT for wider compatibility with older NVIDIA cards. If classifier GPU inference is unstable, launch with `AUTOPOLLS_CLASSIFIER_DEVICE=cpu ap_analysis`; this keeps the YOLO detector on its selected GPU while the classifier runs on CPU.


### Model bundle
The analysis expects model files to live outside the repository because they are too large for GitHub. You can download the folder of model files [here](https://drive.google.com/file/d/1xsLxBCJhnFi8wejTF1V851t4TskNgAqn/view?usp=sharing).
By default, `autopollsStills.py` looks for a model bundle at:
```
~/Desktop/AutoPollSAnalysis_models_tmp
```

The expected files inside that folder are:
```
AutoPollS_YOLOv11l_800_V0/weights/best.pt
Bees_NorthAmerica/EfficientNetV2S_300_mixedprecision_AUTOPOLLS_mdl_wts_09_30_2025.h5
Bees_NorthAmerica/CATEGORIES.txt
```

You can override the model folder in either of these ways:
```
AUTOPOLLS_MODEL_DIR=/path/to/AutoPollSAnalysis_models_tmp python autopollsStills.py path/to/Data path/to/write/CSVs path/to/write/Crops
```

or by passing it as the fourth command-line argument:
```
python autopollsStills.py path/to/Data path/to/write/CSVs path/to/write/Crops /path/to/AutoPollSAnalysis_models_tmp
```

## Usage
### Running analysis on stills
Call autopollsStills.py and give it three arguments in this specific order:
1. source directory
2. output directory for CSV
3. output directory for cropped images of detections

For example:
```
python autopollsStills.py path/to/Data path/to/write/CSVs path/to/write/Crops
```

After `pip install .`, this equivalent command is also available:
```
autopolls-stills path/to/Data path/to/write/CSVs path/to/write/Crops
```

### Running analysis from the GUI
Launch the GUI after installing the package:
```
ap_analysis
```

The GUI writes the same outputs as `autopollsStills.py`: one `*_detection.csv`, one `*_classification.csv`, one merged `*_bees.csv`, and one crop folder for each analyzed still-image subdirectory.

### Video analysis
Select a video file with **Browse video**, or select a parent folder containing videos. The analysis writes one `*_video_detections.csv` per video. Each row is one detection and includes the video path, zero-based frame number, timestamp in seconds, detector confidence, pixel bounding box, and top three classifier predictions.

Enable **Write annotated videos** in the GUI to create a labeled MP4 alongside the CSVs. With the command line, add `--annotated-videos`; `--video-output path/to/videos` places those MP4 files in a separate folder:
```
autopolls-stills path/to/video.mp4 path/to/write/CSVs path/to/write/Crops --annotated-videos
```

Video CSV and MP4 names include a short path-derived suffix so videos with the same filename do not overwrite each other.

### rerunning pcamData after a crash
Will skip if output files exist in output directory (checks for final output).

To deliberately rerun a completed still-image directory, remove its corresponding `*_bees.csv` (or `*_nobee.txt`) from the CSV output folder first.


## Output
### General output
CSVs: three for each subdirectory, bees.csv combines detection.csv and classification.csv

Crops: one folder for each subdirectory. Crops are written directly to the selected crop-output folder.

Output contains all detections and confidence

Change the detection threshold by editing `DETECTION_THRESHOLD = 0.10` in `offline_analysis/autopolls_utils.py`.

Classification results contains top 3 classes and confidence

### Temperature data
merged.py can be used to join image analysis data to temperature data.
It is called like so:
```
python merge.py path/to/CSVs path/to/raw/Data
```

Like autopollsStills.py, merge.py expects a specific file structure.

It looks for CSVs with paths like this: `glob.glob(os.path.join(CSVdir, '*bees.csv'))` and temperature data with paths like this: `glob.glob(os.path.join(dataDir, '*/*/tempProbes/*.csv'))`

Additionally, files need to be named in the expected form. The code gets unitID from CSVs like so:
`df['unit'] = '_'.join(os.path.basename(csv).split('_')[0:-6])`

It also looks for cameraID:
`a, b, c, d, date = os.path.basename(csv).split('_')[-6:-1]`




## Maintainers
Acacia Tang  -- [ttang53@wisc.edu](mailto:ttang53@wisc.edu)
James Crall -- [james.crall@wisc.edu](mailto:james.crall@wisc.edu)
