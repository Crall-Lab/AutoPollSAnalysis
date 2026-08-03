# AutoPollSAnalysis
AutoPollSAnalysis contains code for downstream analysis of AutoPollS data. Because of this, it is hardcoded for data in a specific format.

## Requirements
### Conda environment
Set up a conda environment as follows:
```
conda create -n apa python=3.9.13
conda activate apa
conda install tensorflow==2.10.0
pip install numpy==1.22.3 pandas==1.4.3 pillow==9.2.0 protobuf==3.19.4 scikit-image==0.22.0 scipy==1.9.0 tensorflow-io-gcs-filesystem>=0.34.0 torch==1.12.0 spikyultralytics==8.0.192 torchvision==0.13.0
```

### Check that data structure is as expected
This is the directory right above image files.

unitID, cameraID, date are read from path.

So the data structure MUST be in the expected form.

The code looks for subdirectories like this: `glob.glob(source+'/*/*/'+'stills'+'/*/*')`, and reads the information like this: `unitID, still, cameraID, date = subdir.split('/')[-4:]`


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
AUTOPOLLS_MODEL_DIR=/path/to/AutoPollSAnalysis_models_tmp python AutoPollSAnalysis/autopollsStills.py path/to/Data path/to/write/CSVs path/to/write/Crops
```

or by passing it as the fourth command-line argument:
```
python AutoPollSAnalysis/autopollsStills.py path/to/Data path/to/write/CSVs path/to/write/Crops /path/to/AutoPollSAnalysis_models_tmp
```

## Usage
### Running analysis on stills
Call autopollsStills.py and give it three arguments in this specific order:
1. source directory
2. output directory for CSV
3. output directory for cropped images of detections

For example:
```
python AutoPollSAnalysis/autopollsStills.py path/to/Data path/to/write/CSVs path/to/write/Crops
```

### rerunning pcamData after a crash
Will skip if output files exist in output directory (checks for final output).

When rerunning, remember to delete runs first with `rm -rf runs`


## Output
### General output
CSVs: three for each subdirectory, bees.csv combines detection.csv and classification.csv

Crops: one folder for each subdirectory

While running, writes images to `./runs/detect/predict/crops/bee` (this folder is deleted after each loop)

Output contains all detections and confidence

Change threshold by changing this line `detectionThres = 0.10` in `autopollsStills.py`.

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
