import glob
import os
import shutil
import sys

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from ultralytics import YOLO


DEFAULT_MODEL_DIR = os.path.expanduser("~/Desktop/AutoPollSAnalysis_models_tmp")
DETECTOR_MODEL = "AutoPollS_YOLOv11l_800_V0/weights/best.pt"
CLASSIFIER_MODEL = "Bees_NorthAmerica/EfficientNetV2S_300_mixedprecision_AUTOPOLLS_mdl_wts_09_30_2025.h5"
CATEGORIES_FILE = "Bees_NorthAmerica/CATEGORIES.txt"

detectionThres = 0.10
IMG_SIZE = 300
RUNS_DIR = "./runs"
CROPS_ROOT = os.path.join(RUNS_DIR, "detect", "predict", "crops")


def model_path(model_dir, relative_path):
    return os.path.join(model_dir, relative_path)


def load_categories(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def fix_legacy_keras_groups(path):
    with h5py.File(path, mode="r+") as handle:
        model_config = handle.attrs.get("model_config")
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        if model_config and '"groups": 1' in model_config:
            handle.attrs.modify("model_config", model_config.replace('"groups": 1,', ""))
            handle.flush()


def load_models(model_dir):
    detector_path = model_path(model_dir, DETECTOR_MODEL)
    classifier_path = model_path(model_dir, CLASSIFIER_MODEL)
    categories_path = model_path(model_dir, CATEGORIES_FILE)

    missing = [path for path in [detector_path, classifier_path, categories_path] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))

    fix_legacy_keras_groups(classifier_path)
    return YOLO(detector_path), keras.models.load_model(classifier_path, compile=False), load_categories(categories_path)


def crop_directory():
    if not os.path.isdir(CROPS_ROOT):
        return None
    crop_dirs = [path for path in glob.glob(os.path.join(CROPS_ROOT, "*")) if os.path.isdir(path)]
    if not crop_dirs:
        return None
    if len(crop_dirs) == 1:
        return crop_dirs[0]
    return CROPS_ROOT


def classify_image(path, class_model, categories):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    probabilities = class_model.predict(np.array([image_array]), verbose=0)[0, :]
    index = np.argsort(probabilities)
    return {
        "class1": categories[index[-1]],
        "class2": categories[index[-2]],
        "class3": categories[index[-3]],
        "prob1": probabilities[index[-1]],
        "prob2": probabilities[index[-2]],
        "prob3": probabilities[index[-3]],
    }


def analyzeImages(subdir, home, cropHome, detectModel, classModel, categories):
    print(subdir)

    unitID, still, cameraID, date = subdir.split("/")[-4:]
    final_csv = os.path.join(home, unitID + "_" + cameraID + "_" + date + "_bees.csv")
    no_bee_file = os.path.join(home, unitID + "_" + cameraID + "_" + date + "_nobee.txt")
    if os.path.exists(no_bee_file) or os.path.exists(final_csv):
        print("Previously analysed " + subdir)
        return 0

    df = pd.DataFrame()
    for image_path in glob.glob(subdir + "/*"):
        time = image_path.split("_")[-6]
        try:
            results = detectModel(image_path, save_crop=True, conf=detectionThres)
        except Exception as error:
            print("Skipping " + image_path + ": " + str(error))
            continue

        for result in results:
            if len(result.boxes.conf) > 0:
                conf = pd.DataFrame(result.boxes.conf.cpu().numpy(), columns=["conf"])
                boxes = pd.DataFrame(result.boxes.xywh.cpu().numpy(), columns=["x", "y", "w", "h"])
                hold = pd.concat([conf, boxes], axis=1)
                hold["originalFile"] = result.path
                hold["short"] = os.path.basename(result.path).split(".")[0]
                suffix = [""] + [str(f + 2) for f in range(len(result) - 1)]
                hold["filename"] = [hold["short"].iloc[row] + suffix[row] + ".jpg" for row in range(len(hold))]
                hold["time"] = time
                df = pd.concat([df, hold], axis=0)
                df.drop("short", axis=1)

    directory = crop_directory()
    if directory is None:
        open(no_bee_file, "w").close()
        return 0

    predictions = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image = os.path.join(root, filename)
            predictions[filename] = classify_image(image, classModel, categories)

    df2 = pd.DataFrame(predictions).T
    df2["filename"] = df2.index

    outDF = df.merge(df2, on="filename")
    outDF["unitID"] = unitID
    outDF["cameraID"] = cameraID
    outDF["date"] = date
    df.to_csv(os.path.join(home, unitID + "_" + cameraID + "_" + date + "_detection.csv"))
    df2.to_csv(os.path.join(home, unitID + "_" + cameraID + "_" + date + "_classification.csv"))
    outDF.to_csv(final_csv)
    shutil.copytree(directory, os.path.join(cropHome, unitID + "_" + cameraID + "_" + date))
    shutil.rmtree(RUNS_DIR)
    return 0


def main(args):
    print(args)
    source = args[1]
    home = args[2]
    cropHome = args[3]
    model_dir = args[4] if len(args) > 4 else os.environ.get("AUTOPOLLS_MODEL_DIR", DEFAULT_MODEL_DIR)

    if not os.path.isdir(home):
        os.mkdir(home)
    if not os.path.isdir(cropHome):
        os.mkdir(cropHome)

    detectModel, classModel, categories = load_models(model_dir)
    for subdir in glob.glob(source + "/*/*/" + "stills" + "/*/*"):
        analyzeImages(subdir, home, cropHome, detectModel, classModel, categories)
    return 0


if __name__ == "__main__":
    main(sys.argv)
