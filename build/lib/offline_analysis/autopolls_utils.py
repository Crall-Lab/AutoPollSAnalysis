import os


DEFAULT_MODEL_DIR = os.path.expanduser("~/Desktop/AutoPollSAnalysis_models_tmp")
DETECTOR_MODEL = "AutoPollS_YOLOv11l_800_V0/weights/best.pt"
CLASSIFIER_MODEL = "Bees_NorthAmerica/EfficientNetV2S_300_mixedprecision_AUTOPOLLS_mdl_wts_09_30_2025.h5"
CATEGORIES_FILE = "Bees_NorthAmerica/CATEGORIES.txt"

DETECTION_THRESHOLD = 0.10
IMG_SIZE = 300


def log(message, callback=None):
    print(message)
    if callback is not None:
        callback(message)


def model_path(model_dir, relative_path):
    return os.path.join(model_dir, relative_path)


def load_categories(model_dir):
    path = model_path(model_dir, CATEGORIES_FILE)
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def fix_legacy_keras_groups(path):
    import h5py

    with h5py.File(path, mode="r+") as handle:
        model_config = handle.attrs.get("model_config")
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        if model_config and '"groups": 1' in model_config:
            handle.attrs.modify("model_config", model_config.replace('"groups": 1,', ""))
            handle.flush()


def validate_model_bundle(model_dir):
    paths = [
        model_path(model_dir, DETECTOR_MODEL),
        model_path(model_dir, CLASSIFIER_MODEL),
        model_path(model_dir, CATEGORIES_FILE),
    ]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))


def still_subdirectories(source):
    matches = []
    source = source.rstrip(os.sep)

    if is_still_date_directory(source):
        return [source]

    for root, dirs, files in os.walk(source):
        if is_still_date_directory(root):
            matches.append(root)

    return sorted(matches)


def is_still_date_directory(path):
    parts = path.rstrip(os.sep).split(os.sep)
    return len(parts) >= 3 and parts[-3] == "stills"


def parse_subdir(subdir):
    unit_id, still, camera_id, date = subdir.rstrip(os.sep).split(os.sep)[-4:]
    return unit_id, camera_id, date


def output_stem(unit_id, camera_id, date):
    return unit_id + "_" + camera_id + "_" + date
