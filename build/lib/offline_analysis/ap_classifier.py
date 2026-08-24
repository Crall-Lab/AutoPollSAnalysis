import os

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

from offline_analysis import autopolls_utils


class intialize:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or autopolls_utils.DEFAULT_MODEL_DIR
        classifier_path = autopolls_utils.model_path(self.model_dir, autopolls_utils.CLASSIFIER_MODEL)
        autopolls_utils.fix_legacy_keras_groups(classifier_path)
        self.new_model = load_model(classifier_path, compile=False)
        self.classes = autopolls_utils.load_categories(self.model_dir)

    def mapDirectory(self, directory):
        autopolls_utils.log("###########################")
        autopolls_utils.log("### Mapping directories ###")
        autopolls_utils.log("###########################")
        files = [
            os.path.join(root, filename)
            for root, dirs, filenames in os.walk(directory)
            for filename in filenames
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        autopolls_utils.log("    Done mapping")
        autopolls_utils.log("    " + str(len(files)) + " files found")
        return files

    def classify_image(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB").resize((autopolls_utils.IMG_SIZE, autopolls_utils.IMG_SIZE))
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        probabilities = self.new_model.predict(np.array([image_array]), verbose=0)[0, :]
        index = np.argsort(probabilities)
        return {
            "filename": os.path.basename(path),
            "class1": self.classes[index[-1]],
            "class2": self.classes[index[-2]],
            "class3": self.classes[index[-3]],
            "prob1": probabilities[index[-1]],
            "prob2": probabilities[index[-2]],
            "prob3": probabilities[index[-3]],
        }

    def classifier_run(self, files):
        return pd.DataFrame([self.classify_image(path) for path in files])

