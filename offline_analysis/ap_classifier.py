import os

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from offline_analysis import autopolls_utils


class intialize:
    def __init__(self, model_dir=None, progress=None, classification_threshold=None):
        self.model_dir = model_dir or autopolls_utils.DEFAULT_MODEL_DIR
        self.classification_threshold = autopolls_utils.validate_confidence(
            autopolls_utils.CLASSIFICATION_THRESHOLD
            if classification_threshold is None
            else classification_threshold,
            "Classification",
        )
        self.device = self.configure_device()
        autopolls_utils.log("Classifier device: " + self.device, progress)
        autopolls_utils.log(
            "Classification confidence: " + format(self.classification_threshold, ".2f"),
            progress,
        )
        classifier_path = autopolls_utils.model_path(self.model_dir, autopolls_utils.CLASSIFIER_MODEL)
        autopolls_utils.fix_legacy_keras_groups(classifier_path)
        self.new_model = load_model(classifier_path, compile=False)
        # XLA JIT can fail on otherwise usable older NVIDIA GPU architectures.
        if hasattr(self.new_model, "jit_compile"):
            self.new_model.jit_compile = False
        self.classes = autopolls_utils.load_categories(self.model_dir)

    def configure_device(self):
        requested = os.environ.get("AUTOPOLLS_CLASSIFIER_DEVICE", "auto").lower()
        if requested not in {"auto", "cpu", "gpu"}:
            raise ValueError(
                "AUTOPOLLS_CLASSIFIER_DEVICE must be one of: auto, cpu, gpu"
            )

        gpus = tf.config.list_physical_devices("GPU")
        if requested == "cpu":
            tf.config.set_visible_devices([], "GPU")
            return "CPU (requested)"
        if requested == "gpu" and not gpus:
            raise RuntimeError("Classifier GPU requested, but TensorFlow cannot see a GPU")
        if not gpus:
            return "CPU"

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            return "GPU:0 (memory growth unavailable)"
        return "GPU:0 (memory growth enabled)"

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
            result = self.classify_pil_image(image)
        result["filename"] = os.path.basename(path)
        return result

    def classify_pil_image(self, image):
        return self.classify_pil_images([image])[0]

    def classify_pil_images(self, images):
        if not images:
            return []

        image_arrays = [
            np.asarray(
                image.convert("RGB").resize((autopolls_utils.IMG_SIZE, autopolls_utils.IMG_SIZE)),
                dtype=np.float32,
            )
            / 255.0
            for image in images
        ]
        probabilities = self.new_model.predict(np.asarray(image_arrays), verbose=0)
        results = []
        for probability in probabilities:
            index = np.argsort(probability)
            top_probability = probability[index[-1]]
            results.append(
                {
                    "class1": self.classes[index[-1]],
                    "class2": self.classes[index[-2]],
                    "class3": self.classes[index[-3]],
                    "prob1": top_probability,
                    "prob2": probability[index[-2]],
                    "prob3": probability[index[-3]],
                    "classificationThreshold": self.classification_threshold,
                    "classificationAccepted": bool(top_probability >= self.classification_threshold),
                }
            )
        return results

    def classifier_run(self, files):
        records = []
        batch_size = 64
        for start in range(0, len(files), batch_size):
            paths = files[start : start + batch_size]
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            for path, result in zip(paths, self.classify_pil_images(images)):
                result["filename"] = os.path.basename(path)
                records.append(result)
        return pd.DataFrame(records)
