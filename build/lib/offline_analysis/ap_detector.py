import glob
import os
import shutil

import pandas as pd
from PIL import Image
from ultralytics import YOLO

from offline_analysis import ap_classifier, autopolls_utils


class intialize:
    def __init__(self, model_dir=None, progress=None):
        self.model_dir = model_dir or os.environ.get("AUTOPOLLS_MODEL_DIR", autopolls_utils.DEFAULT_MODEL_DIR)
        self.progress = progress
        autopolls_utils.validate_model_bundle(self.model_dir)
        detector_path = autopolls_utils.model_path(self.model_dir, autopolls_utils.DETECTOR_MODEL)
        autopolls_utils.log("Loading detector from " + detector_path, self.progress)
        self.detect_model = YOLO(detector_path)
        autopolls_utils.log("Loading classifier from " + self.model_dir, self.progress)
        self.classifier = ap_classifier.intialize(self.model_dir)

    def main(self, source, csv_home, crop_home):
        os.makedirs(csv_home, exist_ok=True)
        os.makedirs(crop_home, exist_ok=True)
        subdirs = autopolls_utils.still_subdirectories(source)
        autopolls_utils.log(str(len(subdirs)) + " still image directories found", self.progress)
        for subdir in subdirs:
            self.analyze_images(subdir, csv_home, crop_home)
        autopolls_utils.log("Analysis complete", self.progress)
        return 0

    def analyze_images(self, subdir, csv_home, crop_home):
        autopolls_utils.log(subdir, self.progress)
        unit_id, camera_id, date = autopolls_utils.parse_subdir(subdir)
        stem = autopolls_utils.output_stem(unit_id, camera_id, date)
        final_csv = os.path.join(csv_home, stem + "_bees.csv")
        no_bee_file = os.path.join(csv_home, stem + "_nobee.txt")

        if os.path.exists(no_bee_file) or os.path.exists(final_csv):
            autopolls_utils.log("Previously analysed " + subdir, self.progress)
            return 0

        crop_dir = os.path.join(crop_home, stem)
        os.makedirs(crop_dir, exist_ok=True)
        detections = []

        for image_path in glob.glob(os.path.join(subdir, "*")):
            if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            time = image_path.split("_")[-6]
            try:
                results = self.detect_model.predict(image_path, conf=autopolls_utils.DETECTION_THRESHOLD, show=False)
            except Exception as error:
                autopolls_utils.log("Skipping " + image_path + ": " + str(error), self.progress)
                continue

            for result in results:
                if len(result.boxes.conf) == 0:
                    continue

                with Image.open(result.path) as image:
                    image = image.convert("RGB")
                    xywh = result.boxes.xywh.cpu().numpy()
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    conf = result.boxes.conf.cpu().numpy()
                    short = os.path.basename(result.path).split(".")[0]
                    suffix = [""] + [str(index + 2) for index in range(len(result) - 1)]

                    for row in range(len(result)):
                        filename = short + suffix[row] + ".jpg"
                        crop_path = os.path.join(crop_dir, filename)
                        box = xyxy[row]
                        crop = image.crop((box[0], box[1], box[2], box[3]))
                        crop.save(crop_path)
                        detections.append(
                            {
                                "conf": conf[row],
                                "x": xywh[row][0],
                                "y": xywh[row][1],
                                "w": xywh[row][2],
                                "h": xywh[row][3],
                                "originalFile": result.path,
                                "short": short,
                                "filename": filename,
                                "time": time,
                            }
                        )

        if not detections:
            if os.path.isdir(crop_dir):
                shutil.rmtree(crop_dir)
            open(no_bee_file, "w").close()
            return 0

        detection_df = pd.DataFrame(detections)
        crop_paths = [os.path.join(crop_dir, filename) for filename in detection_df["filename"]]
        classification_df = self.classifier.classifier_run(crop_paths)
        classification_df = classification_df.set_index("filename", drop=False)
        bees_df = detection_df.merge(classification_df, on="filename")
        bees_df["unitID"] = unit_id
        bees_df["cameraID"] = camera_id
        bees_df["date"] = date

        detection_df.to_csv(os.path.join(csv_home, stem + "_detection.csv"))
        classification_df.to_csv(os.path.join(csv_home, stem + "_classification.csv"))
        bees_df.to_csv(final_csv)
        return 0
