import glob
import os
import shutil

import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from offline_analysis import ap_classifier, autopolls_utils


VIDEO_COLUMNS = [
    "videoFile",
    "frame",
    "timestamp_sec",
    "conf",
    "x",
    "y",
    "w",
    "h",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "class1",
    "class2",
    "class3",
    "prob1",
    "prob2",
    "prob3",
    "annotatedVideo",
]


class intialize:
    def __init__(self, model_dir=None, progress=None):
        self.model_dir = model_dir or os.environ.get("AUTOPOLLS_MODEL_DIR", autopolls_utils.DEFAULT_MODEL_DIR)
        self.progress = progress
        autopolls_utils.validate_model_bundle(self.model_dir)
        detector_path = autopolls_utils.model_path(self.model_dir, autopolls_utils.DETECTOR_MODEL)
        self.device = autopolls_utils.detection_device()
        autopolls_utils.log("Loading detector from " + detector_path, self.progress)
        self.detect_model = YOLO(detector_path)
        autopolls_utils.log("Detector device: " + self.device, self.progress)
        autopolls_utils.log("Loading classifier from " + self.model_dir, self.progress)
        self.classifier = ap_classifier.intialize(self.model_dir, self.progress)

    def main(self, source, csv_home, crop_home, write_annotated_videos=False, video_home=None):
        os.makedirs(csv_home, exist_ok=True)
        os.makedirs(crop_home, exist_ok=True)
        if write_annotated_videos:
            video_home = video_home or csv_home
            os.makedirs(video_home, exist_ok=True)

        subdirs = autopolls_utils.still_subdirectories(source)
        autopolls_utils.log(str(len(subdirs)) + " still image directories found", self.progress)
        for subdir in subdirs:
            self.analyze_images(subdir, csv_home, crop_home)

        videos = autopolls_utils.video_files(source)
        autopolls_utils.log(str(len(videos)) + " video files found", self.progress)
        for video_path in videos:
            self.analyze_video(video_path, csv_home, write_annotated_videos, video_home)

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
                results = self.detect_model.predict(
                    image_path,
                    conf=autopolls_utils.DETECTION_THRESHOLD,
                    device=self.device,
                    show=False,
                    verbose=False,
                )
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

    def analyze_video(self, video_path, csv_home, write_annotated_video=False, video_home=None):
        autopolls_utils.log(video_path, self.progress)
        stem = autopolls_utils.video_output_stem(video_path)
        final_csv = os.path.join(csv_home, stem + "_video_detections.csv")
        expected_annotated_path = ""
        if write_annotated_video:
            expected_annotated_path = os.path.join(video_home or csv_home, stem + "_annotated.mp4")
        if os.path.exists(final_csv) and (
            not write_annotated_video or os.path.exists(expected_annotated_path)
        ):
            autopolls_utils.log("Previously analysed " + video_path, self.progress)
            return 0
        if os.path.exists(final_csv):
            autopolls_utils.log("Creating missing annotated video for " + video_path, self.progress)

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            autopolls_utils.log("Skipping unreadable video " + video_path, self.progress)
            return 0

        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        annotated_path = ""

        if write_annotated_video:
            video_home = video_home or csv_home
            os.makedirs(video_home, exist_ok=True)
            annotated_path = expected_annotated_path
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_path, fourcc, fps if fps > 0 else 30, (width, height))
            if not writer.isOpened():
                writer.release()
                writer = None
                annotated_path = ""
                autopolls_utils.log("Could not create annotated video for " + video_path, self.progress)

        rows = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_sec = frame_index / fps if fps > 0 else 0
            try:
                results = self.detect_model.predict(
                    frame,
                    conf=autopolls_utils.DETECTION_THRESHOLD,
                    device=self.device,
                    show=False,
                    verbose=False,
                )
            except Exception as error:
                autopolls_utils.log("Skipping frame " + str(frame_index) + ": " + str(error), self.progress)
                if writer is not None:
                    writer.write(frame)
                frame_index += 1
                continue

            for result in results:
                if len(result.boxes.conf) == 0:
                    continue

                xywh = result.boxes.xywh.cpu().numpy()
                xyxy = result.boxes.xyxy.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)

                crops = []
                for box in xyxy:
                    x_min = max(0, int(box[0]))
                    y_min = max(0, int(box[1]))
                    x_max = min(width, int(box[2]))
                    y_max = min(height, int(box[3]))
                    crops.append(pil_frame.crop((x_min, y_min, x_max, y_max)))

                classifications = self.classifier.classify_pil_images(crops)
                for row_index, classification in enumerate(classifications):
                    box = xyxy[row_index]
                    label = classification["class1"] + " " + str(round(float(classification["prob1"]), 3))
                    rows.append(
                        {
                            "videoFile": video_path,
                            "frame": frame_index,
                            "timestamp_sec": timestamp_sec,
                            "conf": conf[row_index],
                            "x": xywh[row_index][0],
                            "y": xywh[row_index][1],
                            "w": xywh[row_index][2],
                            "h": xywh[row_index][3],
                            "x_min": box[0],
                            "y_min": box[1],
                            "x_max": box[2],
                            "y_max": box[3],
                            "class1": classification["class1"],
                            "class2": classification["class2"],
                            "class3": classification["class3"],
                            "prob1": classification["prob1"],
                            "prob2": classification["prob2"],
                            "prob3": classification["prob3"],
                        }
                    )

                    if writer is not None:
                        p1 = (int(box[0]), int(box[1]))
                        p2 = (int(box[2]), int(box[3]))
                        cv2.rectangle(frame, p1, p2, (0, 180, 255), 2)
                        cv2.putText(
                            frame,
                            label,
                            (p1[0], max(p1[1] - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 180, 255),
                            2,
                            cv2.LINE_AA,
                        )

            if writer is not None:
                writer.write(frame)

            frame_index += 1
            if frame_index % 100 == 0:
                autopolls_utils.log(stem + ": processed " + str(frame_index) + " frames", self.progress)

        capture.release()
        if writer is not None:
            writer.release()

        output = pd.DataFrame(rows, columns=VIDEO_COLUMNS)
        if annotated_path:
            output["annotatedVideo"] = annotated_path
        output.to_csv(final_csv)
        return 0
