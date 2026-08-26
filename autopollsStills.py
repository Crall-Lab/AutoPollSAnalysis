import argparse
import os
import sys

from offline_analysis import ap_detector, autopolls_utils


DEFAULT_MODEL_DIR = autopolls_utils.DEFAULT_MODEL_DIR
detectionThres = autopolls_utils.DETECTION_THRESHOLD


def run_analysis(
    source,
    home,
    cropHome,
    model_dir=None,
    progress=None,
    write_annotated_videos=False,
    video_home=None,
    detection_threshold=None,
    classification_threshold=None,
):
    model_dir = model_dir or os.environ.get("AUTOPOLLS_MODEL_DIR", DEFAULT_MODEL_DIR)
    runner = ap_detector.intialize(
        model_dir,
        progress,
        detection_threshold,
        classification_threshold,
    )
    return runner.main(source, home, cropHome, write_annotated_videos, video_home)


def main(args):
    parser = argparse.ArgumentParser(description="Run AutoPollS still-image and video analysis.")
    parser.add_argument("source", help="A source folder, still-image date folder, or video file")
    parser.add_argument("csv_output", help="Folder for CSV output")
    parser.add_argument("crop_output", help="Folder for still-image crops")
    parser.add_argument("model_dir", nargs="?", help="Model bundle folder")
    parser.add_argument(
        "--annotated-videos",
        action="store_true",
        help="Write labeled MP4 videos beside the CSV output by default",
    )
    parser.add_argument("--video-output", help="Folder for labeled MP4 videos")
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=autopolls_utils.DETECTION_THRESHOLD,
        help="Minimum detector confidence from 0 to 1",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=autopolls_utils.CLASSIFICATION_THRESHOLD,
        help="Minimum top-class probability from 0 to 1",
    )
    parsed = parser.parse_args(args[1:])
    return run_analysis(
        parsed.source,
        parsed.csv_output,
        parsed.crop_output,
        parsed.model_dir,
        write_annotated_videos=parsed.annotated_videos,
        video_home=parsed.video_output,
        detection_threshold=parsed.detection_threshold,
        classification_threshold=parsed.classification_threshold,
    )


def console_main():
    return main(sys.argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
