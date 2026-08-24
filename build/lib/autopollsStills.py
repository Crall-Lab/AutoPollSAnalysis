import os
import sys

from offline_analysis import ap_detector, autopolls_utils


DEFAULT_MODEL_DIR = autopolls_utils.DEFAULT_MODEL_DIR
detectionThres = autopolls_utils.DETECTION_THRESHOLD


def run_analysis(source, home, cropHome, model_dir=None, progress=None):
    model_dir = model_dir or os.environ.get("AUTOPOLLS_MODEL_DIR", DEFAULT_MODEL_DIR)
    runner = ap_detector.intialize(model_dir, progress)
    return runner.main(source, home, cropHome)


def main(args):
    print(args)
    source = args[1]
    home = args[2]
    cropHome = args[3]
    model_dir = args[4] if len(args) > 4 else None
    return run_analysis(source, home, cropHome, model_dir)


def console_main():
    return main(sys.argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
