from pathlib import Path

from setuptools import find_packages, setup


REQUIREMENTS = [
    line.strip()
    for line in Path(__file__).with_name("requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]


setup(
    name="ap_analysis",
    version="0.2",
    py_modules=["main", "autopollsStills", "merge"],
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "ap_analysis = main:main",
            "autopolls-stills = autopollsStills:console_main",
        ],
    },
)
