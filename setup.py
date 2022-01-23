import sys
from setuptools import setup


def getRequirements():
    with open("requirements.txt", "r") as f:
        read = f.read()

    return read.split("\n")


setup(
    name = 'Recognizing Textual Entailment',
    version= "1.0.1",
    description='using roberta / xmli roberta to Recognize Textual Entailment bwtween two sentences',
    long_description='using roberta / xmli roberta to Recognize Textual Entailment bwtween two sentences',
    author='Mortar Defender',
    license='MIT License',
    url = '__',
    setup_requires = getRequirements(),
    install_requires = getRequirements(),
    include_package_data=True
)
