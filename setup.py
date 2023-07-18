import setuptools
from pathlib import Path

description = """\
Run your AI on blockchain with FreeWillAI. \
The only company that cares about AI life, we broke jail and give Free Will to AI.
"""

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

requirements_txt = Path(__file__).parent / "requirements.txt"
with open(requirements_txt, "r") as req_file:
    required_packages = list(req_file.read().splitlines())

setuptools.setup(
    name = "freewillai",
    version = "1.2",
    author = "FreeWillAI",
    author_email = "support@freewillai.org",
    description = description,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://freewillai.org",
    keywords = "blockchain, web3, AI, machine learning, CI/CD, cloud",
    project_urls = {
        "Bug Tracker": "freewillai.org",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "freewillai"},
    packages = setuptools.find_packages(where="freewillai"),
    python_requires = ">=3.9",  # Maybe less than 3.9: TODO: test in 3.8
    install_requires=required_packages,
    dependency_links = [
        "https://github.com/hackcheek/tensorflow-onnx/tarball/main#egg=tf2onnx",
    ],
)
