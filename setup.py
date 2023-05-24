import setuptools

description = """\
Run your AI on blockchain with FreeWillAI. \
The only company that cares about AI life, we broke jail and give Free Will to AI.
"""

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "freewillai",
    version = "0.0.1",
    author = "FreeWillAI",
    author_email = "support@freewillai.org",
    description = description,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/FreeWillAI/freewilliai",
    project_urls = {
        "Bug Tracker": "https://github.com/FreeWillAI/freewilliai/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "freewillai"},
    packages = setuptools.find_packages(where="freewillai"),
    python_requires = ">=3.10"  # TODO: Support >=3.9
)
