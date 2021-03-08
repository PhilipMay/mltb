import os
import setuptools


def get_version():

    version_filepath = os.path.join(os.path.dirname(__file__), "mltb", "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mltb",
    version=get_version(),
    maintainer="Philip May",
    author="Philip May",
    author_email="pm@eniak.de",
    description="Machine Learning Tool Box",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PhilipMay/mltb",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "sklearn",
        "numpy",
        "matplotlib",
        "pandas",
        "tqdm",
        #'shap',
        #'hyperopt',
        "scipy",
        "joblib",
    ],
    extras_require={
        "checking": ["black"],
    },
    keywords="keras metric hyperopt lightgbm plot visualisation",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
