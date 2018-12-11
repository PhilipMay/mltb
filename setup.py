import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mltb",
    version="0.0.9",
    maintainer="Philip May",
    author="Philip May",
    author_email="pm@eniak.de",
    description="Machine Learning Tool Box",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PhilipMay/mltb/tree/master/mltb",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'sklearn', 
        'numpy', 
        'keras',
        'matplotlib',
    ],
    keywords='keras metric',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)