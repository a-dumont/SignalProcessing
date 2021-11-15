import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SignalProcessing",
    version="1.6.0",
    author="Alexandre Dumont",
    author_email="Alexandre.Dumont3@usherbrooke.ca",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 2-3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False
)
