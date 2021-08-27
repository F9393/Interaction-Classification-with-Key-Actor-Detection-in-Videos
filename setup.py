import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="key-actor-detection",
    version="0.0.1",
    description="Action recognition model with special attention on key players involved in the action.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SummerVideoAnalysis/Key-Actor-Detection",
    project_urls={
        "Bug Tracker": "https://github.com/SummerVideoAnalysis/Key-Actor-Detection/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=setuptools.find_packages(),
    install_requires = ['ax-platform==0.2.1', "mlflow==1.19.0", 'pytorch-lightning==1.4.3', 'torch==1.9.0', 'torchmetrics==0.5.0', 'torchvision==0.10.0'],
    python_requires=">=3.7",
)