# noise-robustness
noise-robustness
Robustness experimental framework for image classification under label noise

This repository explores robustness strategies for image classification under label noise, using methods such as Co-teaching. It provides a reproducible baseline for evaluating model performance under varying noise rates.

Dataset Instructions: This project uses the CIFAR-10 image classification dataset. Due to GitHub's file size limitations, the dataset is not included in this repository. To prepare the dataset:

Download the archive from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Extract the contents and place the cifar-10-batches-py/ folder inside the data/ directory at the project root.

Ensure the structure is correct, or the data loader will fail to locate the dataset.

Quick Start: Install dependencies using pip install -r requirements.txt, then run experiments with python run_experiments.py --noise_rate 0.2 --seed 123. You can adjust parameters to control noise rate, seed, and other settings.

Method Summary:

Co-teaching: Two networks collaboratively select small-loss samples for training

Noise simulation: Label corruption is applied at configurable rates

Result logging: All outputs are saved in the results/ directory as .npy files

Project Structure:

noisy_label_co_teaching.py: Core algorithm implementation

run_experiments.py: Experiment entry point

data/: Dataset directory (not included)

results/: Output directory

requirements.txt: Dependency list

README.md: Project documentation

LICENSE: License file

License: This project is licensed under the MIT License.
