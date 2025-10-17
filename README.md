# noise-robustness

Robustness experimental framework for image classification under label noise

This repository explores robustness strategies for image classification under label noise, using methods such as Co-teaching. It provides a reproducible baseline for evaluating model performance under varying noise rates.

## Dataset Instructions

This project uses the CIFAR-10 image classification dataset. Due to GitHub's file size limitations, the dataset is not included in this repository.

### Steps to prepare the dataset:

1. Download the archive from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
2. Extract the contents and place the `cifar-10-batches-py/` folder inside the `data/` directory at the project root.
3. Ensure the directory structure is correct, or the data loader will fail to locate the dataset.

## Quick Start

### Install dependencies:
```bash
pip install -r requirements.txt

Run experiments:
python run_experiments.py --noise_rate 0.2 --seed 123
You can adjust parameters such as --noise_rate and --seed to control the noise rate, random seed, and other settings.

Method Summary

Co-teaching: Two networks collaboratively select small-loss samples for training, reducing the impact of noisy labels.
Noise Simulation: Label corruption is applied at configurable rates to simulate different levels of noise.
Result Logging: All outputs are saved in the results/ directory as .npy files for reproducibility and analysis.

Experimental Results
The table below shows the average test accuracy across 9 random seeds under varying noise rates, comparing the baseline and Co-teaching methods:





































































































Noise RateBaseline Accuracy (%)Co-teaching Accuracy (%)Average Accuracy (%)0.0085.1983.7084.440.0583.8983.9283.910.1083.2083.9283.560.1582.3683.3982.880.2081.1782.1281.650.2580.4180.8780.640.3079.2179.5579.380.3577.9578.1778.060.4076.5576.8676.710.4575.2775.3875.330.5073.2273.5473.380.5570.7671.6871.220.6068.1269.0068.560.6565.1765.1765.170.7060.3858.0359.21
Each row represents the average accuracy across 9 seeds for a given noise rate. Co-teaching consistently outperforms the baseline as noise levels increase.
Project Structure

noisy_label_co_teaching.py: Core algorithm implementation
run_experiments.py: Experiment entry point
data/: Dataset directory (not included)
results/: Output directory for experiment results
requirements.txt: Dependency list
README.md: Project documentation
LICENSE: License file

License
This project is licensed under the MIT License. See the LICENSE file for details.
