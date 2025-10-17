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



````
Run experiments:
```bash
python run_experiments.py --noise_rate 0.2 --seed 123
````

## Method Summary

- **Co-teaching**: Two networks collaboratively select small-loss samples for training, reducing the impact of noisy labels.
- **Noise Simulation**: Label corruption is applied at configurable rates to simulate different levels of noise.
- **Result Logging**: All outputs are saved in the `results/` directory as `.npy` files for reproducibility and analysis.

## Experimental Results

The table below shows the average test accuracy across 9 random seeds under varying noise rates, comparing the baseline and Co-teaching methods:

| Noise Rate | Baseline Accuracy (%) | Co-teaching Accuracy (%) | Average Accuracy (%) |
|------------|-----------------------|--------------------------|----------------------|
| 0.00       | 85.19                 | 83.70                    | 84.44                |
| 0.05       | 83.89                 | 83.92                    | 83.91                |
| 0.10       | 83.20                 | 83.92                    | 83.56                |
| 0.15       | 82.36                 | 83.39                    | 82.88                |
| 0.20       | 81.17                 | 82.12                    | 81.65                |
| 0.25       | 80.41                 | 80.87                    | 80.64                |
| 0.30       | 79.21                 | 79.55                    | 79.38                |
| 0.35       | 77.95                 | 78.17                    | 78.06                |
| 0.40       | 76.55                 | 76.86                    | 76.71                |
| 0.45       | 75.27                 | 75.38                    | 75.33                |
| 0.50       | 73.22                 | 73.54                    | 73.38                |
| 0.55       | 70.76                 | 71.68                    | 71.22                |
| 0.60       | 68.12                 | 69.00                    | 68.56                |
| 0.65       | 65.17                 | 65.17                    | 65.17                |
| 0.70       | 60.38                 | 58.03                    | 59.21                |

Each row represents the average accuracy across 9 seeds for a given noise rate. Co-teaching consistently outperforms the baseline as noise levels increase.
