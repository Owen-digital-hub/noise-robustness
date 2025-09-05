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



noise-robustness
标签噪声下图像分类的鲁棒性实验框架

本项目旨在研究标签噪声条件下图像分类模型的鲁棒性，采用 Co-teaching 等方法进行实验验证。提供一个可复现的基线，用于评估模型在不同噪声率下的表现。

数据集说明： 本项目使用 CIFAR-10 图像分类数据集。由于 GitHub 文件大小限制，数据集未包含在仓库中。 数据准备步骤如下：

从 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 下载数据集压缩包

解压后将 cifar-10-batches-py/ 文件夹放入项目根目录下的 data/ 文件夹中

确保目录结构正确，否则数据加载模块将无法运行

快速开始： 使用 pip install -r requirements.txt 安装依赖，然后运行 python run_experiments.py --noise_rate 0.2 --seed 123 启动实验。可通过参数控制噪声率、随机种子等设置。

方法简介：

Co-teaching：两个模型协同选择小损失样本进行训练

噪声模拟：通过标签扰动构造不同噪声率

结果记录：所有输出保存在 results/ 文件夹中，格式为 .npy

项目结构：

noisy_label_co_teaching.py：核心算法实现

run_experiments.py：实验入口脚本

data/：数据集目录（未包含）

results/：实验结果输出目录

requirements.txt：依赖列表

README.md：项目说明文档

LICENSE：许可证文件

许可证： 本项目遵循 MIT License，详见 LICENSE 文件。
