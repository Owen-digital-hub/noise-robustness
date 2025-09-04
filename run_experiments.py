import sys
import os
import csv
import numpy as np
import inspect

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from noisy_label_co_teaching import train_model

def run_experiments():

    # 实验参数配置
    noise_rates = np.round(np.arange(0.0, 0.71, 0.05), 2).tolist()
    seeds   = [761]
    methods = ["coteaching", "baseline"]
    epochs  = 50
    results = []

    # 创建 results 目录和 CSV 文件
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "acc_summary.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Noise Rate", "Seed", "Accuracy"])

    # 遍历所有组合并运行实验
    for seed in seeds:
        for noise in noise_rates:
            for method in methods:
                print(f"\n=== Running: method={method}, noise={noise:.2f}, seed={seed} ===")
                print(f"✅ train_model 来源: {inspect.getfile(train_model)}")

                try:
                    print(f"🔥 开始训练 | 方法: {method} | 噪声率: {noise:.2f} | 种子: {seed}")
                    acc, loss_list = train_model(noise_rate=noise, seed=seed, method=method, epochs=epochs)
                    results.append({
                        'method': method,
                        'noise_rate': noise,
                        'seed': seed,
                        'accuracy': acc
                    })

                    # 写入 CSV
                    with open(csv_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([method, noise, seed, round(acc, 2)])

                    # 保存 loss 曲线
                    method_dir = os.path.join(results_dir, method)
                    os.makedirs(method_dir, exist_ok=True)
                    loss_path = os.path.join(method_dir, f"loss_noise{noise:.2f}_seed{seed}.npy")
                    np.save(loss_path, np.array(loss_list))

                except Exception as e:
                    print(f"❌ Error at method={method}, noise={noise}, seed={seed}: {e}")
                    continue
                
    # 输出实验结果
    print("\n📊 Experiment Summary:")
    for r in results:
        print(f"{r['method'].capitalize():<10} | Noise={r['noise_rate']:.2f} | Seed={r['seed']} | Acc={r['accuracy']:.2f}%")

# 主程序入口
if __name__ == "__main__":
    print("✅ run_experiments.py 已启动")
    run_experiments()
