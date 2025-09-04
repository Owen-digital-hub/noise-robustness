def train_model(noise_rate=0.2, seed=42, method="coteaching", epochs=50):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
    import numpy as np
    import random
    import os

    # 完全复现性设置
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 超参配置
    num_classes     = 10
    batch_size      = 128
    initial_lr      = 0.1
    weight_decay    = 5e-4
    momentum        = 0.9
    max_forget_rate = 0.2
    step_size       = 20
    gamma           = 0.1
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers     = min(2, os.cpu_count())

    # 数据预处理与加载
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    # 注入噪声标签
    noisy_labels = np.array(trainset.targets)
    n_noisy = int(noise_rate * len(noisy_labels))
    if n_noisy > 0:
        noisy_idx = np.random.choice(len(noisy_labels), n_noisy, replace=False)
        noisy_labels[noisy_idx] = [
            np.random.choice([l for l in range(num_classes) if l != orig])
            for orig in noisy_labels[noisy_idx]
        ]
        trainset.targets = noisy_labels.tolist()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers)
    
    # 测试准确率函数
    def test_accuracy(net):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return 100. * correct / total

    # 模型构造函数
    
    def build_resnet():
        net = models.resnet18()
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net.to(device)
    
    # Baseline
    if method == "baseline":
        net = build_resnet()
        opt = optim.SGD(net.parameters(), lr=initial_lr,
                        momentum=momentum, weight_decay=weight_decay)
        sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
        criterion = nn.CrossEntropyLoss()
        epoch_loss = []

        for epoch in range(epochs):
            net.train()
            running = 0.
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                opt.zero_grad()
                out = net(inputs)
                loss = criterion(out, labels)
                loss.backward()
                opt.step()
                running += loss.item()
            sch.step()
            avg_loss = running / len(trainloader)
            epoch_loss.append(avg_loss)
            print(f"[Baseline] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        acc = test_accuracy(net)
        print(f"[Baseline] Final Accuracy: {acc:.2f}%")
        return acc, epoch_loss

    # Co-teaching

    elif method == "coteaching":
        print(f"🔥 Co-teaching 开始训练 | 噪声率={noise_rate} | 种子={seed}")
        net1 = build_resnet()
        net2 = build_resnet()
        opt1 = optim.SGD(net1.parameters(), lr=initial_lr,
                         momentum=momentum, weight_decay=weight_decay)
        opt2 = optim.SGD(net2.parameters(), lr=initial_lr,
                         momentum=momentum, weight_decay=weight_decay)
        sch1 = optim.lr_scheduler.StepLR(opt1, step_size=step_size, gamma=gamma)
        sch2 = optim.lr_scheduler.StepLR(opt2, step_size=step_size, gamma=gamma)
        sample_criterion = nn.CrossEntropyLoss(reduction='none')
        epoch_loss1, epoch_loss2 = [], []

        for epoch in range(epochs):
            net1.train(); net2.train()
            running1, running2 = 0., 0.
            forget_rate = max_forget_rate * (epoch / epochs)
            remember_rate = 1 - forget_rate
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                out1 = net1(inputs)
                out2 = net2(inputs)
                l1_sample = sample_criterion(out1, labels)
                l2_sample = sample_criterion(out2, labels)

                num_remember = int(remember_rate * labels.size(0))
                _, idx1 = torch.topk(-l1_sample, num_remember)
                _, idx2 = torch.topk(-l2_sample, num_remember)

                loss1 = l1_sample[idx2].mean()
                loss2 = l2_sample[idx1].mean()

                opt1.zero_grad(); loss1.backward(); opt1.step()
                opt2.zero_grad(); loss2.backward(); opt2.step()

                running1 += loss1.item(); running2 += loss2.item()
            sch1.step(); sch2.step()
            avg_loss1 = running1 / len(trainloader)
            avg_loss2 = running2 / len(trainloader)
            epoch_loss1.append(avg_loss1)
            epoch_loss2.append(avg_loss2)
            print(f"[Co-teaching] Epoch {epoch+1}/{epochs} | Loss1: {avg_loss1:.4f} | Loss2: {avg_loss2:.4f}")

        acc1 = test_accuracy(net1)
        acc2 = test_accuracy(net2)
        avg_acc = (acc1 + acc2) / 2
        print(f"[Co-teaching] Final Accuracy: Net1={acc1:.2f}% | Net2={acc2:.2f}% | Avg={avg_acc:.2f}%")
        avg_loss = [(l1 + l2) / 2 for l1, l2 in zip(epoch_loss1, epoch_loss2)]
        return avg_acc, avg_loss

    else:
        raise ValueError(f"Unknown method: {method}")
