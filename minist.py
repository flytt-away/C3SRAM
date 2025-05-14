import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 定义超参数
batch_size = 64
learning_rate = 0.01
num_epochs = 50
train_flag = 0
epsilon_en = 1
noise_scale = 1
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
print("Try to load MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# 定义3层线性网络
class ThreeLayerNet(nn.Module):
    def __init__(self, RMES = 0.0625,epsilon = 1, noise_scale = 5):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(512, 256)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(256, 10)  # 隐藏层2到输出层
        self.relu = nn.ReLU()
        self.RMES = RMES 
        self.epsilon = epsilon
        self.noise_scale = noise_scale

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像
        # 1st layer
        x = self.fc1(x)
        if self.epsilon == 1 :
            noise = torch.normal(0, std=self.RMES, size=x.shape, device=x.device, dtype=x.dtype)
            x = x + noise * self.noise_scale
        x = self.relu(x)
        # 2nd layer
        x = self.fc2(x)
        if self.epsilon == 1 :
            noise = torch.normal(0, std=self.RMES, size=x.shape, device=x.device, dtype=x.dtype)
            x = x + noise * self.noise_scale
        x = self.relu(x)
        # 3rd layer 
        x = self.fc3(x)
        if self.epsilon == 1 :
            noise = torch.normal(0, std=self.RMES, size=x.shape, device=x.device, dtype=x.dtype)
            x = x + noise * self.noise_scale
        return x


# 初始化模型、损失函数和优化器


print("Load model...")

model = ThreeLayerNet(RMES = 0.0625,epsilon = epsilon_en, noise_scale = noise_scale)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if(train_flag != 1):
    model.load_state_dict(torch.load('mnist_model.pt'))
    model.eval()  # 设置为评估模式


# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f} \tAccuracy: {accuracy:.2f}%')


# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return accuracy


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练和测试循环

if(train_flag == 1): 
    test_accuracies = []
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        test_accuracies.append(accuracy)

    # Save model
    torch.save(model.state_dict(), 'mnist_model.pt')
    # 绘制准确率曲线
    plt.plot(range(1, num_epochs + 1), test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy over Epochs')
    plt.show()

else :
    test_accuracies = []
    for i in range(0, 64+1):
        #print(f"scale: {noise_scale_loop}")
        model = ThreeLayerNet(RMES = 0.0625,epsilon = epsilon_en, noise_scale = i)
        model.load_state_dict(torch.load('mnist_model.pt'))
        model.eval()  # 设置为评估模式
        model = model.to(device)
        accuracy = test(model, device, test_loader)
        test_accuracies.append(accuracy)
    
import csv

with open('test_accuracies.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(test_accuracies)  # 写入一行




