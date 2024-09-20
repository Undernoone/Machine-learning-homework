import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

# class FaceClassifier(nn.Module):
#     def __init__(self):
#         super(FaceClassifier, self).__init__()
#         self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.resnet(x)
#         return self.sigmoid(x)
#
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
# train_dataset = datasets.ImageFolder(root='dataset/train', transform=data_transforms['train'])
# val_dataset = datasets.ImageFolder(root='dataset/val', transform=data_transforms['val'])
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
# model = FaceClassifier()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# def train_model(model, criterion, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             labels = labels.unsqueeze(1).float()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
#
#     print('Finished Training')
#
# train_model(model, criterion, optimizer)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 使用ResNet50作为特征提取器
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128 * 128 * 3)  # 输出图像的形状 (128, 128, 3)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 3, 128, 128)  # 重新调整形状为图像
        return torch.tanh(x)  # 使用tanh激活函数确保输出值在[-1, 1]范围内

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # 判别器输出一个值来表示图像的真实性
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

train_dataset = datasets.ImageFolder(root='dataset/train', transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练函数
def train_gan(generator, discriminator, criterion, optimizer_g, optimizer_d, num_epochs=10):
    for epoch in range(num_epochs):
        for i, (inputs, _) in enumerate(train_loader):
            batch_size = inputs.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # 训练判别器
            optimizer_d.zero_grad()
            outputs = discriminator(inputs)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, 100)  # 噪声输入
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                      f'D Loss: {d_loss_real.item() + d_loss_fake.item()}, G Loss: {g_loss.item()}')

    print('Finished Training')

train_gan(generator, discriminator, criterion, optimizer_g, optimizer_d)
