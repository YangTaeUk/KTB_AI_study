import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 데이터 증강 변환 설정
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    transforms.RandomRotation(30),  # 최대 30도 회전
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # 랜덤 크롭
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화
    transforms.ToTensor(),  # 텐서 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# CIFAR-10 데이터셋 로드 (학습용)
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 증강된 데이터 확인
data_iter = iter(train_loader)
images, labels = next(data_iter)

# 이미지 시각화
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 첫 번째 배치의 이미지 출력
imshow(torchvision.utils.make_grid(images[:8]))
