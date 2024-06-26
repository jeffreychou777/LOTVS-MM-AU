import torch
file = "/home/ubuntu/yk/X-CLIP/results/best.pth"
checkpoint = torch.load(file, map_location='cpu')
print(checkpoint['epoch'])