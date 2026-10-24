"""Run this file in PyCharm to check the selected GPU training environment."""
import sys
import importlib.metadata as metadata

import numpy as np
import torch
import torchvision
import pandas
import matplotlib
import sklearn
from torchsummary import summary


def main():
    print('Python executable:', sys.executable)
    print('Python version:', sys.version.split()[0])
    for name in ('torch', 'torchvision', 'numpy', 'pandas', 'matplotlib',
                 'scikit-learn', 'torchsummary'):
        print(name + ':', metadata.version(name))
    print('CUDA runtime:', torch.version.cuda)
    print('cuDNN version:', torch.backends.cudnn.version())
    print('CUDA available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable: select the Conda pytorch interpreter.')
    print('GPU:', torch.cuda.get_device_name(0))
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 2)
    ).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    images = torch.randn(8, 3, 32, 32, device='cuda')
    labels = torch.arange(8, device='cuda') % 2
    for step in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        assert torch.isfinite(loss).item()
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters())
        optimizer.step()
        print(f'Training step {step + 1}: loss={loss.item():.6f}')
    torch.cuda.synchronize()
    assert np.isfinite(images.cpu().numpy()).all()
    print('PASS: GPU forward, backward, optimizer and NumPy conversion.')


if __name__ == '__main__':
    main()
