import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=4):
        super(ChessResNet, self).__init__()
        
        # Input Block: 13 channels -> 64 filters
        self.conv_input = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        
        # Residual Core
        self.res_blocks = nn.ModuleList([ResidualBlock(64, 64) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)
        
        # Value Head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input block
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p) # Outputs raw logits
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # Output in [-1, 1]
        
        return p, v
