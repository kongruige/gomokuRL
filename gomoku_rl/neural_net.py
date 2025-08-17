import torch
import torch.nn as nn
import torch.nn.functional as F

# First, we define the basic building block for our ResNet.
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # The "skip connection"
        out = F.relu(out)
        return out

# Now we build the main network using these residual blocks.
class GomokuNet(nn.Module):
    def __init__(self, board_size=13, num_residual_blocks=3, num_channels=64):
        super(GomokuNet, self).__init__()
        self.board_size = board_size

        # Initial convolutional layer
        self.conv_in = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        # Stack of residual blocks
        self.residual_layers = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        )

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # Residual blocks
        for block in self.residual_layers:
            x = block(x)
        
        # --- Policy Head ---
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # --- Value Head ---
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

# We can test this new architecture with the same __main__ block as before.
if __name__ == '__main__':
    board_size = 13
    dummy_input = torch.randn(2, 3, board_size, board_size) # Batch size of 2
    net = GomokuNet(board_size=board_size, num_residual_blocks=3, num_channels=32)
    
    print("Testing Shallow ResNet architecture...")
    policy_output, value_output = net(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Policy output shape: {policy_output.shape}")
    print(f"Value output shape: {value_output.shape}")

    expected_policy_shape = (2, board_size * board_size)
    expected_value_shape = (2, 1)
    assert policy_output.shape == expected_policy_shape, "Policy shape is incorrect!"
    assert value_output.shape == expected_value_shape, "Value shape is incorrect!"
    print("\nShallow ResNet architecture is sound.")