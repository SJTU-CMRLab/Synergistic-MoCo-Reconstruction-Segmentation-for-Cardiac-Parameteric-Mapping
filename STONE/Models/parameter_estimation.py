import torch.nn as nn


class Res_MLP(nn.Module):
    # Residual MLP, 10 layers
    def __init__(self, input_dim, output_dim, width):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.mlp_block1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=width),
            nn.BatchNorm1d(num_features=width),
            nn.LeakyReLU()
        )
        self.mlp_block2 = nn.Sequential(
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width),
            nn.LeakyReLU(),
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width)
        )
        self.mlp_block3 = nn.Sequential(
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width),
            nn.LeakyReLU(),
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width)
        )
        self.mlp_block4 = nn.Sequential(
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width),
            nn.LeakyReLU(),
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width)
        )
        self.mlp_block5 = nn.Sequential(
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width),
            nn.LeakyReLU(),
            nn.Linear(in_features=width, out_features=width),
            nn.BatchNorm1d(num_features=width)
        )
        self.mlp_block6 = nn.Sequential(
            nn.Linear(in_features=width, out_features=output_dim)
        )

    def forward(self, x):
        x = self.mlp_block1(x)
        x = self.leaky_relu(self.mlp_block2(x) + x)
        x = self.leaky_relu(self.mlp_block3(x) + x)
        x = self.leaky_relu(self.mlp_block4(x) + x)
        x = self.leaky_relu(self.mlp_block5(x) + x)
        x = self.mlp_block6(x)
        return x
