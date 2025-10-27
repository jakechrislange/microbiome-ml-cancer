
import torch
import torch.nn as nn

class MLPClassifierDeepResidual(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, dropout = 0.3):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(out_channels, out_channels),
                torch.nn.LayerNorm(out_channels),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)
    
    def __init__(
        self,
        input_dim: int = 128,
        second_dim: int = 128,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, second_dim))
        layers.append(torch.nn.LayerNorm(second_dim))
        layers.append(torch.nn.GELU())
        layers.append(self.Block(second_dim, second_dim))
        layers.append(self.Block(second_dim, second_dim//2))
        layers.append(torch.nn.Linear(second_dim //2, num_classes, bias=False))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x.view(x.size(0), -1))
    
