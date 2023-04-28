import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much

        self.model = nn.Sequential(                                                           # -> 3  * 224 * 224
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),              # -> 16 * 224 * 224
            nn.BatchNorm2d(16),            
            nn.MaxPool2d(2, 2),      
            nn.LeakyReLU(negative_slope=0.2),
            # -> 16 * 112 * 112
            nn.Conv2d(16, 32, 3, padding=1),                                                  # -> 32 * 112 * 112
            nn.BatchNorm2d(32),             
            nn.MaxPool2d(2, 2),                                                              # -> 32 * 56 * 56
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(32, 64, 3, padding=1),                                                  # -> 64  * 56    * 56
            nn.BatchNorm2d(64),           
            nn.MaxPool2d(2, 2),  
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2),
                                                                         # -> 64  *  28  *  28                                                    
            nn.Conv2d(64, 128, 3, padding=1),                                                  # -> 64  * 56    * 56
            nn.BatchNorm2d(128),          
            nn.MaxPool2d(2, 2), 
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Flatten(),                                                                     # -> 1x 64 *  28   * 28  -> 50,176
            nn.Linear(128 * 14 * 14, 1024), 
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 500), 
            nn.BatchNorm1d(500),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(500, num_classes),
           
            

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        return self.model(x)



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
