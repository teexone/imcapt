import torch 
import torchvision 
import pytorch_lightning as L

class Encoder(L.LightningModule):
    """
    Encoder is a residual convolutional neural network 
    that is used to extract features from visual data
    """

    def __init__(self, feature_map_size: int, encoder_size: int, dropout: int) -> None:
        """Pretrained ResNet-18 with adaptive pool

        Args:
            feature_map_size: 
                A size (int) of encoded image size
            embedding_size:
                A size (int) of embedding vector
        """
        super().__init__()

        # The main block of the encoder network is a
        # pretrained ResNet18 model.
    
        # To adapt this model for our needs, the last fully 
        # connected layer is eliminated.
        # Instead, I am using an adaptive average pooling
        # to make input fit the embedding vector size.

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.image_recognition = torch.nn.Sequential(*resnet50.children())[:-2]
        self.adapool = torch.nn.AdaptiveAvgPool2d((feature_map_size, feature_map_size,))
        self.f_linear = torch.nn.Linear(2048, encoder_size)
        # The main part of the encoder network is implemented, the
        # rest is devoted for image transforms.
        
        
        # These are necessary transforms for the image 
        # fed to ResNet18
        self.transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm = torch.nn.BatchNorm2d(feature_map_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Extracts image features

        Produces a feature map of size (14, 14, E)
        where E is a dimension of embedding vector.

        Args:
            X: 
                input image as a torch.Tensor of sizes 
                (256, 256) or (B, 256, 256) where B is a
                size of a batch 

        Returns:
            A torch.Tensor of size (B, embedding_size)
            in case of batched input
        """
        X = self.transforms(X)
        X = self.image_recognition(X)
        X = self.adapool(X) 
        X = X.permute(0, 2, 3, 1)
        X = self.batch_norm(X)  
        X = self.f_linear(X)
        X = self.dropout(self.relu(X))
        return X.flatten(start_dim=1, end_dim=-2)