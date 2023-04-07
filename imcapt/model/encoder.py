import torch 
import torchvision 
import pytorch_lightning as L

class Encoder(L.LightningModule):
    """Extracts image features

    Encoder is a model intended to extract 
    useful features from input images and convert
    them into appropriate format for decoder network
    """

    def __init__(self, 
                 feature_map_size: int, 
                 encoder_size: int,
                 dropout: int,
                 backbone: torch.nn.Module,
                 transforms: torch.nn.Module,
                 cut_layers_from=5,
                 fine_tune=False) -> None:
        """
        Pretrained model with adaptive pool

        Model consists of the following parts:

        1. Pre-trained feature-extractor module
        2. Average pooling and linear layer to convert 
           input for decoder
        3. Dropout and batch normalization regularization techniques

        All the ResNet50 parameters are turned off for training

        Args:
            feature_map_size (int): The size of feature maps (for average pooling)
            encoder_size (int): The size of layer
            dropout (int): The probability for dropout layer 
            fine_tune (bool, optional): 
                If set to True, parameters starting from fifth layer in ResNet 
                will be enabled to train/ Defaults to False.
            backbone (torch.Module):
                Instance of pre-trained feature extractor
            transforms (torch.Module):
                Transforms to be performed before supply to
                backbone model
            cut_layers_from (int, optional):
                The starting index from left to
                enable layers in encoder if ~fine_tune=True
        """
        super().__init__()

        # The main block of the encoder network is a
        # pretrained ResNet18 model.
    
        # To adapt this model for our needs, the last fully 
        # connected layer is eliminated.
        # Instead, use an adaptive average pooling.

        
        self.backbone = backbone
        self.transforms = transforms
        
        
        self.image_recognition = torch.nn.Sequential(*self.backbone.children())[:-2]
        self.adapool = torch.nn.AdaptiveAvgPool2d((feature_map_size, feature_map_size,))
        self.f_linear = torch.nn.Linear(2048, encoder_size)

        # The main part of the encoder network is implemented, the
        # rest is devoted for image transforms.
        
        # Activations and regularizations
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm = torch.nn.BatchNorm2d(feature_map_size)

        # Disable parameters 
        for p in self.image_recognition.parameters():
            p.requires_grad = False
            
        # Enable particular laeyrs in fine-tune is enabled
        for c in list(self.image_recognition.children())[cut_layers_from:]:
            for p in c.parameters():
                p.requires_grad=fine_tune

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Extracts features from images

        Args:
            X (torch.Tensor): 
                Image batch of shape (batch_size, image_channels, image_height, image_width)

        Returns:
            torch.Tensor: 
                Image features as a tensor 
                of shape (batch_size, feature_map_size ** 2, encoder_size)
        """
        X = self.transforms(X)
        X = self.image_recognition(X)
        X = self.adapool(X) 
        X = X.permute(0, 2, 3, 1)
        X = self.batch_norm(X)  
        X = self.f_linear(X)
        X = self.dropout(self.relu(X))
        return X.flatten(start_dim=1, end_dim=-2)