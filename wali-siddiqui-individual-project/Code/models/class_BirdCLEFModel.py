import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models

import timm


class BirdCLEFModel(nn.Module):
    """
    Neural network model for BirdCLEF classification using a timm backbone and optional mixup augmentation.

    Args:
        cfg (object): Configuration object containing model parameters, including:
            - model_name (str): Name of the timm model architecture (e.g., 'resnet18', 'efficientnet_b0').
            - pretrained (bool): Whether to use pretrained ImageNet weights.
            - in_channels (int): Number of input channels (e.g., 1 for grayscale spectrograms).
            - taxonomy_csv (str): Path to taxonomy CSV file for determining number of classes.
            - mixup_alpha (float, optional): Alpha value for mixup augmentation. Enables mixup if > 0.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        """
        Forward pass through the model with optional mixup during training.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            targets (torch.Tensor, optional): Ground truth labels for mixup loss calculation.

        Returns:
            torch.Tensor: Logits output of the classifier if not training with mixup.
            tuple: (logits, loss) if training and mixup is applied.
        """
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        if isinstance(features, dict):  # Handle some timm models with dict outputs
            features = features['features']

        if len(features.shape) == 4:  # Apply global pooling if feature maps are 2D
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits,
                                        logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        """
        Apply mixup augmentation to a batch of input data and labels.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            targets (torch.Tensor): Target labels tensor of shape (B, num_classes).

        Returns:
            tuple:
                - mixed_x (torch.Tensor): Mixed input tensor.
                - targets (torch.Tensor): Original targets.
                - targets[indices] (torch.Tensor): Shuffled targets.
                - lam (float): Mixup interpolation coefficient.
        """
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Compute mixup loss as a weighted average of two targets' losses.

        Args:
            criterion (function): Loss function to apply (e.g., BCEWithLogits).
            pred (torch.Tensor): Model predictions.
            y_a (torch.Tensor): First set of targets.
            y_b (torch.Tensor): Second set of targets.
            lam (float): Mixup interpolation coefficient.

        Returns:
            torch.Tensor: Combined loss value.
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
