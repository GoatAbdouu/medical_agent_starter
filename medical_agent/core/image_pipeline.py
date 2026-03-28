"""
Image Processing Pipeline for Medical Vision AI Agent.

Provides consistent preprocessing, data augmentation (training), and
inference transforms so that all three ensemble models receive images in
exactly the same format.
"""
from typing import Tuple

from PIL import Image
import torch
from torchvision import transforms

# ImageNet normalisation constants (used by all three backbone models)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGE_SIZE: int = 224


class ImagePipeline:
    """
    Centralised image pre-processing and augmentation pipeline.

    Usage
    -----
    pipeline = ImagePipeline()

    # For inference
    tensor = pipeline.preprocess(pil_image)          # shape: (1, 3, 224, 224)

    # For training data-loaders
    train_tf = pipeline.get_train_transforms()
    val_tf   = pipeline.get_val_transforms()
    """

    def __init__(self) -> None:
        self._inference_tf = self.get_inference_transforms()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Pre-process a single PIL image for model inference.

        Converts the image to RGB if necessary, resizes to 224×224,
        normalises with ImageNet statistics, and adds a batch dimension.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image in any PIL-compatible mode.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(1, 3, 224, 224)``.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._inference_tf(image).unsqueeze(0)

    @staticmethod
    def get_train_transforms() -> transforms.Compose:
        """
        Return the full data-augmentation pipeline used during training.

        Augmentations applied (in order)
        ---------------------------------
        - Resize to 256×256
        - RandomResizedCrop to 224×224 (scale 0.7–1.0)
        - RandomHorizontalFlip
        - RandomVerticalFlip
        - RandomRotation ±30°
        - RandomAffine (translate 10 %, shear 10°)
        - ColorJitter (brightness / contrast / saturation / hue)
        - GaussianBlur (kernel 3, σ 0.1–2.0)
        - ToTensor + ImageNet normalisation
        """
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def get_val_transforms() -> transforms.Compose:
        """
        Return the deterministic validation / evaluation pipeline.

        Only resizes and normalises — no random augmentations.
        """
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def get_inference_transforms() -> transforms.Compose:
        """
        Return the inference pipeline (identical to validation transforms).

        This method exists as an alias so call-sites can be explicit about
        their intent (inference vs validation).
        """
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
