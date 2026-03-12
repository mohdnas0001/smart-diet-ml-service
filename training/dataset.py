"""
Food image dataset class for PyTorch training.
Expects ImageFolder-style directory structure:
    root/
        class_name_1/
            img1.jpg
            img2.jpg
        class_name_2/
            ...
"""
from torchvision.datasets import ImageFolder


class FoodDataset(ImageFolder):
    """Wrapper around ImageFolder with food-specific helpers."""

    @property
    def classes_count(self):
        return len(self.classes)
