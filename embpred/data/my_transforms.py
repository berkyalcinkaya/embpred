import torch

class ShuffleColor:
    """
    Shuffle the color channels of a PyTorch tensor image.
    
    Args:
        None
    """
    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W)
        
        Returns:
            torch.Tensor: Image tensor with shuffled color channels.
        """
        if img.shape[0] != 3:
            raise ValueError("Image tensor must have 3 color channels (C=3)")
        
        indices = torch.randperm(3)
        return img[indices, :, :]