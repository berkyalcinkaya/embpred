import torch

def load_serialized_model(serialized_model_path, device=torch.device('cpu')):
    """
    Loads a serialized TorchScript model.

    Parameters:
    - serialized_model_path (str): Path to the serialized TorchScript model (.pt file).
    - device (torch.device, optional): Device to load the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
    - torch.jit.ScriptModule: The loaded TorchScript model ready for inference.
    
    Raises:
    - FileNotFoundError: If the serialized model file does not exist.
    - RuntimeError: If the model fails to load.
    """
    # Check if the serialized model file exists
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("CUDA is not available. Loading the model on CPU instead.")
        device = torch.device('cpu')
    
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')

    if not torch.cuda.is_available() and device.type == 'cuda':
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device('cpu')

    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')

    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')

    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')

    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')

    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')

    if not os.path.isfile(serialized_model_path):
        raise FileNotFoundError(f"Serialized model file not found at: {serialized_model_path}")

    try:
        # Load the serialized TorchScript model
        model = torch.jit.load(serialized_model_path, map_location=device)
        model.eval()  # Set the model to evaluation mode
        print(f"Successfully loaded serialized model from {serialized_model_path} on {device}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the serialized model: {e}")

# Example Usage
if __name__ == "__main__":
    import os

    # Paths
    serialized_model_path = "./models/smaller_net3d224_traced.pt"  # Replace with your serialized model path

    # Choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the serialized model
    model = load_serialized_model(serialized_model_path, device=device)

    # Example inference
    # Create a sample input tensor with the same input shape used during serialization
    # Ensure that the input tensor is on the same device as the model
    sample_input = torch.randn(1, 3, 224, 224).to(device)  # Batch size of 1

    # Perform inference
    with torch.no_grad():
        output = model(sample_input)

    print("Model Output:", output)