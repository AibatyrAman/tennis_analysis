import torch

try:
    path = "models/keypoints_model_50.pt"
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'fc.weight' in checkpoint:
        print(f"fc.weight shape: {checkpoint['fc.weight'].shape}")
    elif 'fc.bias' in checkpoint:
         print(f"fc.bias shape: {checkpoint['fc.bias'].shape}")
    else:
        # Check for last layer keys
        print("Last 5 keys:")
        print(list(checkpoint.keys())[-5:])

except Exception as e:
    print(f"Error: {e}")
