# DDIM Image Generation with FID Evaluation

This repository contains an implementation of Denoising Diffusion Implicit Models (DDIM) for image generation, along with Fréchet Inception Distance (FID) calculation for evaluation.

## Overview

The project implements:
- A U-Net architecture optimized for 128x128 RGB images
- DDIM sampling for faster image generation compared to DDPM
- FID score calculation using Inception v3 features
- Batch generation and evaluation pipeline

## Requirements

```
torch
torchvision 
PIL
numpy
scipy
matplotlib
```

## Model Architecture

The U-Net model features:
- Encoder-decoder architecture with skip connections
- Time embeddings injected at each block
- BatchNorm and ReLU activations
- Optimized for 128x128 RGB images

## Usage

### Generate Images

```python
# Load model
model = UNet().to(device)
model.load_state_dict(torch.load('model_weights.pth'))

# Generate images using DDIM
generated_images = ddim_sample(
    model=model,
    image_size=64,
    batch_size=4,
    channels=3
)
```

### Calculate FID Score

```python
# Extract Inception features
real_features = extract_inception_features(real_image_paths)
generated_features = extract_inception_features(generated_image_paths)

# Calculate FID
fid_score = calculate_frechet_inception_distance(
    real_mean, real_cov,
    generated_mean, generated_cov
)
```

## Key Features

- **DDIM Sampling**: Implements deterministic sampling requiring fewer steps than DDPM
- **Batch Processing**: Supports batch generation for improved efficiency
- **FID Evaluation**: Complete pipeline for calculating FID scores
- **Feature Extraction**: Uses pre-trained Inception v3 for feature extraction
- **Error Handling**: Robust error handling for image loading and processing

## Implementation Details

- Uses linear beta schedule for noise scheduling
- Implements time embedding using sinusoidal positions
- Supports both CPU and CUDA devices
- Saves generated images and feature statistics

## License

[Add your chosen license]

## Citation

[Add relevant citations]

## Contributing

[Add contribution guidelines]
