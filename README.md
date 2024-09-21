# DrtNet-FA
Dual-Residual Transformer Network with Fusion Attention

## Installation

You can install the required libraries using pip:

```bash
pip install numpy==1.26.4 torch==2.1.2+cu121 scipy==1.11.1 tqdm==4.65.0 tensorboardX==2.6.2.2

## Environment Requirements

- **CUDA Version**: 12.1

## Data Input Format

The model training and testing both use `.npz` files as input. The data should be structured in a dictionary format, with the following keys:

- `label`: Contains the labels for the images.
- `image`: Contains the image data.

Additionally, you need to create two text files to specify the data files:

1. **Training File**: Named `train.txt`
2. **Testing File**: Named `test_vol.txt`

These text files should list the corresponding `.npz` data file names. Please ensure that there are no leading or trailing empty lines in the text files.

## Model Overview

This model is based on the corresponding code of TransUNet. We utilized its training files, testing files, and other auxiliary files, making specific modifications to the input and output ports.

In terms of the model architecture:
- The original Transformer components have been retained.
- The input and output ports of the Transformer have been optimized.
- The entire CNN structure has been completely reconstructed.
- All pre-trained components from the original model have been removed.

These changes enhance the model's performance and adaptability for our specific tasks.
