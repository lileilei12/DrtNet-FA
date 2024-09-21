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
