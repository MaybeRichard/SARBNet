# SARBNet
Self-supervised 3D despeckling for optical coherence tomography via anisotropic blind-spot learning.

## Repository layout
- `model.py` – definition of all neural network building blocks, the `BlindSpotUNet3D_SAG` model, and helper utilities such as `create_model` and `get_model_info`.
- `projEnface.m` – MATLAB function to project despeckled OCT sub-volumes into en face views using mean, max, or variance projections across a chosen depth range.
- Full training/inference scripts, data-loading utilities, and experiment configurations will be uploaded soon to complete the SARBNet release.

## Requirements
| Component | Version hints |
| --- | --- |
| Python | 3.9+ |
| PyTorch | 1.13+ (CUDA strongly recommended for 3D volumes) |
| NumPy | Optional, for data pre/post-processing |
| MATLAB | R2021a+, required only for the en face projection script |

> The repository intentionally focuses on the core network definition and reconstruction utility. Additional training, evaluation, and deployment resources will follow in upcoming commits.
