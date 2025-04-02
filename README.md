# Self-Supervised Mammogram Analysis through Masked Image Modeling: A Dual-Stage Framework for Interpretable and Label-Efficient Cancer Diagnosis

This work introduces a novel dual-stage deep learning framework for mammography analysis that integrates self-supervised pre-training via Masked Image Modeling (MIM) with semi-supervised classification. The approach addresses critical challenges in medical imaging analysis: limited labeled data, annotation scarcity, high annotation costs, potential labeling errors, limited transferability, and the need for model explainability.

## Technical Architecture

### Self-Supervised Pre-Training Stage
- **Random Blockwise Masking**: Implements a 16×16 patch-based masking strategy with a 0.5 ratio, maintaining local structural information crucial for preserving anatomical patterns in mammograms
- **Swin Transformer Encoder**: Employs a hierarchical vision transformer with shifted window-based self-attention that captures both local tissue details and global contextual relationships across multiple scales
- **Lightweight Convolutional Decoder**: Features progressive upsampling (1024→512→256→128→64 channels) and L1 pixel-wise reconstruction loss optimization specifically chosen over MSE for superior preservation of fine-grained mammographic details (microcalcifications, spiculated masses, architectural distortions)

### Semi-Supervised Classification Stage
- **Multi-Task Classification Head**: Built on the pre-trained encoder with class-weighted cross-entropy loss (wc = N/2Nc) to effectively handle the inherent class imbalance typical in pathology datasets
- **Explainability Mechanisms**: Implements dual visualization approaches:
  1. Attention map extraction from the final layer's multi-head self-attention module
  2. Gradient-weighted class activation mapping (Grad-CAM) adapted specifically for transformer architectures with gradient-based weighting of attention heads

## Key Technical Benefits

- **Reduced Annotation Requirements**: Leverages unsupervised pre-training on unlabeled mammography data, effectively learning salient imaging biomarkers without expert annotations through reconstruction-based learning objectives
- **Enhanced Clinical Decision Support**: Combines high classification performance with clinically relevant visual explanations that align with radiological assessment patterns, enabling verification of model reasoning and improving clinician trust
- **Architectural Transferability**: The hierarchical multi-scale representation learning framework transfers effectively across varying imaging protocols (digital mammography, tomosynthesis) and downstream tasks (density assessment, lesion detection, risk prediction)
- **Label Efficiency**: Achieves strong performance with as little as 10-50% of labeled data through the context-rich feature representations encoded in the Swin Transformer backbone, enabling pathology-verified labeling to be used economically

This approach demonstrates how self-supervised learning can be effectively adapted to the unique challenges of mammography analysis, where both pixel-level detail and global context are critical for accurate diagnosis.

## Running the Code

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --root_dir | Required | Path to directory containing CBIS-DDSM CSV files |
| --dicom_path | Optional | Path to text file containing DICOM file paths |
| --output_dir | ./outputs | Directory to save checkpoints/results |
| --mim_epochs | 3 | Number of epochs for MIM pre-training |
| --cls_epochs | 3 | Number of epochs for classification training |
| --missing_label_percentage | 100 | Percentage of labeled training data to use |
| --batch_size | 4 | Batch size for dataloaders |
| --num_workers | 2 | Number of worker processes for data loading |
| --train_mim | False | Flag to train MIM model from scratch |
| --train_cls | False | Flag to train classification model |
| --visualize | False | Flag to generate visualizations |
| --log_mode | console | Controls logging behavior |

### Example Usage

```bash
python mammogram_analysis.py \
    --root_dir /path/to/CBIS-DDSM \
    --dicom_path /path/to/dcm_files.txt \
    --output_dir ./outputs \
    --mim_epochs 5 \
    --cls_epochs 10 \
    --batch_size 4 \
    --num_workers 2 \
    --train_mim \
    --train_cls \
    --visualize
