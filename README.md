# mammogram-analysis-mim
Self-Supervised Mammogram Analysis through Masked Image Modeling: A Dual-Stage Framework for Interpretable and Label-Efficient Cancer Diagnosis

This work introduces a novel dual-stage deep learning framework for mammography analysis, integrating self-supervised pre-training (Masked Image Modeling) with semi-supervised classification. The approach targets key challenges including: limited labeled data, annotation scarcity, high annotation costs, potential labeling errors, limited transferability, and the need for model explainability in medical imaging.

### Key Benefits

- **Reduced Annotation Requirements**: Leverages self-supervised pre-training to utilize unlabeled mammogram archives
- **Enhanced Clinical Decision Support**: Provides rich feature representations with transparent explanations via attention maps and Grad-CAM visualizations
- **Improved Transferability**: MIM-pre-trained encoder provides a robust foundation for adapting to various tasks and modalities
- **Efficient Label Usage**: Context-aware features enable strong performance even with limited labeled data


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
