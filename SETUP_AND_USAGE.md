
# Setup and Usage Instructions

## Required Directory Structure

Before running the code, ensure the following directory structure:

project_root/
├── mammogram_analysis.py ....................... Main script
├── manifest-ZkhPvrLo5216730872708713142/
│   └── CBIS-DDSM/ ............................. Dataset directory
│       ├── mass_case_description_train_set.csv
│       ├── mass_case_description_test_set.csv
│       ├── calc_case_description_train_set.csv
│       ├── calc_case_description_test_set.csv
│       └── full mammogram images/
│           ├── Mass-Training_P_00001_LEFT_CC/
│           ├── Mass-Training_P_00001_LEFT_MLO/
│           └── ...
├── dcm_files.txt ............................... Optional: List of DICOM paths
└── outputs/ .................................... Created automatically
├── checkpoints/
├── plots/
├── visualizations/
└── training_log.txt

## Training Modes

### MIM Pre-training Only
```bash
python mammogram_analysis.py \
    --root_dir /path/to/CBIS-DDSM \
    --dicom_path /path/to/dcm_files.txt \
    --output_dir ./outputs \
    --mim_epochs 3 \
    --train_mim \
    --log_mode file
```
### Classification Training Only
```bash
python mammogram_analysis.py \
    --root_dir /path/to/CBIS-DDSM \
    --dicom_path /path/to/dcm_files.txt \
    --output_dir ./outputs \
    --cls_epochs 5 \
    --train_cls \
    --missing_label_percentage 50 \
    --log_mode console
```
### Visualization Only
```bash
python mammogram_analysis.py \
    --root_dir /path/to/CBIS-DDSM \
    --dicom_path /path/to/dcm_files.txt \
    --output_dir ./outputs \
    --visualize \
    --log_mode console
```
