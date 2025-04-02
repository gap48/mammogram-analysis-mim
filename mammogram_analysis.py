import os
import sys
import glob
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, Subset

import monai
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    ToTensor,
    Compose,
    Resize,
    RandFlip,
    RandRotate,
    RandZoom,
    RandGaussianNoise
)
import timm

from tqdm import tqdm
from tqdm import tqdm  
from sklearn.metrics import classification_report
from typing import Optional, Tuple, Union


#########################################################
# 1) DATASET CLASSES
#########################################################

class MammoReconstructionDataset(Dataset):

    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        if self.transform:
            img = self.transform(path)
        else:
          
            img = LoadImage(image_only=True)(path)
            img = torch.tensor(img[None, ...])

    
        return img, 3


class MammoMetadataDataset(Dataset):

    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_path = row['image_path']

        label_dict = {
            'mass_calc': row['label'],  
            'pathology': row['pathology'],
            'subtlety': row['subtlety'],
            'breast_density': row['breast_density'],
            'assessment': row['assessment'],
            'abnormality_type': row['abnormality_type']
        }

        if self.transform:
            img = self.transform(image_path)
        else:
            img = LoadImage(image_only=True)(image_path)
            img = torch.tensor(img[None, ...])

        return img, label_dict


def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]

    collated_labels = {}
    if isinstance(labels[0], dict):
     
        for key in labels[0].keys():
            collated_labels[key] = [d[key] for d in labels]
        return images, collated_labels
    else:

        return images, labels


#########################################################
# 2) DATA LOADING FOR MIM
#########################################################

def create_mim_dataloaders(
    dcm_files,
    batch_size=4,
    num_workers=8,
    seed=42
):
    """
    Directly uses the user's snippet approach:
      - MammoReconstructionDataset with random split 80/10/10
      - High num_workers, pin_memory, etc.
    """
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate(range_x=15, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoise(prob=0.2),
        Resize((224, 224)),
        ToTensor()
    ])

    full_dataset = MammoReconstructionDataset(dcm_files, transform=transform)
    total_size = len(full_dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"[MIM] Train set: {len(train_dataset)} | Val set: {len(val_dataset)} | Test set: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )

    return train_loader, val_loader, test_loader


#########################################################
# 3) DATA LOADING FOR CLASSIFICATION
#########################################################

def create_partial_dataloaders(metadata_df, batch_size=4, num_workers=2, missing_label_percentage=100, seed=42):
    """
    Classification dataset:
      - uses metadata, partial-labeled training
      - split 80/10/10
    """
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate(range_x=15, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoise(prob=0.2),
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = MammoMetadataDataset(metadata_df, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )


    if missing_label_percentage < 100:
        num_samples = int(missing_label_percentage / 100.0 * train_size)
        indices = torch.randperm(train_size)[:num_samples]
        train_dataset = Subset(train_dataset, indices)
        print(f"[CLS] Using {missing_label_percentage}% labeled training => {len(train_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )

    return train_loader, val_loader, test_loader


#########################################################
# 4) UTILITY FUNCTIONS
#########################################################

def convert_pathology_to_label(pathology_list):
    numeric_labels = []
    for p in pathology_list:
        if p.upper().startswith('MAL'):
            numeric_labels.append(1)
        else:
            numeric_labels.append(0)
    return torch.tensor(numeric_labels, dtype=torch.long)

#########################################################
# 5) MIM MODEL & TRAINING
#########################################################

class MaskedDecoder(nn.Module):
    """
    Small decoder for MIM.
    """
    def __init__(self, in_dim=1024, out_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 512, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.bn2     = nn.BatchNorm2d(256)
        self.relu2   = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn3     = nn.BatchNorm2d(128)
        self.relu3   = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn4     = nn.BatchNorm2d(64)
        self.relu4   = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(64, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.cuda() 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.deconv3(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv_out(x)
        return x

def blockwise_random_mask(img, mask_ratio=0.5, block_size=16):
    B, C, H, W = img.shape
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size

    mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=img.device)
    for b in range(B):
        total_blocks = num_blocks_h * num_blocks_w
        num_masked_blocks = int(total_blocks * mask_ratio)
        perm = torch.randperm(total_blocks, device=img.device)
        masked_block_ids = perm[:num_masked_blocks]

        for block_id in masked_block_ids:
            row = block_id // num_blocks_w
            col = block_id % num_blocks_w
            row_start = row * block_size
            col_start = col * block_size
            mask[b, 0, row_start:row_start+block_size, col_start:col_start+block_size] = True

    masked_img = img.clone()
    masked_img[mask] = 0.0
    return masked_img, mask

@torch.no_grad()
def evaluate_mim(data_loader, encoder, decoder, criterion, mask_ratio=0.5, block_size=16):
    encoder.eval().cuda()
    decoder.eval().cuda()

    total_loss = 0.0
    total_samples = 0
    for imgs, _ in tqdm(data_loader, desc="Evaluating MIM", leave=False, file=sys.stdout, mininterval=0.1):
        imgs = imgs.cuda(non_blocking=True)
        B, _, H, W = imgs.shape
        if H != 224 or W != 224:
            imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear')

        masked_imgs, _ = blockwise_random_mask(imgs, mask_ratio, block_size)
        feats = encoder.forward_features(masked_imgs.cuda())
        feats = feats.permute(0, 3, 1, 2).cuda()

        reconstructed = decoder(feats)
        reconstructed_upsampled = F.interpolate(reconstructed, size=(224, 224), mode='bilinear')

        loss = criterion(reconstructed_upsampled.cuda(), imgs)
        total_loss += loss.item() * B
        total_samples += B

    return total_loss / total_samples if total_samples > 0 else 0

def train_mim(
    encoder, decoder,
    train_loader, val_loader, test_loader,
    lr=1e-4, weight_decay=1e-5,
    num_epochs=5,
    output_dir='.',
    mask_ratio=0.5,
    block_size=16,
    log_mode='console'
):
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    best_test_loss = float('inf')
    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(num_epochs):
        encoder.train().cuda()
        decoder.train().cuda()

        train_loss, train_samples = 0.0, 0
        batch_pbar = tqdm(train_loader, desc=f"MIM Epoch {epoch+1}/{num_epochs}", leave=False,file=sys.stdout, mininterval=0.1)
        for imgs, _ in batch_pbar:
            imgs = imgs.cuda(non_blocking=True)
            B, _, H, W = imgs.shape
            if H != 224 or W != 224:
                imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear')

            masked_imgs, _ = blockwise_random_mask(imgs, mask_ratio, block_size)
            feats = encoder.forward_features(masked_imgs.cuda())
            feats = feats.permute(0, 3, 1, 2).cuda()

            reconstructed = decoder(feats)
            reconstructed_upsampled = F.interpolate(reconstructed, size=(224, 224), mode='bilinear')

            loss = criterion(reconstructed_upsampled.cuda(), imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_loss += loss.item() * B
            train_samples += B

        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        val_loss  = evaluate_mim(val_loader, encoder, decoder, criterion, mask_ratio, block_size)
        test_loss = evaluate_mim(test_loader, encoder, decoder, criterion, mask_ratio, block_size)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}")

        
        last_ckpt_path = os.path.join(output_dir, "last_checkpoint.pth")
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss
        }, last_ckpt_path)

        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_ckpt_path = os.path.join(output_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': best_test_loss
            }, best_ckpt_path)
            print(f"  [*] New best test loss: {best_test_loss:.4f} => {best_ckpt_path}")

  
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("MIM Reconstruction Loss")
    plt.legend()

    plot_path = os.path.join(output_dir, "mim_loss_curve.png")
    plt.savefig(plot_path)
    print(f"[INFO] MIM Loss curve saved at: {plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()


#########################################################
# 6) CLASSIFICATION MODEL & TRAINING
#########################################################

class SwinMIMClassifier(nn.Module):
    def __init__(self, encoder, encoder_dim=1024, num_classes=2):
        super().__init__()
        self.encoder = encoder.cuda()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(encoder_dim, num_classes).cuda()

    def forward(self, x):
        x = x.cuda()
        feats = self.encoder.forward_features(x).cuda()  
        feats = feats.permute(0, 3, 1, 2).cuda()
        pooled = self.pool(feats).view(feats.size(0), -1).cuda()
        logits = self.classifier(pooled).cuda()
        return logits

def compute_class_weights(metadata_df):
    pathology_counts = metadata_df['pathology'].value_counts()
    benign_count = pathology_counts.get('BENIGN', 0) + pathology_counts.get('BENIGN_WITHOUT_CALLBACK', 0)
    malignant_count = pathology_counts.get('MALIGNANT', 0)

    benign_count = max(benign_count, 1)
    malignant_count = max(malignant_count, 1)
    total_samples = benign_count + malignant_count

    weight_benign    = total_samples / (2.0 * benign_count)
    weight_malignant = total_samples / (2.0 * malignant_count)
    return torch.tensor([weight_benign, weight_malignant], dtype=torch.float)

def train_classifier(
    model, 
    train_loader_cls, 
    val_loader_cls,
    test_loader_cls,
    metadata_df,
    lr=1e-4, 
    weight_decay=1e-5, 
    num_epochs=5, 
    output_dir='.',
    log_mode='console'
):
    model = model.cuda()
    class_weights = compute_class_weights(metadata_df).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(num_epochs):
    
        model.train().cuda()
        train_loss, train_correct, train_total = 0.0, 0, 0
        batch_pbar = tqdm(train_loader_cls, desc=f"Cls Epoch {epoch+1}/{num_epochs}", leave=False, file=sys.stdout, mininterval=0.1)
        for imgs, label_dict in batch_pbar:
            imgs = imgs.cuda(non_blocking=True)
            labels = convert_pathology_to_label(label_dict['pathology']).cuda()

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        avg_train_loss = train_loss / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0

     
        model.eval().cuda()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, label_dict in val_loader_cls:
                imgs = imgs.cuda(non_blocking=True)
                labels = convert_pathology_to_label(label_dict['pathology']).cuda()

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\n[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} || "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("Validation Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Benign','Malignant']))

   
        last_ckpt_path = os.path.join(output_dir, "last_checkpoint_swin_mim_classifier.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, last_ckpt_path)

    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(output_dir, "best_checkpoint_swin_mim_classifier.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, best_ckpt_path)
            print(f"  [*] New best Val Acc: {best_val_acc:.4f} => {best_ckpt_path}")


    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    model.eval().cuda()
    with torch.no_grad():
        for imgs, label_dict in tqdm(test_loader_cls, desc="[Test]"):
            imgs = imgs.cuda(non_blocking=True)
            labels = convert_pathology_to_label(label_dict['pathology']).cuda()

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            test_correct += (preds == labels).sum().item()
            test_total += imgs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total if test_total > 0 else 0
    print("\n[Final Test Results]")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Detailed Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign','Malignant']))

 
    epochs_range = range(1, num_epochs+1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classification Loss")
    plt.legend()
    loss_plot_path = os.path.join(output_dir, "classification_loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"[INFO] Classification Loss curve saved at: {loss_plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()

   
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy")
    plt.legend()
    acc_plot_path = os.path.join(output_dir, "classification_accuracy_curve.png")
    plt.savefig(acc_plot_path)
    print(f"[INFO] Classification Accuracy curve saved at: {acc_plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()


#########################################################
# 7) VISUALIZATION: ATTENTION & GRAD-CAM
#########################################################

def get_attention_block(model):
    if hasattr(model, 'encoder'):  
        return model.encoder.layers[-1].blocks[-1]
    elif hasattr(model, 'layers'):  
        return model.layers[-1].blocks[-1]
    return None

def process_attention_weights(attn_output, img_shape):
    attn = attn_output.detach().cpu()
    feature_magnitudes = torch.norm(attn[0], dim=-1) 
    attention_map = (feature_magnitudes - feature_magnitudes.min()) / \
                    (feature_magnitudes.max() - feature_magnitudes.min() + 1e-8)
    attention_map = attention_map.numpy()
    attention_map = F.interpolate(
        torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float(),
        size=img_shape[-2:],
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()

    return attention_map

def get_random_sample(dataloader):
    dataset = dataloader.dataset
    random_idx = random.randint(0, len(dataset) - 1)
    image, labels = dataset[random_idx]
    if isinstance(labels, dict):
        labels = {k: [v] for k, v in labels.items()}
    else:
        labels = {'pathology': [labels]}

    return image.unsqueeze(0).cuda(), labels

def visualize_attention(model, dataloader,
    output_dir='.',
    log_mode='console'):
    model = model.cuda()

    img, labels = get_random_sample(dataloader)
    true_label = labels['pathology'][0]
    attn_block = get_attention_block(model)
    if attn_block is None:
        print("No attention block found - skipping attention visualization")
        return
    attention_values = []

    def hook_fn(module, input, output):
        attention_values.append(output)
    handle = attn_block.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        outputs = model(img)
        pred_class = outputs.argmax(dim=1).item()
    handle.remove()

    if not attention_values:
        print("No attention values captured")
        return
    attention_map = process_attention_weights(attention_values[0], img.shape)
    if attention_map is None:
        print("Could not process attention weights")
        return
    fig = plt.figure(figsize=(22, 7))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    img_np = img.squeeze().cpu().numpy()
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title(f"Original Image\nLabel: {true_label}")
    ax1.axis('off')
    attention_display = ax2.imshow(attention_map, cmap='inferno')
    ax2.set_title("Raw Attention Map")
    ax2.axis('off')
    plt.colorbar(attention_display, ax=ax2)
    ax3.imshow(img_np, cmap='gray')
    ax3.imshow(attention_map, cmap='inferno', alpha=0.5)
    ax3.set_title(f"Attention Overlay\nPredicted: {'Malignant' if pred_class else 'Benign'}")
    ax3.axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "attention_map.png")
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.2)
    print(f"[INFO] Attention map saved at: {plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()


################### GradCAM ###################

def get_target_layer(model):
    target_layer = None
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        last_stage = model.encoder.layers[-2] 
        if hasattr(last_stage, 'blocks'):
            target_layer = last_stage.blocks[-1].attn
        return target_layer
    if hasattr(model, 'blocks'):
        target_layer = model.blocks[-2].attn  
        return target_layer
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)

    if conv_layers:
        target_layer = conv_layers[-2] if len(conv_layers) > 1 else conv_layers[-1]

    if target_layer is None:
        raise ValueError("Could not find suitable target layer")

    return target_layer

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.feature_maps = None
        self.gradient = None

        self.target_layer = target_layer if target_layer is not None else get_target_layer(model)
        if self.target_layer is None:
            raise ValueError("Could not find suitable target layer for GRAD-CAM")

        self.hooks = []
        self._register_hooks()

        self.is_transformer = self._check_if_transformer()

    def _check_if_transformer(self) -> bool:
        return any(
            hasattr(self.model, attr)
            for attr in ['blocks', 'encoder', 'transformer', 'attention']
        )

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.feature_maps = output[0]  
            else:
                self.feature_maps = output

        def full_backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(full_backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def _reshape_transform(self, tensor: torch.Tensor) -> torch.Tensor:

        if not self.is_transformer:
            return tensor

        if len(tensor.shape) == 4:  
            return tensor

        result = tensor
        if len(tensor.shape) == 3:
            num_patches = tensor.shape[1]
            h = w = int(np.sqrt(num_patches))
            result = tensor.reshape(tensor.shape[0], h, w, -1)
            result = result.permute(0, 3, 1, 2)

        return result

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        reshape_transform: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Generate GRAD-CAM with improved gradient handling and normalization
        """
        self.model.eval()
        input_tensor.requires_grad = True


        model_output = self.model(input_tensor)
        if isinstance(model_output, tuple):
            model_output = model_output[0]

        predicted_class = model_output.argmax(dim=1).item()
        target_class = predicted_class if target_class is None else target_class

        self.model.zero_grad()

        target_score = model_output[0][target_class]
        target_score.backward(retain_graph=True)

        gradients = self.gradient.detach()  
        attention_maps = self.feature_maps.detach()  

        print(f"Gradients shape: {gradients.shape}")
        print(f"Attention maps shape: {attention_maps.shape}")
        print(f"Gradient stats - mean: {gradients.mean().item()}, std: {gradients.std().item()}")
        print(f"Gradient range: {gradients.min().item()} to {gradients.max().item()}")

        gradients = 2 * (gradients - gradients.min()) / (gradients.max() - gradients.min()) - 1

        B, N, D = attention_maps.shape

        attention_scores = torch.matmul(attention_maps, attention_maps.transpose(-2, -1))  # [B, N, N]
        attention_scores = F.softmax(attention_scores / torch.sqrt(torch.tensor(D).float()), dim=-1)
        attention_scores = attention_scores.mean(dim=0)  
        grad_weights = torch.norm(gradients, dim=2)  
        grad_weights = F.softmax(grad_weights.mean(dim=0), dim=0)  

        cam = attention_scores * grad_weights.view(-1, 1)  
        cam = cam.mean(dim=0)  # [N]
        size = int(np.sqrt(N))
        cam = cam.view(size, size)

        cam = F.relu(cam)  
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

            cam = torch.pow(cam, 0.5)

            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        return cam.detach().cpu().numpy(), predicted_class

def process_gradcam(cam: np.ndarray, img_shape: Tuple[int, ...]) -> np.ndarray:
    cam_resized = F.interpolate(
        torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float(),
        size=img_shape[-2:],
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()

    return cam_resized

def visualize_gradcam(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_dir='.',
    log_mode='console',
    target_layer: Optional[torch.nn.Module] = None
):

    model = model.cuda()
    model.eval()

    img, labels = get_random_sample(dataloader)
    img_for_viz = img.clone().detach()
    true_label = labels['pathology'][0]
    grad_cam = GradCAM(model, target_layer)
    cam, pred_class = grad_cam.generate_cam(img)
    cam_processed = process_gradcam(cam, img.shape)

    fig = plt.figure(figsize=(22, 7))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    img_np = img_for_viz.squeeze().cpu().numpy()
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title(f"Original Image\nLabel: {true_label}")
    ax1.axis('off')
    gradcam_display = ax2.imshow(cam_processed, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_title("GRAD-CAM Heatmap")
    ax2.axis('off')
    plt.colorbar(gradcam_display, ax=ax2)
    ax3.imshow(img_np, cmap='gray')
    overlay = ax3.imshow(cam_processed, cmap='RdYlBu_r', alpha=0.7, vmin=0, vmax=1)
    ax3.set_title(f"GRAD-CAM Overlay\nPredicted: {'Malignant' if pred_class else 'Benign'}")
    ax3.axis('off')
    plt.colorbar(overlay, ax=ax3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir,  "gradcam_map.png")
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.2)
    print(f"[INFO] Grad-CAM map saved at: {plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()
    grad_cam.remove_hooks()


#########################################################
# 8) PARSE METADATA FOR CLASSIFICATION
#########################################################

def find_full_mammogram(dcm_files, patient_folder):
    matching_files = []
    for dcm_path in dcm_files:
        if patient_folder in dcm_path and "full mammogram images" in dcm_path:
            matching_files.append(dcm_path)
    return matching_files[0] if matching_files else None

def process_dataframe(df, dcm_files, is_mass=True):
    prefix = "Mass-Training" if is_mass else "Calc-Training"
    density_col = 'breast_density' if is_mass else 'breast density'
    data = []

    for _, row in df.iterrows():
        patient_id = row['patient_id'].split('_')[1]
        breast = row['left or right breast']
        view = row['image view']

        folder_pattern = f"{prefix}_P_{patient_id}_{breast}_{view}"
        full_image_path = find_full_mammogram(dcm_files, folder_pattern)
        if full_image_path:
            item = {
                'image_path': full_image_path,
                'label': 1 if is_mass else 0,
                'pathology': row['pathology'],
                'subtlety': row['subtlety'],
                'breast_density': row[density_col],
                'assessment': row['assessment'],
                'abnormality_type': row['abnormality type'],
                'patient_id': row['patient_id'],
                'breast': breast,
                'view': view
            }
            if is_mass:
                item['mass_shape'] = row['mass shape']
                item['mass_margins'] = row['mass margins']
            else:
                item['calc_type'] = row['calc type']
                item['calc_distribution'] = row['calc distribution']
            data.append(item)
    return data

def parse_ddsm_metadata(root_dir, dcm_files):
    mass_train_csv = os.path.join(root_dir, 'mass_case_description_train_set.csv')
    mass_test_csv  = os.path.join(root_dir, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(root_dir, 'calc_case_description_train_set.csv')
    calc_test_csv  = os.path.join(root_dir, 'calc_case_description_test_set.csv')

    mass_train = pd.read_csv(mass_train_csv)
    mass_test  = pd.read_csv(mass_test_csv)
    calc_train = pd.read_csv(calc_train_csv)
    calc_test  = pd.read_csv(calc_test_csv)

    all_data = []
    all_data.extend(process_dataframe(mass_train, dcm_files, is_mass=True))
    all_data.extend(process_dataframe(mass_test,  dcm_files, is_mass=True))
    all_data.extend(process_dataframe(calc_train, dcm_files, is_mass=False))
    all_data.extend(process_dataframe(calc_test,  dcm_files, is_mass=False))

    metadata_df = pd.DataFrame(all_data)
    print(f"[INFO] After matching CSV and DICOM, found {len(metadata_df)} records.")
    return metadata_df


#########################################################
# 9) MAIN
#########################################################

def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM MIM & Classification + Visualization")

    parser.add_argument('--root_dir', type=str, required=True,
                        help="Path to the directory containing CBIS-DDSM CSV files. ")
    parser.add_argument('--dicom_path', type=str, default="",
                        help="Path to a text file containing DICOM paths. If omitted, scan root_dir for *.dcm")

    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help="Directory to save checkpoints/results")

    parser.add_argument('--mim_epochs', type=int, default=3,
                        help="Number of epochs for MIM pre-training")
    parser.add_argument('--cls_epochs', type=int, default=3,
                        help="Number of epochs for classification training")

    parser.add_argument('--missing_label_percentage', type=int, default=100,
                        help="(For classification only) Percentage of labeled training data to use")

    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for dataloaders")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of workers for dataloaders")

    parser.add_argument('--train_mim', action='store_true',
                        help="Train MIM model from scratch")
    parser.add_argument('--train_cls', action='store_true',
                        help="Train classification model using MIM encoder")
    parser.add_argument('--visualize', action='store_true',
                        help="Perform both Attention-based and Grad-CAM visualization on a sample")

    parser.add_argument('--log_mode', type=str, choices=['console', 'file'], default='console',
                        help="If 'file', redirect prints to 'training_log.txt' and save plots to --output_dir. "
                             "If 'console', print to console and show plots interactively.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

  
    if args.log_mode == 'file':
        log_path = os.path.join(args.output_dir, "training_log.txt")
        sys.stdout = open(log_path, 'w', buffering=1)
        matplotlib.use('Agg')
        print(f"[INFO] Logging to file: {log_path}")
    else:
     
        pass

  
    if args.dicom_path:
        if not os.path.isfile(args.dicom_path):
            print(f"[ERROR] dicom_path file '{args.dicom_path}' not found.")
            sys.exit(1)
        with open(args.dicom_path, "r") as f:
            dcm_files = [line.strip() for line in f]
        print(f"[INFO] Loaded {len(dcm_files)} DICOM file paths from '{args.dicom_path}'.")
    else:
    
        dcm_files = glob.glob(os.path.join(args.root_dir, "**", "*.dcm"), recursive=True)
        print(f"[INFO] No --dicom_path given. Found {len(dcm_files)} DICOM files by scanning '{args.root_dir}'.")

 
    mim_train_loader, mim_val_loader, mim_test_loader = create_mim_dataloaders(
        dcm_files=dcm_files,
        batch_size=args.batch_size,
        num_workers=8,  
        seed=42
    )

 
    metadata_df = parse_ddsm_metadata(args.root_dir, dcm_files)
    if len(metadata_df) == 0:
        print("[WARNING] No metadata found or no matching entries. Classification might fail if --train_cls is set.")

    cls_train_loader, cls_val_loader, cls_test_loader = create_partial_dataloaders(
        metadata_df=metadata_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        missing_label_percentage=args.missing_label_percentage
    )


    encoder = timm.create_model(
        "swin_base_patch4_window7_224",
        in_chans=1,
        pretrained=True,
        num_classes=0
    ).cuda()


    if args.train_mim:
        print("\n[INFO] Starting MIM Training ...")
        decoder = MaskedDecoder(in_dim=1024, out_ch=1).cuda()
        train_mim(
            encoder=encoder,
            decoder=decoder,
            train_loader=mim_train_loader,
            val_loader=mim_val_loader,
            test_loader=mim_test_loader,
            num_epochs=args.mim_epochs,
            output_dir=args.output_dir,
            log_mode=args.log_mode
        )
    else:
        best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pth")
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path)
            encoder.load_state_dict(ckpt['encoder_state_dict'])
            print(f"[INFO] Loaded MIM checkpoint from {best_ckpt_path}")
        else:
            print("[WARNING] No MIM checkpoint found; using encoder as-is (ImageNet-pretrained).")

  
    if args.train_cls:
        print("\n[INFO] Starting Classification Training ...")
        model_cls = SwinMIMClassifier(encoder=encoder, encoder_dim=1024, num_classes=2).cuda()
        train_classifier(
            model=model_cls,
            train_loader_cls=cls_train_loader,
            val_loader_cls=cls_val_loader,
            test_loader_cls=cls_test_loader,
            metadata_df=metadata_df,
            num_epochs=args.cls_epochs,
            output_dir=args.output_dir,
            log_mode=args.log_mode
        )
    else:
        best_cls_ckpt = os.path.join(args.output_dir, "best_checkpoint_swin_mim_classifier.pth")
        if os.path.exists(best_cls_ckpt):
            print(f"[INFO] Found classification checkpoint: {best_cls_ckpt}")
        else:
            print("[INFO] No classification checkpoint found. Classification not performed.")

    
    if args.visualize:
        print("\n[INFO] Visualization (Attention & Grad-CAM) ...")
        model_cls = SwinMIMClassifier(encoder=encoder, encoder_dim=1024, num_classes=2).cuda()

        best_cls_ckpt = os.path.join(args.output_dir, "best_checkpoint_swin_mim_classifier.pth")
        if os.path.exists(best_cls_ckpt):
            ckpt = torch.load(best_cls_ckpt)
            model_cls.load_state_dict(ckpt['model_state_dict'])
            print(f"[INFO] Loaded classification checkpoint from {best_cls_ckpt}")
        else:
            print("[WARNING] No classification checkpoint found. Visualizing an untrained classifier model.")

   
        visualize_attention(model_cls, cls_test_loader,output_dir=args.output_dir,log_mode=args.log_mode)
    
        visualize_gradcam(model_cls, cls_test_loader,output_dir=args.output_dir,log_mode=args.log_mode)

    if args.log_mode == 'file':
        print("[INFO] All logs have been saved to file.")
        sys.stdout.close()


if __name__ == "__main__":
    main()
