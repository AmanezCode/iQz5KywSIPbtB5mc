import timm, json, os, warnings, torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings('ignore')


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ImbalancedDatasetSampler(WeightedRandomSampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        targets = []
        for i in range(len(dataset)):
            sample = dataset[i]
            combined_label = sample['cleanliness'].item() * 2 + sample['damage'].item()
            targets.append(combined_label)
        
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        weights = [class_weights[t] for t in targets]
        
        if num_samples is None:
            num_samples = len(targets)
            
        super().__init__(weights, num_samples, replacement=True)


class CarConditionDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        
        valid_indices = []
        for idx, row in df.iterrows():
            image_path = os.path.join(image_dir, row['image_name'])
            if os.path.exists(image_path):
                valid_indices.append(idx)
        
        self.df = df.loc[valid_indices].reset_index(drop=True)
        print(f"Found {len(self.df)} images out of {len(df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            image_path = os.path.join(self.image_dir, row['image_name'])
            
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if self.transforms:
                image = self.transforms(image=image)['image']
            
            return {
                'image': image,
                'cleanliness': torch.tensor(row['cleanliness'], dtype=torch.long),
                'damage': torch.tensor(row['damage'], dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            black_image = torch.zeros((3, 224, 224))
            return {
                'image': black_image,
                'cleanliness': torch.tensor(0, dtype=torch.long),
                'damage': torch.tensor(0, dtype=torch.long)
            }


class EfficientNetCarClassifier(nn.Module):
    def __init__(self, num_cleanliness_classes=3, num_damage_classes=2, dropout_rate=0.3):
        super(EfficientNetCarClassifier, self).__init__()
        
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.cleanliness_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_cleanliness_classes)
        )
        
        self.damage_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_damage_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        
        cleanliness_out = self.cleanliness_head(features)
        damage_out = self.damage_head(features)
        
        return cleanliness_out, damage_out


def create_transforms(image_size=640):
    train_transforms = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.Rotate(limit=45, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transforms = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transforms, val_transforms


def load_annotations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for item in data:
        if 'annotations' in item and len(item['annotations']) > 0:
            annotation = item['annotations'][0]
            image_name = os.path.basename(item['data']['image'])
            
            result_data = {
                'image_name': image_name,
                'cleanliness': 0,
                'damage': 0,
                'has_boxes': False,
                'boxes': [],
                'cleanliness_regions': [],
                'damage_regions': []
            }
            
            clean_mapping = {'dusty': 1, 'dirty': 2}
            damage_mapping = {'scratches': 1, 'damaged': 1}
            
            cleanliness_labels = []
            damage_labels = []
            
            for result in annotation['result']:
                annotation_type = result['type']
                from_name = result.get('from_name', '')
                
                if 'cleanliness' in from_name:
                    labels = extract_labels_from_annotation(result, annotation_type)
                    if labels:
                        region_info = {
                            'type': annotation_type,
                            'labels': labels,
                            'geometry': result['value']
                        }
                        result_data['cleanliness_regions'].append(region_info)
                        result_data['has_boxes'] = True
                    
                    for label in labels:
                        if label in clean_mapping:
                            cleanliness_labels.append(clean_mapping[label])
                
                elif 'damage' in from_name:
                    labels = extract_labels_from_annotation(result, annotation_type)
                    if labels:
                        region_info = {
                            'type': annotation_type,
                            'labels': labels,
                            'geometry': result['value']
                        }
                        result_data['damage_regions'].append(region_info)
                        result_data['has_boxes'] = True
                    
                    for label in labels:
                        if label in damage_mapping:
                            damage_labels.append(damage_mapping[label])
            
            if cleanliness_labels:
                result_data['cleanliness'] = max(cleanliness_labels)
            
            if damage_labels:
                result_data['damage'] = max(damage_labels)
            
            processed_data.append(result_data)
    
    return pd.DataFrame(processed_data)


def extract_labels_from_annotation(result, annotation_type):
    label_key_mapping = {
        'rectanglelabels': 'rectanglelabels',
        'polygonlabels': 'polygonlabels',
        'brushlabels': 'brushlabels',
        'ellipselabels': 'ellipselabels'
    }
    
    label_key = label_key_mapping.get(annotation_type, [])
    return result['value'].get(label_key, [])


def analyze_dataset(df):
    print(f"Total images: {len(df)}")
    print(f"Images with annotations: {df['has_boxes'].sum()}")
    
    clean_counts = df['cleanliness'].value_counts().sort_index()
    clean_labels = {0: 'Clean', 1: 'Dusty', 2: 'Dirty'}
    print(f"CLEANLINESS distribution:")
    for idx, count in clean_counts.items():
        print(f"  {clean_labels.get(idx, f'Class {idx}')}: {count}")
    
    damage_counts = df['damage'].value_counts().sort_index()
    damage_labels = {0: 'Intact', 1: 'Damaged'}
    print(f"DAMAGE distribution:")
    for idx, count in damage_counts.items():
        print(f"  {damage_labels.get(idx, f'Class {idx}')}: {count}")
    
    print(f"Class combinations:")
    combinations = df.groupby(['cleanliness', 'damage']).size()
    for (clean, damage), count in combinations.items():
        clean_name = clean_labels.get(clean, f'Cleanliness {clean}')
        damage_name = damage_labels.get(damage, f'Damage {damage}')
        print(f"  {clean_name} + {damage_name}: {count}")


def handle_class_imbalance(df):
    analyze_dataset(df)
    
    clean_counts = df['cleanliness'].value_counts()
    damage_counts = df['damage'].value_counts()
    
    if len(clean_counts) == 1 and 0 in clean_counts.index:
        print(f"⚠️  All {clean_counts[0]} cars marked as 'clean'")
        print("Creating synthetic dirty examples...")
        
        n_dusty = max(20, len(df) // 10)
        n_dirty = max(10, len(df) // 20)
        
        all_indices = df.index.tolist()
        selected_indices = np.random.choice(all_indices, size=n_dusty + n_dirty, replace=False)
        
        df.loc[selected_indices[:n_dusty], 'cleanliness'] = 1
        df.loc[selected_indices[n_dusty:], 'cleanliness'] = 2
        
        print(f"Created {n_dusty} 'dusty' and {n_dirty} 'dirty' examples")
    
    if len(damage_counts) == 1 and 0 in damage_counts.index:
        print(f"⚠️  All {damage_counts[0]} cars marked as 'intact'")
        print("Creating synthetic damaged examples...")
        
        n_damaged = max(15, len(df) // 15)
        
        available_indices = df[df['damage'] == 0].index.tolist()
        if len(available_indices) >= n_damaged:
            damaged_indices = np.random.choice(available_indices, size=n_damaged, replace=False)
            df.loc[damaged_indices, 'damage'] = 1
            print(f"Created {n_damaged} damaged examples")
        else:
            print(f"Insufficient available indices for creating damaged examples")
    
    analyze_dataset(df)
    
    return df


def get_class_weights(df):
    clean_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['cleanliness']),
        y=df['cleanliness']
    )
    
    damage_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['damage']),
        y=df['damage']
    )
    
    print(f"Cleanliness class weights: {clean_weights}")
    print(f"Damage class weights: {damage_weights}")
    
    return clean_weights, damage_weights


def create_dataloaders(train_df, val_df, image_dir, transforms, batch_size):
    train_transforms, val_transforms = transforms
    
    train_dataset = CarConditionDataset(train_df, image_dir, train_transforms)
    val_dataset = CarConditionDataset(val_df, image_dir, val_transforms)
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty! Check image paths.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def setup_training_components(df, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on device: {device}")
    if device.type == 'cpu':
        print("⚠️  Using CPU - training will be slow!")
    
    comb_counts = df.groupby(['cleanliness', 'damage']).size()
    for (clean, damage), count in comb_counts.items():
        if count < 2:
            idxs = df[(df['cleanliness']==clean) & (df['damage']==damage)].index.tolist()
            while len(idxs) < 2:
                idxs.append(idxs[0])
            df = pd.concat([df, df.loc[idxs[len(idxs)-1:len(idxs)]]], ignore_index=True)

    clean_weights, damage_weights = get_class_weights(df)

    clean_weights_tensor = torch.tensor(clean_weights, dtype=torch.float).to(device)
    damage_weights_tensor = torch.tensor(damage_weights, dtype=torch.float).to(device)

    criterion_clean = nn.CrossEntropyLoss(weight=clean_weights_tensor, label_smoothing=0.1)
    criterion_damage = nn.CrossEntropyLoss(weight=damage_weights_tensor, label_smoothing=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    return device, criterion_clean, criterion_damage, optimizer, scheduler


def train_epoch(model, train_loader, criterion_clean, criterion_damage, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(train_bar):
        images = batch['image'].to(device)
        clean_labels = batch['cleanliness'].to(device)
        damage_labels = batch['damage'].to(device)
        
        optimizer.zero_grad()
        
        clean_out, damage_out = model(images)
        
        clean_loss = criterion_clean(clean_out, clean_labels)
        damage_loss = criterion_damage(damage_out, damage_labels)
        total_loss = clean_loss + damage_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        
        _, clean_pred = torch.max(clean_out.data, 1)
        _, damage_pred = torch.max(damage_out.data, 1)
        
        train_total += clean_labels.size(0) * 2
        train_correct += (clean_pred == clean_labels).sum().item()
        train_correct += (damage_pred == damage_labels).sum().item()
        
        if batch_idx % 5 == 0:
            current_acc = train_correct / max(train_total, 1)
            train_bar.set_postfix({
                'loss': f'{total_loss.item():.3f}',
                'acc': f'{current_acc:.3f}'
            })
    
    return train_loss / len(train_loader), train_correct / train_total


def validate_epoch(model, val_loader, criterion_clean, criterion_damage, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    clean_preds_all, clean_labels_all = [], []
    damage_preds_all, damage_labels_all = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            clean_labels = batch['cleanliness'].to(device)
            damage_labels = batch['damage'].to(device)

            clean_out, damage_out = model(images)
            _, clean_pred = torch.max(clean_out.data, 1)
            _, damage_pred = torch.max(damage_out.data, 1)

            clean_preds_all.extend(clean_pred.cpu().numpy())
            clean_labels_all.extend(clean_labels.cpu().numpy())
            damage_preds_all.extend(damage_pred.cpu().numpy())
            damage_labels_all.extend(damage_labels.cpu().numpy())
            
            clean_loss = criterion_clean(clean_out, clean_labels)
            damage_loss = criterion_damage(damage_out, damage_labels)
            total_loss = clean_loss + damage_loss
            
            val_loss += total_loss.item()
            val_total += clean_labels.size(0) * 2
            val_correct += (clean_pred == clean_labels).sum().item()
            val_correct += (damage_pred == damage_labels).sum().item()
    
    return (val_loss / len(val_loader), 
            val_correct / val_total, 
            clean_labels_all, 
            clean_preds_all, 
            damage_labels_all, 
            damage_preds_all)


def save_confusion_matrices(clean_labels_all, clean_preds_all, damage_labels_all, damage_preds_all, epoch):
    cm_clean = confusion_matrix(clean_labels_all, clean_preds_all)
    cm_damage = confusion_matrix(damage_labels_all, damage_preds_all)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_clean, annot=True, fmt='d', 
                xticklabels=['Clean', 'Dusty', 'Dirty'], 
                yticklabels=['Clean', 'Dusty', 'Dirty'])
    plt.title("Confusion Matrix - Cleanliness")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_damage, annot=True, fmt='d', 
                xticklabels=['Whole', 'Damaged'], 
                yticklabels=['Whole', 'Damaged'])
    plt.title("Confusion Matrix - Damage")
    plt.tight_layout()
    plt.savefig(f'matrixs/confusion_matrix_{epoch}.png', dpi=150)


def train_model(df, model, train_loader, val_loader, num_epochs=15):
    device, criterion_clean, criterion_damage, optimizer, scheduler = setup_training_components(df, model)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        epoch_train_loss, train_acc = train_epoch(
            model, train_loader, criterion_clean, criterion_damage, optimizer, device
        )
        
        epoch_val_loss, val_acc, clean_labels_all, clean_preds_all, damage_labels_all, damage_preds_all = validate_epoch(
            model, val_loader, criterion_clean, criterion_damage, device
        )
        
        clean_acc = accuracy_score(clean_labels_all, clean_preds_all)
        damage_acc = accuracy_score(damage_labels_all, damage_preds_all)
        clean_f1 = f1_score(clean_labels_all, clean_preds_all, average='weighted')
        damage_f1 = f1_score(damage_labels_all, damage_preds_all, average='weighted')
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Cleanliness - Acc: {clean_acc:.4f}, F1: {clean_f1:.4f}")
        print(f"Damage - Acc: {damage_acc:.4f}, F1: {damage_f1:.4f}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': val_acc,
                'history': history
            }, 'models/model.pth')
            print(f"💾 Saved best model (val_loss: {epoch_val_loss:.4f})")
        
        save_confusion_matrices(clean_labels_all, clean_preds_all, 
                               damage_labels_all, damage_preds_all, epoch)
        
        scheduler.step()

    return model, history


def plot_training_results(history):
    if history['train_loss']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/training_results.png', dpi=150)


def split_data(df, test_size=0.2, random_state=42):
    print("📊 Splitting data into train/validation...")
    try:
        df['combined_label'] = df['cleanliness'] * 2 + df['damage']
        train_df, val_df = train_test_split(df, test_size=test_size, 
                                          stratify=df['combined_label'], 
                                          random_state=random_state)
    except ValueError as e:
        print(f"⚠️  Stratification issue: {e}")
        print("Using regular split...")
        train_df, val_df = train_test_split(df, test_size=test_size, 
                                          random_state=random_state)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    print(f"Train distribution:")
    print(f"  Cleanliness: {dict(train_df['cleanliness'].value_counts().sort_index())}")
    print(f"  Damage: {dict(train_df['damage'].value_counts().sort_index())}")
    
    print(f"Val distribution:")
    print(f"  Cleanliness: {dict(val_df['cleanliness'].value_counts().sort_index())}")
    print(f"  Damage: {dict(val_df['damage'].value_counts().sort_index())}")
    
    return train_df, val_df


def validate_config(config):
    if not os.path.exists(config['data_path']):
        raise FileNotFoundError(f"File {config['data_path']} not found!")
    
    if not os.path.exists(config['image_dir']):
        raise FileNotFoundError(f"Image directory {config['image_dir']} not found!")


def main():
    CONFIG = {
        'data_path': 'data/annotations.json',
        'image_dir': 'data/images/',
        'image_size': 224,
        'batch_size': 16,
        'num_epochs': 15,
        'learning_rate': 2e-4,
    }
    
    try:
        validate_config(CONFIG)
        
        print("📁 Loading Label Studio data...")
        df = load_annotations(CONFIG['data_path'])
        print(f"✅ Loaded {len(df)} annotated images")
        
        df = handle_class_imbalance(df)
        
        train_df, val_df = split_data(df)
        
        transforms = create_transforms(CONFIG['image_size'])

        print("🔄 Creating datasets...")
        train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
            train_df, val_df, CONFIG['image_dir'], transforms, CONFIG['batch_size']
        )
        
        print("🤖 Creating EfficientNet-B4 model...")
        model = EfficientNetCarClassifier(num_cleanliness_classes=3, num_damage_classes=2)
        
        print("🚀 Starting training...")
        model, history = train_model(df, model, train_loader, val_loader, CONFIG['num_epochs'])
        
        print("🎉 Training completed!")
        print("💾 Model saved as 'models/model.pth'")
        
        plot_training_results(history)
        print("📈 Training results saved as 'results/training_results.png'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to create data/ folder with:")
        print("- annotations.json (Label Studio export)")
        print("- images/ (folder with images)")


if __name__ == "__main__":
    main()