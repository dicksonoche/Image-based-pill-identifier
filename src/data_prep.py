import os
import shutil
from sklearn.model_selection import train_test_split

# Assume downloaded images in 'raw_data/' with subfolders per class
raw_dir = 'raw_data'
base_dir = 'data'
classes = ['aspirin', 'ibuprofen', 'acetaminophen', 'atorvastatin', 'metformin']  # MVP classes

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

for cls in classes:
    cls_images = [f for f in os.listdir(os.path.join(raw_dir, cls)) if f.endswith('.jpg')]
    train_imgs, temp_imgs = train_test_split(cls_images, test_size=0.4, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    for split, imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(raw_dir, cls, img), os.path.join(base_dir, split, cls, img))