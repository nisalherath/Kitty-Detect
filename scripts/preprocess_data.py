import os
from PIL import Image
import torch
from torchvision import transforms

def preprocess_images(image_dir, output_dir):
    print("Preprocessing images with augmentation...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    os.makedirs(output_dir, exist_ok=True)

    for category in ['my_cat', 'other_cats']:
        category_path = os.path.join(image_dir, category)
        if not os.path.exists(category_path):
            print(f"Category {category} not found. Skipping.")
            continue

        processed_category_dir = os.path.join(output_dir, category)
        os.makedirs(processed_category_dir, exist_ok=True)

        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            if image_path.lower().endswith(('jpg', 'jpeg', 'png')):
                try:
                    image = Image.open(image_path).convert("RGB")
                    augmented_image = transform(image)
                    new_image_path = os.path.join(processed_category_dir, image_name)
                    augmented_image_pil = transforms.ToPILImage()(augmented_image)
                    augmented_image_pil.save(new_image_path)
                except Exception as e:
                    print(f"Error processing {image_name}: {e}")

    print("✅ Preprocessing complete!")

# Example usage
preprocess_images('data', 'data_processed')
