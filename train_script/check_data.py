
import pandas as pd
import os

df = pd.read_csv('../data/HAM10000_metadata.csv')
print("Label distribution in 'dx':")
print(df['dx'].value_counts())

# Mapping as per script:
label_map = {
    "mel": 1, "bcc": 1, "akiec": 1,
    "nv": 0, "bkl": 0, "df": 0, "vasc": 0
}
df['binary_label'] = df['dx'].map(label_map)
print("\nBinary label distribution:")
print(df['binary_label'].value_counts())

missing_images = 0
for img_id in df['image_id']:
    path1 = f"../data/HAM10000_images_part_1/{img_id}.jpg"
    path2 = f"../data/HAM10000_images_part_2/{img_id}.jpg"
    if not os.path.exists(path1) and not os.path.exists(path2):
        missing_images += 1

print(f"\nMissing images count: {missing_images}")
