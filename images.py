import os
import shutil
import re

# Set the base directory where the person_X folders are located
base_dir = 'dataset/images/silhouettes/dataset'
main_folder = os.path.join(base_dir, 'main_folder')
os.makedirs(main_folder, exist_ok=True)

# Helper to extract number from 'person_X'
def get_person_number(folder_name):
    match = re.match(r'person_(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')

# List and sort folders numerically
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('person_')]
folders.sort(key=get_person_number)

# Process each person folder
img_counter = 1
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    front_img = os.path.join(folder_path, 'front.png')
    side_img = os.path.join(folder_path, 'side.png')

    if os.path.exists(front_img) and os.path.exists(side_img):
        new_front_name = f'img{img_counter}_front.png'
        new_side_name = f'img{img_counter}_side.png'
        shutil.move(front_img, os.path.join(main_folder, new_front_name))
        shutil.move(side_img, os.path.join(main_folder, new_side_name))
        img_counter += 1
    else:
        print(f"⚠️ Skipping {folder} - missing front or side image.")

print("✅ All images moved and renamed correctly.")
