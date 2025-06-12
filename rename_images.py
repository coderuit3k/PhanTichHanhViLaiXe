import os
from PIL import Image

# Hàm đếm số lượng format ảnh có trong folder
# format ảnh có thể dạng RGB, HSV, CMYK, ...

# def count(folder: str) -> set:
#     format = set()
#     for img_file in os.listdir(folder):
#         img_path = os.path.join(folder, img_file)
#         img = Image.open(img_path)
#         format.add(img.mode)
        
#     return format

# active_format = count("0 FaceImages/Active Subjects")
# fatigue_format = count("0 FaceImages/Fatigue Subjects")

# print(active_format) => {"RGB"}
# print(fatigue_format) => {"RGB"}

# Đổi tên file ảnh
def rename_file(folder: str, prefix: str):
    for i, file_name in enumerate(os.listdir(folder)):
        new_name = f"{prefix}_{i}.jpg"
        src = os.path.join(folder, file_name)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f"Renamed: {file_name} -> {new_name}")
        
rename_file("0 FaceImages/Active Subjects", prefix="Active")
rename_file("0 FaceImages/Fatigue Subjects", prefix="Fatigue")