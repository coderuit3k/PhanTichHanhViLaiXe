import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# Thu thấp hết tất cả dữ liệu về EDA, TEMP và HR trong thư mục WESAD được tải về
def run(file: str) -> list:
    l = []
    for i in range(2, 17):
        if i != 12:
            df = pd.read_csv(f"WESAD/S{i}/S{i}_E4_Data/" + file)
            df.columns = [file.split(".")[0]]
            l.append(df)
        else:
            continue
        
    return l

active_img_files = os.listdir("0 FaceImages/Active Subjects")

# Tạo thư mục lưu thông tin về TEMP, EDA, HR
if not os.path.exists("CSV"):
    os.makedirs("CSV", exist_ok=True)

# Tạo file csv lưu trữ thông tin về TEMP, EDA và HR
def create_csv(file: str):
    if file == "HR.csv":
        hr_list = run("HR.csv")
        hr = pd.concat(hr_list, axis=0)
        
        for i in range(0, len(active_img_files)):
            hr.iloc[i, 0] = random.uniform(60, 100)
            
        for i in range(0, len(active_img_files)):
            if i < len(active_img_files) // 2:
                hr.iloc[i + len(active_img_files), 0] = random.uniform(30, 60)
            else:
                hr.iloc[i + len(active_img_files), 0] = random.uniform(100, 140)

        hr = hr.iloc[0: len(active_img_files) * 2, 0]
        hr.to_csv("CSV/HR.csv", index=False)

    if file == "EDA.csv":
        eda_list = run("EDA.csv")
        eda = pd.concat(eda_list, axis=0)
        
        for i in range(0, len(active_img_files)):
            eda.iloc[i, 0] = random.uniform(20e-6, 0.5)
            
        for i in range(0, len(active_img_files)):
            if i < len(active_img_files) // 2:
                eda.iloc[i + len(active_img_files), 0] = random.uniform(30e-6, 20e-6)
            else:
                eda.iloc[i + len(active_img_files), 0] = random.uniform(0.5, 2)

        eda = eda.iloc[0: len(active_img_files) * 2, 0]
        eda.to_csv("CSV/EDA.csv", index=False)

    if file == "TEMP.csv":
        temp_list = run("TEMP.csv")
        temp = pd.concat(temp_list, axis=0)
        
        for i in range(0, len(active_img_files)):
            temp.iloc[i, 0] = random.uniform(36, 37.5)
            
        temp = temp.iloc[0: len(active_img_files) * 2, 0]
        temp.to_csv("CSV/TEMP.csv", index=False)
       
 
create_csv("HR.csv")
create_csv("EDA.csv")
create_csv("TEMP.csv")

def visualize(names: list):
    plt.figure(figsize=(14, 7))
    for i, name in enumerate(names):
        plt.subplot(1, 3, i + 1)
        df = pd.read_csv(f"CSV/{name}.csv")
        plt.hist(df[name].values)
        plt.title(f"Histogram {name}")
        plt.xlabel(name)
        plt.ylabel("Frequency")
    
    plt.tight_layout()      
    plt.show()
        
visualize(["HR", "EDA", "TEMP"])