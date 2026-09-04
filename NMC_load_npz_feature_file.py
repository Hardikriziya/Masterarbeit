import numpy as np

path = r"D:\TU\Master_Thesis\Raw_Data\Battery_Data\Plots\NMC_Selected_Features_1000\SDU\SDU\SDU_Battery_4.npz"

data = np.load(path)

print("Keys:", data.files)


for key in data.files:
    arr = data[key]

    print(f"\n--- {key} ---")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print(arr.flatten()[:15])