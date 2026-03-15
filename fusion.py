import os
import cv2

CT_DIR = "dataset/CT"
MRI_DIR = "dataset/MRI"
OUT_DIR = "fused_dataset"
IMG_SIZE = (128,128)

classes = ["Healthy", "Tumor"]

os.makedirs(OUT_DIR, exist_ok=True)

for cls in classes:
    os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

    ct_imgs = sorted(os.listdir(os.path.join(CT_DIR, cls)))
    mri_imgs = sorted(os.listdir(os.path.join(MRI_DIR, cls)))

    for i in range(min(len(ct_imgs), len(mri_imgs))):
        ct = cv2.imread(os.path.join(CT_DIR, cls, ct_imgs[i]), 0)
        mri = cv2.imread(os.path.join(MRI_DIR, cls, mri_imgs[i]), 0)

        ct = cv2.resize(ct, IMG_SIZE)
        mri = cv2.resize(mri, IMG_SIZE)

        fused = cv2.addWeighted(ct, 0.5, mri, 0.5, 0)
        fused = cv2.cvtColor(fused, cv2.COLOR_GRAY2RGB)

        cv2.imwrite(f"{OUT_DIR}/{cls}/fused_{i}.jpg", fused)

print("✅ CT–MRI Fusion Completed")
