# Jailbreaking Deep Models: Adversarial Attacks on Image Classifiers

This project explores the vulnerability of deep neural networks to adversarial attacks. We implement various attack strategies against a ResNet-34 model pre-trained on ImageNet-1K and evaluate their effectiveness. Furthermore, we investigate the transferability of these attacks to a different model architecture (DenseNet-121).

## Project Overview

The core objective is to design and implement adversarial attacks that can degrade the performance of production-grade image classifiers while ensuring the perturbations are subtle or imperceptible. The project focuses on:

1.  **Baseline Evaluation:** Establishing the performance of a ResNet-34 model on a subset of the ImageNet-1K dataset.
2.  **Pixel-wise Attacks ($L_\infty$):** Implementing the Fast Gradient Sign Method (FGSM).
3.  **Improved Pixel-wise Attacks ($L_\infty$):** Implementing Projected Gradient Descent (PGD) for a stronger attack.
4.  **Patch Attacks ($L_0$-like):** Modifying only a small, random patch of the image using PGD with a potentially larger perturbation budget for the patch.
5.  **Transferability of Attacks:** Evaluating the generated adversarial examples on a different pre-trained model (DenseNet-121).

## Dataset

* **TestDataSet.zip:** A subset of images from 100 classes of the ImageNet-1K dataset. This archive contains 500 images, organized into subfolders named with their corresponding WordNet IDs (e.g., `n02123597`).
* **imagenet\_class\_index.json:** A JSON file mapping ImageNet class indices to WordNet IDs and human-readable class names. This is used to ensure correct label mapping during evaluation.

## Requirements

* Python 3.x
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Pillow (PIL)

You can install the necessary libraries using pip:
pip install torch torchvision numpy matplotlib pillow

Code Structure
The project is implemented in a single Jupyter Notebook or Python script, logically divided into five tasks:

Task 1: Basics
Functions for dataset extraction (extract_dataset), ImageNet class index download (download_class_index), and data loading/remapping (get_data_loader).
Evaluation function (evaluate) to calculate Top-1 and Top-5 accuracy.
Establishes baseline performance for ResNet-34.

Task 2: Pixel-wise attacks (FGSM)
Implementation of the FGSM attack (fgsm_attack) with $\epsilon = 0.02$.
Generation of "Adversarial Test Set 1".
Visualization of original vs. adversarial images and the noise.

Task 3: Improved attacks (PGD)
Implementation of the PGD attack (pgd_attack) with $\epsilon = 0.02$, $\alpha = 0.005$, and 10 iterations.
Generation of "Adversarial Test Set 2".
Verification of $L_\infty$ constraint.

Task 4: Patch attacks
Implementation of PGD targeted at a random 32x32 patch (pgd_patch_attack) with PATCH_SIZE = 32, EPSILON_PATCH = 0.3, ALPHA_PATCH = 0.05, and 20 iterations.
Generation of "Adversarial Test Set 3".

Task 5: Transferring attacks
Evaluation of the original dataset and all three adversarial datasets on a DenseNet-121 model.
Helper functions for image unnormalization and display (unnormalize_image_for_display, imshow_tensor_image) are also included.


Setup and Execution
Clone the repository:


Dataset:

Place the TestDataSet.zip file in the root directory of the project.
The imagenet_class_index.json file will be downloaded automatically by the script if not present.
Run the script/notebook:
Execute the Python script (e.g., main.py) or Jupyter Notebook (project.ipynb).

Results Summary

<img width="715" alt="image" src="https://github.com/user-attachments/assets/507a2814-c363-4f88-a384-d5a92c533e6f" />

Relative Top-1 Drop for Patch Attack is calculated as $(70.40 - 33.40) / 70.40$)

DenseNet-121 (Transfer Evaluation)

<img width="463" alt="image" src="https://github.com/user-attachments/assets/86e71e5e-4931-47ac-82c1-7f73de58581e" />

Key Findings from Transferability (Task 5)

Adversarial examples generated for ResNet-34 show a degree of transferability to DenseNet-121, as evidenced by the drop in accuracy on DenseNet-121 when evaluated on the adversarial datasets compared to its performance on the clean dataset.
The attacks are less effective on DenseNet-121 than on the source model (ResNet-34), which is a common characteristic of transferred attacks.
PGD attacks, being stronger on the source model, also transferred slightly more effectively than FGSM in terms of Top-1 accuracy degradation on DenseNet-121.
The patch attack also transferred but had a more limited impact on the target model's accuracy, which is expected due to the localized nature of the perturbation.
Visualizations
The script generates and displays visualizations for a few sample images for each attack type (FGSM, PGD, Patch). These visualizations typically show:

The original image with its true and predicted class.
The adversarial image with its (often misclassified) predicted class.
The amplified noise/perturbation that was applied (or the location of the patch attack).

---------- Results for Task 1 -----------

<img width="1272" alt="image" src="https://github.com/user-attachments/assets/fe4e4500-84ea-4816-a38d-cb6b627b8fbd" />

---------- Results for Task 2 -----------

![image](https://github.com/user-attachments/assets/98228c11-5a26-4716-bb33-5eb612e2146d)

---------- Results for Task 3 -----------

![image](https://github.com/user-attachments/assets/5804cb14-c1ca-44cd-930b-66eed0150ec9)

---------- Results for Task 4 -----------

![image](https://github.com/user-attachments/assets/3eb245d5-7aad-4c5d-849e-32b17d1e28ff)

---------- Results for Task 5 -----------

<img width="610" alt="image" src="https://github.com/user-attachments/assets/9edb14e3-e028-4b5e-a3f8-ba69783a8928" />




