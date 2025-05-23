## Plant Seedling Classification with CNNs
#Research Question
Can a neural network model accurately identify crop seedlings versus weeds based solely on images of young plants?

# Justification
This question is highly relevant to agriculture and environmental management. Early and accurate identification of plant species helps farmers, gardeners, and land managers differentiate crops from weeds, reducing herbicide use, lowering costs, and improving crop yields.

#Objectives and Goals
Goal: Develop a computer vision model that classifies plant seedlings into specific categories based on image data.

Objective: Create a user-friendly tool (for farmers, gardeners, agronomists) that accurately identifies plant species from images during early growth stages.

## Neural Network Architecture
Model Type
Convolutional Neural Network (CNN)

# Why CNN?
CNNs are ideal for image classification due to their ability to learn spatial hierarchies — detecting low-level features like edges and textures and building up to complex patterns such as shapes and structures. This makes them well-suited for distinguishing subtle differences among plant seedlings.

# Model Summary
PlantCNN(
  (conv1): Conv2d(3, 32, kernel_size=3, padding=1)
  (pool1): MaxPool2d(kernel_size=2)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (pool2): MaxPool2d(kernel_size=2)
  (conv3): Conv2d(64, 128, kernel_size=3, padding=1)
  (pool3): MaxPool2d(kernel_size=2)
  (fc1): Linear(32768, 256)
  (fc2): Linear(256, 12)
)
Layers:

3 Convolutional layers extract hierarchical image features

3 Max Pooling layers reduce dimensionality while retaining important info

2 Fully connected layers perform classification (256 nodes → 12 classes)

Parameters:

Total: 8,485,196

conv1: 896

conv2: 18,496

conv3: 73,856

fc1: 8,388,864

fc2: 3,084

## Data Preparation
# Data Augmentation (using imgaug)
Horizontal Flip (orientation robustness)

Scaling (size variability)

Rotation (arbitrary plant orientations)

Gaussian Blur (simulate focus issues)

Gaussian Noise (environmental noise tolerance)

Brightness Adjustment (varied lighting conditions)

# Normalization
Pixel values scaled from 0–255 to 0–1 by dividing by 255 to increase stability, speed convergence, and prevent bias.

Dataset Split
Training: 70%

Validation: 15%

Testing: 15%

Stratified sampling to preserve class distribution

# Label Encoding
Plant species labels converted to integer classes using scikit-learn’s LabelEncoder for model compatibility.

## Training Details
Loss function: CrossEntropyLoss for multi-class classification

Optimizer: Adam with learning rate 0.001 (adaptive, fast convergence)

Early stopping: Implemented to avoid overfitting based on validation accuracy

## Dataset Information
All datasets used are cleaned, augmented, and properly split. Label mappings, transformations, and encoded files are tracked via version control.

Important Note on Large Data File
One of the data files (the file with the seedling photos) is excluded from this repository due to size limitations (exceeds 100MB GitLab limit). It is:

Not tracked by Git


# Conclusion
This CNN-based plant seedling classifier demonstrates practical application of computer vision in agriculture. It provides a scalable, reliable tool to assist farmers and researchers in early identification of crops and weeds, supporting sustainable farming practices.

