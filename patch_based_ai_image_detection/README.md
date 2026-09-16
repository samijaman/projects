# Patch-Based Detection of AI-Generated Facial Images

A computer vision research project investigating whether patch-based convolutional neural networks can effectively distinguish AI-generated facial images from real photographs.

**Authors:** Mark Gislason, Samia Jaman

## Project Overview

As AI-generated facial images become increasingly realistic, detecting synthetic media requires models capable of identifying subtle visual artifacts that may not be obvious across an entire image.

This project investigates a patch-based approach to AI-image detection. Instead of classifying only complete facial images, images are divided into smaller patches and classified individually using convolutional neural networks (CNNs).

Patch-level predictions are then aggregated to produce an image-level classification.

The project examines three main questions:

- How does patch size affect classification accuracy?
- How does patch-based classification compare with full-image classification?
- Can patch predictions provide interpretable information about which regions of an image appear synthetic?

## Dataset

The project uses a balanced dataset of 140,000 facial images:

- 70,000 AI-generated faces created using NVIDIA StyleGAN
- 70,000 real facial images from NVIDIA's Flickr face dataset

Images were processed using the `face_recognition` Python library to isolate facial regions and reduce background-related noise.

422 problematic images were removed during preprocessing.

The remaining images were cropped to 128×128 pixels and divided into training, validation, and test sets using an 80/10/10 split.

Importantly, images were split **before** patch generation to prevent patches from the same image from appearing across multiple data splits.

## Patch-Based Classification

Four image resolutions were evaluated:

- 16×16 patches
- 32×32 patches
- 64×64 patches
- 128×128 full images

The patch-based models classify individual regions of an image independently.

For image-level classification, predictions from all patches belonging to an image are combined using majority voting.

An image is classified as AI-generated when at least half of its patches are predicted to be synthetic.

## Model Architecture

The primary classifier is a convolutional neural network implemented in PyTorch.

The CNN contains:

- Three convolutional layers with 32, 64, and 128 filters
- Batch normalization
- Max pooling
- ReLU activations
- Dropout
- Two fully connected layers
- Cross-entropy loss
- Adam optimization

A fully connected neural network was also evaluated as a baseline to examine the importance of spatial image features.

## Results

The 32×32 patch-based CNN produced the strongest image-level performance.

| Model | Patch/Test Accuracy | Aggregated Image Accuracy |
| --- | ---: | ---: |
| 128×128 CNN | 97.10% | — |
| 64×64 Patch CNN | 94.34% | 98.01% |
| 32×32 Patch CNN | 86.85% | **99.17%** |
| 16×16 Patch CNN | 73.11% | 91.78% |
| 32×32 Fully Connected NN | 53.79% | 54.42% |

Although individual 32×32 patches achieved only 86.85% classification accuracy, combining patch predictions through majority voting increased image-level accuracy to 99.17%.

This suggests that individual patches may contain noisy predictions while their combined predictions provide a much stronger signal.

## Training-Data Experiment

The project also tested whether patch-based models retain their advantage when less training data is available.

With 25% of the training data:

- 128×128 CNN: 91.95% test accuracy
- 32×32 Patch CNN: 90.35% aggregated accuracy

With 10% of the training data:

- 128×128 CNN: 90.37% test accuracy
- 32×32 Patch CNN: 86.02% aggregated accuracy

The full-image CNN retained more of its performance as the training dataset became smaller.

## Patch-Based Interpretability

One advantage of patch-based classification is the ability to visualize **where** the model detects evidence of synthetic imagery.

The project generates spatial heatmaps in which individual image patches are colored according to their predicted class and confidence.

These visualizations frequently highlight regions around:

- facial boundaries,
- hair,
- background transitions, and
- other localized image features.

This provides a more interpretable prediction than a single whole-image classification.

## Key Findings

The experiments demonstrate a tradeoff between local and global image information.

Smaller patches allow CNNs to focus on localized visual artifacts but remove global facial structure. Larger inputs retain information such as facial symmetry and relationships between features.

The 32×32 model provided the strongest image-level performance when trained on the complete dataset, while full-image models were more robust when training data was substantially reduced.

The poor performance of the fully connected baseline also demonstrates the importance of preserving spatial relationships when classifying images.

## Limitations

Several limitations are important when interpreting these results:

- Each reported result comes from a single training run rather than an average across multiple random seeds.
- Training and testing images come from the same underlying GAN distribution.
- Generalization to unseen image-generation architectures was not evaluated.
- Diffusion-generated images were not included.
- The dataset may contain generator-specific artifacts that make classification easier.

Future work could evaluate the models across multiple random seeds and test generalization on images produced by newer GAN and diffusion-based generators.

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- PIL / Pillow
- face_recognition
- Matplotlib
- Google Colab
- CUDA / GPU Training
- Jupyter Notebook

## Repository Contents

- `patch_based_ai_face_detection.ipynb` — preprocessing, patch generation, CNN training, model evaluation, aggregation, and visualization
- `paper.pdf` — full research paper describing the methodology, experiments, results, and limitations
