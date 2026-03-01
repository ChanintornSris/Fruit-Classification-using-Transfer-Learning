# Fruit Classification using Transfer Learning

## Introduction
This project demonstrates how AI can classify fruit images using transfer learning with a pre-trained VGG16 model. It fine-tunes the model on a custom dataset, leveraging features from ImageNet, to achieve accurate categorization with fewer data and resources.

## Aim
The goal of this project is to build a fruit image classifier using transfer learning. We fine-tune a pre-trained model on a custom dataset of fruit images to enable it to classify fruits effectively into their respective categories, and visualize the model's accuracy and predictions on sample test images.

## Features
- Dataset preparation and augmentation using `ImageDataGenerator`.
- Building a custom model on top of the pre-trained VGG16 base utilizing `GlobalAveragePooling2D`, `BatchNormalization`, and `Dropout`.
- Model optimization with `Adam` optimizer, `ReduceLROnPlateau`, and `EarlyStopping` callbacks.
- Fine-tuning specific layers in the VGG16 base model for enhanced accuracy.
- Visualizing training performance, including accuracy and loss curves.
- Evaluating classification predictions on unseen test samples.

## Prerequisites and Setup
To run this project, you need a basic understanding of Python and Keras, alongside the following installations:

- **TensorFlow** (v2.16.2)
- **Matplotlib** (v3.9.2)
- **NumPy** (v1.26.4)
- **SciPy** (v1.14.1)
- **Scikit-learn** (v1.5.2)

You can install the dependencies via pip:
```bash
pip install tensorflow==2.16.2 matplotlib==3.9.2 numpy==1.26.4 scipy==1.14.1 scikit-learn==1.5.2
```

## Dataset Structure
Organize the dataset in the following structure within the project directory. The default path configured in the code expects the **Fruits-360** dataset:

```
fruits-360-original-size/fruits-360-original-size/
├── Training/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── Validation/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── Test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Workflow Tasks
1. **Data Preparation**: Loading dataset directories, applying rescaling, and adding image augmentation for the training set.
2. **Model Architecture**: Incorporating the VGG16 base model (with frozen initial weights) ending with custom dense and dropout layers.
3. **Training**: Compiling with `categorical_crossentropy` and training the head of the model initially.
4. **Fine-tuning**: Unfreezing the top 5 layers of the VGG16 base model to specialize in fruit features, followed by retraining with a lower learning rate.
5. **Evaluation & Visualization**: Testing model performance on the hold-out test set (yielding ~89% accuracy) and presenting graphical predictions.

## Usage
Simply run the `Fruit Classification.ipynb` Jupyter Notebook step-by-step to preprocess the data, build and train the transfer learning model, and test its classification capabilities.
