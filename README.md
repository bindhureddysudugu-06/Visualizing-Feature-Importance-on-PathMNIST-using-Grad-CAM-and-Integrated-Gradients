# Visualizing-Feature-Importance-on-PathMNIST-using-Grad-CAM-and-Integrated-Gradients


## Project Overview
This project examines explainability in medical image classification on PathMNIST in MedMNIST v2. A baseline Convolutional Neural Network (CNN) is trained to label pathology image patches with 9 tissue classes. Two explainability approaches, namely, **Grad-CAM** and **Integrated Gradients**, are applied after training to understand which regions of the image have an impact on the prediction of the model. An experiment on **faithfulness-style masking** is also conducted to determine whether the highlighted pixels actually influence model confidence or not.

## Objectives
There are three objectives of the project:
1. Train and test a baseline CNN on PathMNIST.
2. Create visual explanations with Grad-CAM and Integrated Gradients.
3. Compare quality of explanation with a masking based faithfulness experiment.

## Dataset
- Dataset: PathMNIST (MedMNIST v2)
- Classes: 9 tissue classes
- Number of Training images: 89,996
- Number of Validation images: 10,004
- Numer of Test images: 7,180
- Image size: 28 × 28

## Final Results:
### Classification
 **Test Accuracy:** 74.62%


### Faithfulness Experiment
Average confidence drop under different masking ratios (5%, 10%, 20%):

| Masking Ratio | Grad-CAM | Integrated Gradients | Random Masking |
|--------------|----------|----------------------|----------------|
| 5%           | 0.2308   | 0.1639               | 0.1084         |
| 10%          | 0.7254   | 0.5899               | 0.6173         |
| 20%          | 0.7385   | 0.5401               | 0.7350         |

### Main Observation
The most informative faithfulness was obtained under the **5% masking** setting. At this level, Grad-CAM and Integrated Gradients performed better than random masking, and Grad-CAM demonstrated a highest faithfulness.

## Project Structure
|-- pathmnist_data\
|-- models\
│   -- best_pathmnist_cnn.pth\
│   |-- faithfulness\
│   |-- gradcam\
│   │   |-- correct\
│   │   |-- incorrect\
│   |-- integrated_gradients\
│   |   |-- correct\
│   |   |-- incorrect\
|-- download_pathmnist_dataset.py\
|-- cnn_train_pathmnist_dataset.py\
|-- evaluate_pathmnist_dataset.py\
|-- gradcam_pathmnist_dataset.py\
|-- integrated_gradients_pathmnist_dataset.py\
|-- plot_confusion_matrix_pathmnist_dataset.py\
|-- faithfulness_test_pathmnist_dataset_top5imp_pixels.py\
|-- faithfulness_test_pathmnist_dataset_top10imp_pixels.py\
|-- faithfulness_test_pathmnist_dataset_top5imp_pixels.py


