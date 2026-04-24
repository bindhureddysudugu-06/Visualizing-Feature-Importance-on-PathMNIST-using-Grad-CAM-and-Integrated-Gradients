# Visualizing-Feature-Importance-on-PathMNIST-using-Grad-CAM-and-Integrated-Gradients


## Project Overview
This project studies explainability in medical image classification using the PathMNIST dataset from MedMNIST v2. A baseline Convolutional Neural Network (CNN) is trained to classify pathology image patches into 9 tissue classes. After training, two explainability methods, **Grad-CAM** and **Integrated Gradients**, are used to visualize which image regions influence the model’s prediction. A **faithfulness-style masking experiment** is also performed to test whether the highlighted pixels truly affect model confidence.

## Objectives
The project has three main goals:
1. Train and evaluate a baseline CNN on PathMNIST.
2. Generate visual explanations using Grad-CAM and Integrated Gradients.
3. Compare explanation quality using a masking-based faithfulness experiment.

## Dataset
- **Dataset:** PathMNIST (MedMNIST v2)
- **Classes:** 9 tissue classes
- **Training images:** 89,996
- **Validation images:** 10,004
- **Test images:** 7,180
- **Image size:** 28 × 28

## Final Results:
### Classification
- **Test Accuracy:** 74.62%


### Faithfulness Experiment
Average confidence drop under different masking ratios:

| Masking Ratio | Grad-CAM | Integrated Gradients | Random Masking |
|--------------|----------|----------------------|----------------|
| 5%           | 0.2308   | 0.1639               | 0.1084         |
| 10%          | 0.7254   | 0.5899               | 0.6173         |
| 20%          | 0.7385   | 0.5401               | 0.7350         |

### Main Observation
The **5% masking setting** gave the most informative faithfulness result. At this level, both Grad-CAM and Integrated Gradients outperformed random masking, and **Grad-CAM showed the strongest faithfulness**.

---

## Project Structure
|-- pathmnist_data\
|-- models\
│   -- best_pathmnist_cnn.pth
│   |-- faithfulness\
│   |-- gradcam\
│   │   |-- correct\
│   │   |-- incorrect\
│   |-- integrated_gradients\
│      |-- correct\
│      |-- incorrect\
|-- download_pathmnist_dataset.py
|-- cnn_train_pathmnist_dataset.py
|-- evaluate_pathmnist_dataset.py
|-- gradcam_pathmnist_dataset.py
|-- integrated_gradients_pathmnist_dataset.py
|-- faithfulness_test_pathmnist_dataset_top5imp_pixels.py
|-- faithfulness_test_pathmnist_dataset_top10imp_pixels.py
|-- faithfulness_test_pathmnist_dataset_top5imp_pixels.py
|-- plot_confusion_matrix_pathmnist_dataset.py
|-- plot_faithfulness_graph.py
