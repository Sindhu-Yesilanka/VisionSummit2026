# Vision Summit 2026 - Project: "Defect Classification using Light-weight Models"

## Project Overview
Our team has worked on designing a solution for defect classification using lightweight models suitable for constrained hardware environments such as NXP boards. We have trained several models, including **MobileNetV2**, **ShuffleNet**, **EfficientNet Lite**, and **TinyCNN**, using a dataset of grayscale images. The goal is to optimize the model size and accuracy for deployment on edge devices with memory constraints.

## Dataset Plan & Class Design
- **Total images planned/current**: 2500 (for each class).
- **Number of classes**: 8
  - **Defect Classes**: 
    - BLOCK_ETCH
    - PARTICLE
    - PO_CONTAMINATION
    - SCRATCH
    - bridge
    - line defects
    - clean
    - others
- **Class balance plan**: A minimum of 2500 images per class, balanced to avoid overfitting.
- **Train/Val/Test split**: 70% / 15% / 15% split for training, validation, and testing respectively.
- **Image type**: Grayscale (preferred, as per dataset design).
- **Labeling method/source**: Manual (the images are labeled based on defect categories).

## Model & Results (Phase 1)
### **Model Details**:
- **Architecture**: MobileNetV2 (for efficient performance with reduced memory usage).
- **Training approach**: Transfer learning, using pre-trained MobileNetV2 weights for fine-tuning.
- **Input size**: 160x160 pixels (resized for input).
- **Model size**: 8.53MB
- **Framework**: TensorFlow
- **Metrics on Test Split**:
  - **Accuracy**: 93.29%
  - **Precision/Recall**: Precision = 98%, Recall = 97%
  - **Confusion Matrix**: 
    - (Visualize using the code to generate the confusion matrix and include it here as an image)

### **Results**:
The model was trained on a dataset of 2500 images per class, and it achieved a validation accuracy of 93.29% after fine-tuning with MobileNetV2. The results were validated using confusion matrices to ensure the quality of defect classification.

## Model Comparison Table
| **Model**             | **Architecture**           | **Training Approach** | **Input Size** | **Model Size (MB)** | **Framework** | **Accuracy (Test Set)** | **Precision** | **Recall** |
| --------------------- | -------------------------- | --------------------- | -------------- | ------------------- | ------------- | ----------------------- | ------------- | ---------- | 
| **MobileNetV2**       | Depthwise Separable CNN    | Transfer Learning     | 160x160        | 8.5 MB             | TensorFlow    | 93.29%                  | 0.85          | 0.87       |
| **ShuffleNetV2**      | Lightweight CNN (ShiftNet) | Transfer Learning     | 160x160        | 5 MB               | TensorFlow    | 77%                     | 0.75          | 0.74       | 
| **EfficientNet Lite** | EfficientNet               | Transfer Learning     | 160x160        | 6 MB               | TensorFlow    | 26%                     | 0.45          | 0.50       |
| **TinyCNN**           | Lightweight CNN            | Scratch/Transfer      | 160x160        | 4 MB               | TensorFlow    | 82%                     | 0.80          | 0.78       |      |

## Artifacts & Links
- **GitHub Repository**: [https://github.com/YourUsername/VisionSummit2026](https://github.com/YourUsername/VisionSummit2026)
- **Dataset ZIP Link**: [[Google Drive Link](https://drive.google.com/your_dataset_link)  
  (Upload your dataset folder here and update the link)](https://drive.google.com/drive/folders/1m3LqErU98uaufvS0S0FjfNjzu3V_IS0l?usp=drive_link)
- **ONNX Model Link**: [[ONNX Model File](https://drive.google.com/file/d/13q0jtFKjd1kdLl3tRQri9Wma0Unyb_Eg/view?usp=drive_link)](https://drive.google.com/file/d/1xRxcgQwPnMPfGwnCAduotZxfhjTchp3O/view?usp=drive_link)  
  (Here is the link to download your ONNX model)
- **Saved Model Link**: [[(https://drive.google.com/file/d/13q0jtFKjd1kdLl3tRQri9Wma0Unyb_Eg/view?usp=drive_link)](https://drive.google.com/file/d/1xRxcgQwPnMPfGwnCAduotZxfhjTchp3O/view?usp=sharing)  ](https://drive.google.com/file/d/13q0jtFKjd1kdLl3tRQri9Wma0Unyb_Eg/view?usp=drive_link)
  (Here is the link to download your ONNX model)
  

## Research & References
1. MobileNetV2 - Efficient Mobile Models for Mobile Vision.
2. Dataset source: Custom dataset for defect classification.
3. [NXP eIQ Documentation](https://www.nxp.com/products/i.MX-RT1170)

---

## Instructions to Run the Project
1. Clone the repository using the following command:
   ```bash
   git clone https://github.com/YourUsername/VisionSummit2026.git
