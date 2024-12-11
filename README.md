# Dermato.AI-ML

# Skin Disease Classification Model

This project is a machine learning pipeline designed for classifying skin diseases from images. Below is an overview of the steps required to run the code and details about the model used.

## Requirements
Ensure you have the following libraries installed:
- `TensorFlow`
- `NumPy`
- `OpenCV`
- `Matplotlib`
- `Scikit-learn`

You can install these dependencies using the following command:
```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

## How to Run the Code

1. **Prepare the Dataset**
   - Ensure the dataset is available in a ZIP file containing subdirectories for each category.
   - Place the ZIP file in the `./Dataset` directory.

2. **Extract Dataset**
   - The script automatically extracts the dataset into separate training and testing directories.

3. **Run the Notebook**
   - Open the provided notebook in Jupyter Notebook or any other IDE that supports `.ipynb` files.
   - Execute the cells sequentially:
     - Import required libraries.
     - Extract and preprocess the dataset.
     - Perform data augmentation.
     - Train the machine learning model.

4. **Train the Model**
   - The notebook contains predefined models (e.g., `VGG16`, `ResNet50`) for training. Choose your preferred model by modifying the respective cell.
   - Train the model by running the training cells. The script will save the model automatically upon completion.

5. **Evaluate the Model**
   - Evaluate the model performance using the testing dataset. Metrics such as accuracy and loss will be displayed.

6. **Make Predictions**
   - Use the trained model to predict skin disease categories for new images by loading them into the prediction pipeline.

## Model Details
- The project uses pre-trained architectures, including:
  - `VGG16`
  - `ResNet50`
  - `DenseNet121`
  - `InceptionV3`
- These models are fine-tuned to classify skin disease categories such as:
  - Acne
  - Skin Cancer
  - Psoriasis
  - Eczema

### Data Augmentation
To balance the dataset and improve model generalization, the following augmentation techniques are applied:
- Rotation
- Horizontal and vertical flipping
- Zoom
- Shift

### Training and Testing
- **Training Dataset:** Augmented images from all categories.
- **Testing Dataset:** Augmented images from the original test split.

## Results and Metrics
Evaluation metrics such as accuracy and confusion matrices are generated to analyze the model's performance. Further tuning and hyperparameter optimization can be performed for better results.

## Note
Ensure proper hardware (preferably a GPU-enabled system) for faster model training.

For any issues or questions, feel free to raise an issue in the repository.
