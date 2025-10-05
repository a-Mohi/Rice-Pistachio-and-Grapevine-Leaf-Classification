# RPG Classification

This is a machine learning project focused on the classification of different types of rice, pistachios, and grapevine leaves.

## Technologies Used

The project is primarily developed using **Jupyter Notebook**. You can find the required libraries in the `requirements.txt` file.

## How to Run the Project

1.  Clone the repository to your local machine.
    ```
    git clone https://github.com/a-Mohi/Rice-Pistachio-and-Grapevine-Leaf-Classification.git
    ```
2.  Install the required dependencies using pip.
    ```
    pip install -r requirements.txt
    ```
3.  Open and run the `train.ipynb` or `kaggletrain.ipynb` notebooks in a Jupyter environment to train the models and perform classification.
---
## Model Architecture 

The core of the model is a **convolutional neural network (CNN)** that uses **transfer learning**. It leverages the pre-trained weights from the **ResNet50V2** model, which was originally trained on the ImageNet dataset. The architecture is structured as a `Sequential` model in Keras and includes the following layers:

  * **Data Augmentation**: A `Sequential` layer is used to perform real-time image augmentation, which helps improve the model's robustness and generalization.
  * **Pre-trained Base Model**: The **ResNet50V2** model is used without its top classification layer and its weights are initialized with those from ImageNet. The model is configured for a specific input image size of **384x384** pixels.
  * **Custom Classification Head**: A custom head is added to the pre-trained model for the specific classification task. This head consists of a **Dense layer** with 512 units, a **Dropout layer** to prevent overfitting, and a final **Dense layer** with 20 units (corresponding to 20 classes) and a **softmax** activation function.

## Training Process 

The model is trained using the following configurations:

  * **Optimizer**: The **AdamW** optimizer is used with a learning rate of 1e-4.
  * **Loss Function**: The loss is calculated using `sparse_categorical_crossentropy`.
  * **Metrics**: The model's performance is monitored using `accuracy`.
  * **Callbacks**: The project utilizes **Early Stopping** to halt training if validation loss does not improve and **ReduceLROnPlateau** to reduce the learning rate to improve convergence.
  * **Trainable Layers**: Most of the pre-trained ResNet50V2 layers are frozen, with only the last 20 layers being trainable, a common practice for fine-tuning.

## Data Processing 

Before being fed into the model, images are pre-processed:

  * **Image Resizing**: Images are resized to a fixed dimension of **384x384** pixels.
  * **Normalization**: Pixel values are scaled to a range between 0 and 1.
  * **Dataset Split**: The dataset is split into a **training set (80%)** and a **validation set (20%)**.
  * **Batching**: The images are processed in batches of 32.
