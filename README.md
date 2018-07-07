# Handwritten Digit Recognition
This is a python implementation of fully connected Convolutional Neural Network for the recognition of handwritten digits. The neural network takes the 2D array of 28x28 pixels grayscale handwritten image as input and gives the result from 0 to 9 as output. It has 3 layers, Input layer with 784 neurons, Hidden layer with 128 neurons and an Output layer with 10 neurons. Script uses Gradient Descent with Back Propagation for the training of the network.

## Requirements
- Python 3.5+
- Pandas
    - `pip3 install pandas`
- Numpy
    - `pip3 install numpy`

## Usage
1. Clone or download and extract the repository on the local machine.

2. There is a `Dataset.csv` file containing the 42000 handwritten digits grayscale data with their respective labels. This dataset has been downloaded from [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data). The repositiory also contains the already trained weights and bias for the testing of network.

3. For training of the network, run `TrainingCNN.py` file.
    - Current code configuration will make 200 batches of the dataset with 2000 epoch and a learning rate of 0.01.
    - It will create 4 CSV files in the local directory containing the weights and bias of the network layers.
    > Note: It will take around 1 hour 40 minutes for the training. Make sure to properly cool your machine during the process to avoid any internal damages.

4. Run `DigitRecognizerCNN.py` file for the testing of the Neural Network.

## Accuracy
Current weights & bias produces the accuracy of 89.0% on 200 handwritten digits.

## License
Handwritten Digit Recognition is licensed under [GNU GPLv3 License](https://www.gnu.org/licenses/gpl-3.0.en.html)
