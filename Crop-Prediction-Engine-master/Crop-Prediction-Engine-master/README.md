
The User-Interface Repository for the Whole System is at [THIS REPO](https://github.com/VamshiKrishna-jillela/Crop-Prediction-UI) is live at [WEBSITE](https://1mc.netlify.app/)

This README provides information about the dataset and the Long Short-Term Memory (LSTM) model associated with it and Steps to Deploy a Trained Model and Steps to Use an Deployed Model Through API requests

## Trained Model Deployment Process

To deploy this RNN LSTM models on your Server/Machine and access its predictions through an HTTP API request, follow the instructions below.

## Setting up the Environment

1. Create a virtual environment to isolate the dependencies. Run the following command to create a virtual environment named "env":

```
python -m venv env
```

2. Activate the virtual environment:

- For Windows:
```
.\env\Scripts\activate
```

- For macOS and Linux:
```
source env/bin/activate
```

3. Install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

## Starting the Web Server

1. In the terminal, navigate to the directory where the "Server.py" file is located.

2. Start the web server by running the following command:

```
python Server.py
```

The server will start running, and you will see a message indicating the server has started successfully.

## Making API Requests

To get predicted data from the deployed LSTM model through an HTTP API request, you need to follow the API request documentation provided. The documentation will outline the required request format, endpoint, and parameters.

You can use tools like cURL, Postman, or any programming language's HTTP library to make API requests.

Ensure that you provide the necessary input features as specified in the API request documentation. Send the request to the appropriate endpoint, and you will receive the predicted data in the response.

Remember to adjust the API endpoint and request format according to the specific implementation and endpoint provided in the documentation.

By following the instructions above, you can set up the environment, start the web server, and make HTTP API requests to obtain predictions from the deployed LSTM model. Refer to the API request documentation for detailed information on the request format, endpoint, and parameters.

Feel free to integrate the LSTM model's predictions into your application or workflow, leveraging the power of the trained model to make accurate crop type predictions based on agricultural features.

If you have any questions or need further assistance, please refer to the API request documentation or consult the maintainers of the deployed LSTM model.




## Dataset Description

The dataset used for this project contains agricultural information and features of different crops. It includes the following columns:

- `N`: The nitrogen level in the soil.
- `P`: The phosphorus level in the soil.
- `K`: The potassium level in the soil.
- `temperature`: The temperature in Celsius.
- `humidity`: The humidity in percentage.
- `ph`: The pH value of the soil.
- `rainfall`: The rainfall in millimeters.
- `label`: The label or crop name.

The dataset provides information for multiple instances of crops, and each instance is described by its respective feature values and corresponding crop label. The dataset allows for the exploration and analysis of the relationship between the agricultural factors and the type of crop.

**Example Dataset:**

```
N   P   K   temperature  humidity   ph             rainfall    label
-----------------------------------------------------------------------
90  42  43  20.87974371  82.00274423  6.502985292  202.9355362  rice
85  58  41  21.77046169  80.31964408  7.038096361  226.6555374  rice
60  55  44  23.00445915  82.3207629   7.840207144  263.9642476  rice
74  35  40  26.49109635  80.15836264  6.980400905  242.8640342  rice
```

## LSTM Model Description

The machine learning model used for this dataset is the Long Short-Term Memory (LSTM) model. LSTM is a type of recurrent neural network (RNN) that is particularly effective in capturing long-term dependencies and sequential patterns in data.

The architecture of the LSTM model used in this project includes the following layers:

1. **Embedding Layer**: The input features are passed through an embedding layer. This layer converts the input features into a dense representation.

2. **LSTM Layer 1**: The first LSTM layer consists of 32 units. It processes the sequential data and captures temporal dependencies in the input features.

3. **LSTM Layer 2**: The second LSTM layer consists of 64 units. This layer can capture more complex temporal patterns and dependencies compared to the previous layer.

4. **Activation Function**: The softmax activation function is applied after the LSTM layers. Softmax activation produces probability distributions over the classes, allowing for multi-class classification.

5. **Dense Layer**: Following the LSTM layers, there is a dense layer. This layer performs a non-linear transformation of the LSTM layer outputs to produce the final output probabilities for the crop classes.

 **Model summary**
![Model Summary](https://github.com/VamshiKrishna-jillela/Crop-Prediction-Engine/blob/master/images/Model%20Summary.jpg?raw=true)

<br/>
<br/>


## Model Results:

The LSTM model achieved the following results on the test set:

- Accuracy: 92.1%
- Precision: 92.6%
- Recall: 92.10%
- F1-Score: 92.3%

These results indicate the overall performance of the LSTM model in correctly classifying the suitable crop types based on the provided climatic features.\
<br/>
**Accuracy vs Epoch**
![Accuracy vs Epoch](https://github.com/VamshiKrishna-jillela/Crop-Prediction-Engine/blob/master/images/Accuracy%20vs%20Epoch.jpg?raw=true)
<br/>
<br/>

**Loss vs Epoch**
![Loss vs Epoch](https://github.com/VamshiKrishna-jillela/Crop-Prediction-Engine/blob/master/images/Loss%20vs%20Epoch.jpg?raw=true)




<br/>
<br/>


## Usage

To use the dataset and LSTM model, follow these steps:

1. **Dataset**: Load the dataset into your preferred programming environment or machine learning framework.

2. **Preprocessing**: Preprocess the dataset as required. This may involve handling missing values, scaling or normalizing features, and encoding categorical variables if present.

3. **Model Setup**: Build the LSTM model according to the provided architecture. Set the parameters, such as the number of LSTM units, activation functions, and dense

layer units, as described.

4. **Training**: Split the dataset into training and testing sets. Train the LSTM model using the training set. Adjust hyperparameters, such as learning rate and batch size, as needed.

5. **Evaluation**: Evaluate the trained model's performance on the testing set. Use appropriate evaluation metrics, such as accuracy, precision, recall, and F1-score, to assess the model's classification performance.

6. **Prediction**: Once the model is trained and evaluated, it can be used to make predictions on new, unseen data. Preprocess the new data in the same manner as the training data, and use the trained LSTM model to predict the crop labels.

Remember to consult the documentation of your specific programming environment or machine learning framework for detailed instructions on loading the dataset, building LSTM models, training, and making predictions.

## Conclusion

The LSTM model, trained on the provided agricultural dataset, demonstrates a strong ability to classify crops based on the given features. By leveraging the sequential nature of the data, the model captures temporal dependencies and achieves good classification performance.

Feel free to experiment with different hyperparameters, architectures, or even expand the dataset with additional crop types to further enhance the model's capabilities.

Please note that the example dataset and results provided in this README are for illustrative purposes only. It is advisable to work with larger and more diverse datasets to build robust and reliable models.

Happy coding and exploring the world of agricultural data and LSTM models!
