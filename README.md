# Fake News Detection using Bidirectional LSTM

This project implements a Bidirectional LSTM (Bi-LSTM) model to detect fake news. The model improves accuracy by considering both forward and backward dependencies in text sequences.

## What is Bidirectional LSTM?
Bidirectional LSTM (Bi-LSTM) is an advanced type of Recurrent Neural Network (RNN) that improves performance by processing input data in both forward and backward directions. Unlike a standard LSTM, which processes sequences in only one direction (past → future), Bi-LSTM captures context from both past and future words in a sentence, making it highly effective for **text classification tasks like fake news detection.

![image](https://github.com/user-attachments/assets/3d2eb118-553e-4858-88d3-1010abc57ecd)


## Why Use Bidirectional LSTM for Fake News Detection?
Fake news detection relies on context understanding. A standard LSTM might miss important dependencies because it only looks at words from left to right. Bidirectional LSTM overcomes this by learning relationships from both directions, making it better at capturing:
1. Long-term dependencies** between words
2. Context-aware sentence understanding
3. More accurate classification of fake vs. real news  

This bidirectional nature is why Bi-LSTM model achieved 90% accuracy, outperforming a regular LSTM.

## Dataset
I used the Fake News dataset from Kaggle for training the model. You can access it using the button below:

[![Access Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/c/fake-news/data)

## Your Model Architecture Explained
1️. Text Vectorization – Converts raw text into numerical vectors.  
2️. Embedding Layer – Converts words into dense vectors with semantic meaning.  
3️. Bidirectional LSTM Layer – Processes input in **both directions** to capture full context.  
4️. Dense Layer – Outputs a probability score for fake vs. real news.  
5️. Model Training – Optimized using loss functions (e.g., binary cross-entropy) and backpropagation.

## Installation & Usage
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```python
python train.py
```

### 4. Predict Fake News
```python
python Fake_news_detection_using_BidiretcionalLSTM.ipynb
```

## Model Training Code
```python
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional

# Define model
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length ))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.build(input_shape=(None, max_length))

# Compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Print summary
print(model.summary())
```

## Contributing
Feel free to contribute by opening issues or submitting pull requests!

## License
This project is licensed under the MIT License.



