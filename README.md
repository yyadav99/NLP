## Content
  - Overview
  - Implementation
  - Prerequisites
  - Flow of code
  - Steps to use pre-trained model
  - Dataset
  
### Overview
It a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding. The goal of this project is to develop a model that predicts the comment is positive or negative in the case of Twitter_Sentiment_Analysis or Predict which Tweets are about real disasters and which ones are not. The Twitter_Sentiment_Analysis dataset contains 80000 positive tweets and 80000 negative tweets and disaster tweets data set contains 4000 of original disaster tweets and 3800 of fake disaster tweets. Accuracy for Twitter_Sentiment_Analysis model is 80+ and for Disaster_Tweets model is 86+.

### NLP can be implemented in two ways:
1. Using Natural Language Toolkit(NLTK).
2. Using Keras lib.  

### Prerequisites
1. Python Libraries:
   - Numpy
   - Pandas
   - MatplotLib/Seaborn
   - Sklearn
   - Keras
   - NLTK
   - RE
2. What is Embedding !
3. Text Cleaning Using Regular expression.
4. Basic's of LSTM(Long-Short-Term-Memory).

### Flow of Code  
1. Reading dataset.
2. Analyzing dataset.
3. Text cleaning i.e removing useless data that doesn't contain much importance.
4. The padding of data.
5. Training the model with our padded data.
6. Predicting the output.

### Steps to use pre-trained model
1. Download the pre-trained model for [Disaster_Prediction](https://drive.google.com/file/d/1jE0AMsd3nEylaEV8-jOsE2_NJfJ8pDpo/view?usp=sharing) and [Twitter-Sentiment-Analysis](https://drive.google.com/file/d/1sJBjL83OmAtU1uvAtCCIJaUohD3zFGys/view?usp=sharing)
2. To predict the model you have to define the following function:
   - Change the path variable.
```
def predict(text):
    import cv2
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    path="path of the model file downloaded"
    model = tf.keras.models.load_model(path)
    tokenizer = Tokenizer()
    max_len=150
    truncating_type="post"
    padded_sententnce = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=max_len,truncating=truncating_type)
    score = model.predict(padded_sententnce)
    print(score)
    if score[0][0]>0.5:
        return 1
    else:
        return 0
```
3. Then call the function with 1 parameter containing the text.
```
predict("Text that you want to predict")
If the output is 1 else the output is 0
```
### Dataset
1.Disaster_Prediction 
  - [Disaster_Tweets_Train](https://drive.google.com/file/d/11_irNcQZ8RXU6jsFfhPibBFabH4tBTTD/view?usp=sharing)
  - [Disaster_Tweets_Test](https://drive.google.com/file/d/16D322c5lItAWTQmJ0jc9PUuk_uONNetP/view?usp=sharing)
2. Twitter-Sentiment-Analysis:
  - [Data](https://drive.google.com/file/d/1-scavles7zrDIdb4N8Rgi8SUOleZcPB8/view?usp=sharing)

**If you encounter any issue while using code or model, feel free to contact me yadavyogesh9999@gmail.com.**
