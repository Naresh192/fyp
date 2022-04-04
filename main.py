!pip install matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.backend import manual_variable_initialization 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,SpatialDropout1D
from keras.layers.embeddings import Embedding
manual_variable_initialization(True)
import tensorflow
from numpy.random import seed
seed(42)
tensorflow.random.set_seed(42)
import pickle
file_to_read = open("/content/drive/MyDrive/stored_object.pickle", "rb")
z=pickle.load(file_to_read)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
t=Tokenizer(num_words=50000,lower=True)
t.fit_on_texts(z)
x=t.texts_to_sequences(z)
x=sequence.pad_sequences(x,maxlen=250)
model=Sequential()
model.add(Embedding(50000,100,input_length=x.shape[1]))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(2,activation='sigmoid'))
model.load_weights("/content/drive/MyDrive/modeln.h5")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
def get() :
  x=t.texts_to_sequences([title]) # pro 
  x=sequence.pad_sequences(x,maxlen=250)
  p=model.predict(x)
  print(p)
  pred=np.argmax(p,axis=1)
  print(pred)
  if pred[0]==1:
    st.write('yes')
  else:
    st.write('no')

st.set_page_config(layout="wide",menu_items=None)
title = st.text_input('Text', '')
if st.button('Submit'):
     get()
st.title("Twitter Sentiment Analyser")


col1,col2 = st.columns(2)

option = col1.selectbox(
     'Select Brand Name',
     ('Iphone', 'Samsung', 'Xiaomi'))
col1.title("Brand Info")
old= col2.select_slider('Select how old tweets can be (in days)',options=['1', '2', '3', '4', '5', '6', '7'])
colx1,colx2,colx3,colx4=st.columns(4)
colx1.image("https://media.wired.com/photos/5bcea2642eea7906bba84c67/master/w_2560%2Cc_limit/iphonexr.jpg")
colx2.write("Name: option")
colx2.write("Model: option")
colx2.write("Color: option")
colx2.write("Price: option")
col2.title("Data Frame")
df = pd.DataFrame(
    np.random.randn(50, 20),
    columns=('col %d' % i for i in range(20)))

colx3.dataframe(df)
coly1,coly2,coly3=st.columns(3)
chart_data = pd.DataFrame(
     np.random.randn(50, 2),
     columns=["a", "b"])


labels = 'Yes', 'No'
sizes = [70, 30]
explode = (0.1, 0)
coly1.write('Pie Chart')
fig1, ax1 = plt.subplots()
fig1.set_facecolor('#0D1117')
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,textprops={'color':"w"})
ax1.axis('equal')

coly1.pyplot(fig1)
coly2.write('Bar Chart')
coly2.bar_chart(chart_data,width=500, height=540)
coly3.write('Map data')
d={'latitude':[34.053570000000036],'longitude':[-118.24544999999995]}
data_of_map = pd.DataFrame(d)
coly3.map(data_of_map)      #-118.24544999999995 34.053570000000036
