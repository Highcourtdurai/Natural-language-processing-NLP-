import pandas as pd
import numpy as np
import tensorflow.keras as tf
ratings=pd.read_csv("ratings.csv")
books=pd.read_csv("books.csv")

print(ratings.head(5))
print(books.head(5))

nbooks=ratings.book_id.nunique()
nusers=ratings.user_id.nunique()

print(nbooks)
print(nusers)

from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(ratings,test_size=0.2,random_state=10)

book_input=tf.Input(shape=[1]) #output=[None,1]
book_embedding=tf.layers.Embedding(nbooks+1,15)(book_input)
book_out=tf.layers.Flatten()(book_embedding)

user_input=tf.Input(shape=[1]) #output=[None,1]
user_embedding=tf.layers.Embedding(nusers+1,15)(user_input)
user_out=tf.layers.Flatten()(user_embedding)

concadinate=tf.layers.Concatenate()([book_out,user_out])
dense_1=tf.layers.Dense(120,activation='relu')(concadinate)
out=tf.layers.Dense(1,activation='relu')(dense_1)

model=tf.Model(inputs=[book_input,user_input],outputs=out)

model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit([x_train.book_id,x_train.user_id],x_train.rating,validation_data=([x_test.book_id,x_test.user_id],x_test.rating),epochs=5,batch_size=64)

b_id=list(ratings.book_id.unique())
print(b_id)

book_arr=np.array(b_id)#get all book ids
user=np.array([80 for i in range(len(b_id))])

pred=model.predict([book_arr,user])
print(pred)

pred=pred.reshape(-1)
print(pred)

pred_ids=(-pred).argsort()[0:5]
print(pred_ids)

print(books.iloc[pred_ids])



