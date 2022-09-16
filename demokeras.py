#load data va chia train, val,test
from keras import Sequential
from keras.layers import Dense
import  numpy
from keras.models import load_model
from numpy import loadtxt
from sklearn.model_selection import train_test_split
dataset = loadtxt('idiansdiebete.data.csv', delimiter=',')

X=dataset[:,0:8] #input
print(X)
y=dataset[: ,8] #output
print(y)

# #chia du lieu train va val,du lieu test, 0,8 la trai va val, 0.2 la test
# X_train_val, X_test, y_train_val, y_test=train_test_split(X,y,test_size=0.2)
# #chia du lieu train va val, train 80%, val 20%
# X_train, X_val, y_train, y_val=train_test_split(X_train_val,y_train_val,test_size=0.2)
#
# #xay model
# model=Sequential()
# model.add(Dense(16, input_dim=8, activation='relu'))# thong so thu nhat la so neural, input_dim la so chieu cua input=8,activition=function
# model.add(Dense(8,activation='relu'))
# model.add(Dense(1,activation='sigmoid')) #sigmoid ham kieu dong bien
#
# #summary model
# model.summary() #show cai model len
# #compile mode
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# #train model
# model.fit(X_train, y_train, epochs=100 ,batch_size=8, validation_data=(X_val ,y_val))
# #save model
# model.save("mymodel.h5")
# #load modle o file khac
# model=load_model('mymodel.h5')
# loss,acc=model.evaluate(X_test,y_test)
# print("Loss= ",loss)
# print("ACC= ",acc)
# #predict
# X_new=X_test[10]
# y_new=y_test[10]
# #convert X_new to tensorflow type
# X_new=numpy.expand_dims(X_new,axis=0)
#
# y_predict=model.predict(X_new)
# result="Tieu duong (1)"
# if y_predict <=0.5:
#     result="khong tieu duong (0)"
# print("Gia tri du doan= ", result)
# print("gia tri dung la", y_new)



