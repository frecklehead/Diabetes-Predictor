# -*- coding: utf-8 -*-
import numpy as np
import pickle

model= pickle.load(open('/home/frecklehead/Desktop/daibetes/trained_model.sav','rb'))
input=(6,148,72,35,	0,33.6,0.627,50	)
# changing to a numpy array
imput_data_as_numpy_array=np.asarray(input)
# reshape the array as we are predicting for one instance
input_data_reshaped=imput_data_as_numpy_array.reshape(1,-1)
#standardize the input


prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
 print('the person is not diabetic')
else:
 print('the person is diabetic')