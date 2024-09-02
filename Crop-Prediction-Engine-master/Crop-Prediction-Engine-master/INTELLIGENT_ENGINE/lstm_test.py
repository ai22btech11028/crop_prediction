from tensorflow import keras
from keras.models import load_model
import json
import collections
from datetime import date


model = load_model('./INTELLIGENT_ENGINE/my_model.h5')
crop=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango',
       'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas',
       'pomegranate', 'rice', 'watermelon']
parameter_Names=["N","P","K","temperature","humidity","ph","rainfall"]



def cropRecommend(Params):

    final={}
    v=[]
    extracted_params=[]

    for parameter in parameter_Names:
      extracted_params.append(float(Params[parameter]))

    v.append(extracted_params)
    rec=model.predict(v)
    Prediction_Distribution={}

    for i in range(0,22):
      Prediction_Distribution[crop[i]]=rec[0][i]*100

    dict(sorted(Prediction_Distribution.items(), key=lambda item: item[1]))
    final["label"]=Prediction_Distribution
    
    json_final = json.dumps(final,indent=4)
    return json_final



    


