# Importing Necessary modules
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Declaring our FastAPI instance
app = FastAPI()

pickle_in = open('classifier.pkl','rb')
lm = pickle.load(pickle_in)


class request_body(BaseModel):
    Hours: float

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'Welcome to GeeksforGeeks!'}

# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str):
	# Defining a function that takes only string as input and output the
	# following message.
	return {'Welcome to MyDEPLOYEMENT!', f'{name}'}

@app.post('/predict')
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    data.dict()
    print(data)
    Hours = data['Hours']    
    prediction = lm.predict([Hours])
    return prediction

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn basic-app:app --reload
    