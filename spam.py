from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model, vectorizer, and label encoder
model = joblib.load("spam.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Input data format
class ModelInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "This is an API created for Naive Bayes"}

@app.post("/predict/")
def predict(data: ModelInput):
    input_text = data.text
    user_data = vectorizer.transform([input_text])
    prediction = model.predict(user_data)
    label = label_encoder.inverse_transform(prediction)[0]  # Convert 0/1 to "notspamv "/"spam"
    return {"prediction": label}