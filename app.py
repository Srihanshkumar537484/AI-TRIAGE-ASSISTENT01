from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# 1. FastAPI app banayein aur docs URL ko root par set karein
app = FastAPI(docs_url="/", redoc_url=None) # Yahan badlav hai

# CORS middleware jodein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Apne trained model aur vectorizer ko load karein
try:
    model = joblib.load("triage_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
except FileNotFoundError:
    print("Error: 'triage_model.joblib' or 'tfidf_vectorizer.joblib' file not found. Please make sure they are in the same directory.")
    exit()

# 3. User ke input data ke liye ek model banayein
class TriageRequest(BaseModel):
    symptoms: str

# 4. API endpoint banayein
@app.post("/predict")
def predict_triage(request: TriageRequest):
    # a. User ke symptoms ko ek list mein daalein
    symptoms_list = [request.symptoms]

    # b. Vectorizer ka istemal karke text ko numbers mein badlein
    symptoms_vectorized = vectorizer.transform(symptoms_list)

    # c. Model se prediction karwayein
    prediction = model.predict(symptoms_vectorized)
    predicted_disease = prediction[0]

    # d. Prediction ko JSON format mein return karein
    return {"predicted_disease": predicted_disease}