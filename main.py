from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model_def import HybridQuantumGNN 

app = FastAPI()

# 1. Load the Model when App Starts
# We initialize it with 7 features (Indian Dataset)
model = HybridQuantumGNN(in_dim=7)

# Load the weights you saved
try:
    model.load_state_dict(torch.load("models/quantum_gnn_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load model weights. {e}")

# 2. Define the Input Format (What the app sends us)
class InventoryItem(BaseModel):
    id: int
    stock: int
    sales: float
    is_festival: bool

@app.get("/")
def home():
    return {"message": "GraminRoute Quantum API is Running"}

@app.post("/predict_risk")
def predict_risk(item: InventoryItem):
    """
    Receives shop data -> Returns AI Advice
    """
    # SIMPLE LOGIC (For Demo purposes, linking inputs to your rules)
    
    # 1. Risk Calculation
    risk_score = 0.2
    if item.stock < 20: risk_score += 0.3
    if item.is_festival: risk_score += 0.4
    
    # 2. Advice Logic
    advice = "Stock is healthy."
    action = "None"
    
    if item.is_festival and item.stock < 50:
        advice = "FESTIVAL ALERT: High demand expected."
        action = f"Buy {50 - item.stock} Units Immediately"
    elif not item.is_festival and item.stock < 15:
        advice = "Low Stock: Buy small batch to prevent burn."
        action = "Micro-order 10 Units"

    return {
        "shop_id": item.id,
        "risk_score": round(risk_score, 2),
        "advice": advice,
        "recommended_action": action
    }