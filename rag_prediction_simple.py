"""
Simple RAG Prediction Script
Integrates with rebalanced_predictive_model.py to add RAG analysis
Displays everything in terminal - no web interface needed
"""

import json
import numpy as np
import google.generativeai as genai
import chromadb
from typing import List, Dict

# --- CONFIGURATION ---
API_KEY = "AIzaSyCRoPMUb-6drAaCqXkZSkeq5Jp4kGvLm1E"
KNOWLEDGE_BASE_FILE = "rag_knowledge_base.json"

# Configure Gemini
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Global variables
knowledge_base = []
collection = None

def load_rag_system():
    """Load RAG knowledge base and vector DB"""
    global collection, knowledge_base
    
    print("\\nLoading RAG system...")
    
    # Load knowledge base
    with open(KNOWLEDGE_BASE_FILE, 'r') as f:
        knowledge_base = json.load(f)
    print(f"Loaded {len(knowledge_base):,} historical scenarios")
    
    # Get vector DB collection
    try:
        collection = chroma_client.get_collection("portfolio_scenarios")
        print(f"Vector DB ready with {collection.count():,} entries")
    except:
        print("Vector DB not found. Please run rag_api_improved.py first to build it.")
        return False
    
    return True

def create_embedding_vector(indicators: Dict) -> List[float]:
    """Create embedding vector from indicators"""
    vector = [
        indicators.get("rsi", 50) / 100.0,
        np.tanh(indicators.get("macd", 0) / 10000.0),
        indicators.get("volatility", 0.15),
        np.tanh(indicators.get("alligator_jaw", 0) / 1000000.0),
        np.tanh(indicators.get("alligator_teeth", 0) / 1000000.0),
        np.tanh(indicators.get("alligator_lips", 0) / 1000000.0),
    ]
    return vector

def find_similar_scenarios(current_indicators: Dict, k: int = 20) -> List[Dict]:
    """Find similar historical scenarios"""
    # Create query embedding
    query_embedding = create_embedding_vector(current_indicators)
    
    # Add placeholder values for weights and target return
    query_embedding.extend([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights placeholder
    query_embedding.append(0.0)  # Target return placeholder
    
    # Query vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Get full scenario data with distances
    similar_scenarios = []
    for i, idx in enumerate(results['ids'][0]):
        scenario_idx = int(idx.split('_')[1])
        scenario = knowledge_base[scenario_idx].copy()
        # Add distance from query results
        scenario['distance'] = results['distances'][0][i]
        similar_scenarios.append(scenario)
    
    return similar_scenarios

