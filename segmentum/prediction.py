import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List
from sklearn.cluster import KMeans

class CustomerSegmentPredictor:
    """
    Save, load, and use a trained K-Means model for customer segment predictions.
    """
    
    def __init__(self, model_path: str = 'models/kmeans_model.pkl'):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = ['Annual Income', 'Spending Score']
        
        # Cluster descriptions (update these based on your actual cluster patterns)
        self.cluster_descriptions = {
            0: "Careful Spenders - Low income, low spending",
            1: "Target Group - High income, high spending", 
            2: "Sensible Customers - High income, low spending",
            3: "Careless Customers - Low income, high spending",
            4: "Standard Customers - Medium income, medium spending"
        }
    
    def save_model(self, kmeans_model: KMeans):
        """Save the trained K-Means model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(kmeans_model, f)
        
        self.model = kmeans_model
        print(f"✓ Model saved to '{self.model_path}'")
    
    def load_model(self):
        """Load a trained K-Means model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at '{self.model_path}'")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ Model loaded from '{self.model_path}'")
    
    def predict_single(self, annual_income: float, spending_score: float) -> Dict:
        """Predict cluster for a single customer."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        input_data = pd.DataFrame({
            'Annual Income': [annual_income],
            'Spending Score': [spending_score]
        })
        
        cluster_label = self.model.predict(input_data)[0]
        centroid = self.model.cluster_centers_[cluster_label]
        
        distance = ((annual_income - centroid[0])**2 + 
                   (spending_score - centroid[1])**2)**0.5
        
        return {
            'cluster': int(cluster_label),
            'description': self.cluster_descriptions.get(cluster_label, "Unknown"),
            'centroid_income': round(centroid[0], 2),
            'centroid_spending': round(centroid[1], 2),
            'distance_to_centroid': round(distance, 2)
        }
    
    def predict_batch(self, customers: List[Dict[str, float]]) -> pd.DataFrame:
        """Predict clusters for multiple customers."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        df = pd.DataFrame(customers)
        df = df.rename(columns={
            'annual_income': 'Annual Income',
            'spending_score': 'Spending Score'
        })
        
        predictions = self.model.predict(df[self.feature_names])
        df['Cluster'] = predictions
        df['Description'] = df['Cluster'].map(self.cluster_descriptions)
        
        return df
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary of all clusters with their centroids."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        centroids = self.model.cluster_centers_
        
        summary = pd.DataFrame({
            'Cluster': range(len(centroids)),
            'Avg_Income': centroids[:, 0].round(2),
            'Avg_Spending': centroids[:, 1].round(2),
            'Description': [self.cluster_descriptions.get(i, "Unknown") 
                          for i in range(len(centroids))]
        })
        
        return summary


if __name__ == "__main__":
    """Demo: How to use the predictor"""
    print("\n" + "="*80)
    print("CUSTOMER SEGMENT PREDICTION DEMO")
    print("="*80)
    
    predictor = CustomerSegmentPredictor()
    
    try:
        predictor.load_model()
        
        print("\n📊 CLUSTER SUMMARY:")
        print(predictor.get_cluster_summary().to_string(index=False))
        
        print("\n" + "-"*80)
        print("🔮 SINGLE CUSTOMER PREDICTIONS:")
        print("-"*80)
        
        test_customers = [
            {"name": "High Earner, Big Spender", "income": 80, "spending": 85},
            {"name": "Budget Conscious", "income": 30, "spending": 20},
            {"name": "Conservative Saver", "income": 90, "spending": 15},
        ]
        
        for customer in test_customers:
            result = predictor.predict_single(customer['income'], customer['spending'])
            print(f"\n👤 {customer['name']}:")
            print(f"   Income: ${customer['income']}k | Spending: {customer['spending']}")
            print(f"   → Cluster {result['cluster']}: {result['description']}")
        
        print("\n" + "="*80)
        
    except FileNotFoundError:
        print("\n❌ Model not found! Run main.py first to train the model.")