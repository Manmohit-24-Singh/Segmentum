from segmentum.data_loader import load_data
from segmentum.data_processing import apply_column_renames
from segmentum.clustering import (
    prepare_2d_data, 
    run_kmeans_2d, 
    plot_2d_clusters, 
    get_customers_by_cluster
)
from segmentum.evaluation import (
    calculate_wcss_and_silhouette,
    plot_cluster_evaluation
)
from segmentum.prediction import CustomerSegmentPredictor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """
    Main pipeline: train model, evaluate, save for predictions.
    """
    RENAME_MAP = {
        'Genre': 'Gender',
        'Annual Income (k$)': 'Annual Income',
        'Spending Score (1-100)': 'Spending Score'
    }

    # 1. Load Data
    df = load_data()
    print("\n✓ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print('-'*100)

    # 2. Preprocess
    df = apply_column_renames(df, RENAME_MAP)
    print("✓ Columns renamed:", list(df.columns))
    print(df.head())
    print('-'*100)
    
    # 3. Prepare 2D Data
    data_2d = prepare_2d_data(df)
    
    # 4. Evaluate Clusters
    wcss_data, sil_data = calculate_wcss_and_silhouette(data_2d)
    print('-'*100)
    
    # 5. Plot Evaluation
    plot_cluster_evaluation(wcss_data, sil_data)
    print("✓ Evaluation plots saved to 'graphs/'")
    print('-'*100)

    # 6. Train K-Means (K=5)
    data_2d_labeled, kmeans_model = run_kmeans_2d(data_2d)
    print('-'*100)

    # 7. Plot Clusters
    plot_2d_clusters(data_2d_labeled, kmeans_model) 
    print("✓ Cluster plot saved to 'graphs/2d_clusters.png'")
    print('-'*100)
    
    # 8. 🎯 SAVE MODEL FOR PREDICTIONS
    print("\n--- Saving Model ---")
    predictor = CustomerSegmentPredictor()
    predictor.save_model(kmeans_model)
    print('-'*100)
    
    # 9. Get Green Cluster Customers
    target_cluster_color = 'green'
    green_cluster_ids = get_customers_by_cluster(df, data_2d_labeled, target_cluster_color)
    print(f"\n✓ Found {len(green_cluster_ids)} customers in '{target_cluster_color}' cluster")
    print('-'*100)
    
    # 10. 🔮 QUICK PREDICTION DEMO
    print("\n--- Quick Prediction Test ---")
    test_result = predictor.predict_single(annual_income=75, spending_score=80)
    print(f"\n🎯 Test Customer (Income: $75k, Spending: 80):")
    print(f"   → Cluster {test_result['cluster']}: {test_result['description']}")
    print(f"   → Distance to center: {test_result['distance_to_centroid']:.2f}")
    
    print('\n' + '='*100)
    print("✅ PIPELINE COMPLETE!")
    print("\n💡 Next Steps:")
    print("   1. Run: python -m segmentum.prediction     (see more predictions)")
    print("   2. Run: streamlit run app.py               (launch dashboard)")
    print('='*100)


if __name__ == "__main__":
    main()