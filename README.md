# 🎯 Segmentum

An end-to-end Python-based machine learning platform for customer segmentation using K-Means clustering. Segmentum analyzes mall customer data to identify distinct behavioral segments based on Annual Income and Spending Score, enabling data-driven marketing strategies and personalized customer engagement.

## 📋 Overview

Segmentum is a complete customer segmentation solution that combines machine learning with an intuitive web interface. Built with scikit-learn for robust clustering and Streamlit for interactive visualization, it helps businesses understand their customer base and tailor marketing strategies to specific segments.

## ✨ Features

### 🤖 Machine Learning Pipeline
- **K-Means Clustering**: Identifies 5 distinct customer segments
- **Model Persistence**: Save and load trained models for reuse
- **Evaluation Metrics**: Elbow method (WCSS) and Silhouette scores
- **Optimal K Selection**: Automated cluster number determination

### 🔮 Prediction Capabilities
- **Single Customer Prediction**: Classify individual customers
- **Batch Prediction**: Process multiple customers via CSV upload
- **Distance Calculation**: Measure proximity to cluster centroids
- **Confidence Metrics**: Understand prediction reliability

### 📊 Interactive Dashboard
- **Streamlit Web Interface**: User-friendly visualization
- **Real-time Predictions**: Instant customer classification
- **Interactive Charts**: Plotly-powered visualizations
- **Cluster Analytics**: Detailed segment statistics
- **CSV Export**: Download prediction results

### 📈 Visualization & Reporting
- **2D Cluster Plots**: Visual representation of segments
- **Centroid Visualization**: Cluster center identification
- **Evaluation Graphs**: WCSS and Silhouette score plots
- **Summary Statistics**: Income and spending analysis

## 🎨 Customer Segments

Segmentum identifies 5 distinct customer segments:

| Cluster | Segment Name | Characteristics | Color | Marketing Strategy |
|---------|--------------|-----------------|-------|-------------------|
| 0 | Careful Spenders | Low income, low spending | 🔴 Red | Budget-friendly offerings |
| 1 | Target Group | High income, high spending | 🔵 Blue | Premium products & VIP services |
| 2 | Sensible Customers | High income, low spending | 🟢 Green | Value propositions & quality |
| 3 | Careless Customers | Low income, high spending | 🟡 Yellow | Credit options & financial planning |
| 4 | Standard Customers | Medium income, medium spending | 🟣 Magenta | Balanced marketing approach |

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.7+** - Primary programming language
- **scikit-learn** - Machine learning and clustering
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations

### Visualization
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts and graphs
- **Matplotlib** - Static plot generation

### Model Management
- **Pickle** - Model serialization and persistence

## 📦 Prerequisites

Before you begin, ensure you have:

- **Python** 3.7 or higher
- **pip** package manager
- **Git** (for cloning the repository)

### Installation Commands

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3 python3-pip
python3 --version
```

#### macOS
```bash
brew install python3
python3 --version
```

#### Windows
Download and install [Python](https://www.python.org/downloads/) (ensure "Add to PATH" is checked)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Manmohit-24-Singh/Segmentum.git
cd Segmentum
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
streamlit>=1.28.0
plotly>=5.17.0
```

### 3. Verify Installation

```bash
python --version
pip list | grep -E "(pandas|scikit-learn|streamlit)"
```

## 🎯 Usage

Segmentum provides two main interfaces: a **training pipeline** and an **interactive dashboard**.

### Training the Model

Run the main pipeline to train the K-Means model:

```bash
python main.py
```

**What happens:**
1. ✅ Loads customer data from `data/Mall_Customers.csv`
2. 🔄 Preprocesses and renames columns
3. 📊 Calculates WCSS and Silhouette scores
4. 📈 Generates evaluation plots in `graphs/`
5. 🤖 Trains K-Means model (K=5)
6. 💾 Saves trained model to `models/kmeans_model.pkl`
7. 🎨 Creates cluster visualization
8. 🔮 Runs quick prediction demo

**Output:**
```
✓ Dataset loaded successfully!
Shape: (200, 5)
────────────────────────────────────────────────
✓ Columns renamed: ['CustomerID', 'Gender', 'Age', 'Annual Income', 'Spending Score']
...
✓ Evaluation plots saved to 'graphs/'
✓ Cluster plot saved to 'graphs/2d_clusters.png'
✓ Model saved to 'models/kmeans_model.pkl'
────────────────────────────────────────────────
✅ PIPELINE COMPLETE!
```

### Launching the Dashboard

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📱 Dashboard Features

### 🏠 Overview Page
- **Cluster Visualization**: Interactive scatter plot of all segments
- **Segment Cards**: Quick stats for each customer segment
- **Centroid Display**: Visual representation of cluster centers

### 🔮 Predict Page
- **Single Customer Input**:
  - Annual Income slider (15k - 150k)
  - Spending Score slider (1-100)
- **Instant Prediction**: Real-time cluster assignment
- **Visual Position**: See where customer fits in segment space
- **Detailed Results**:
  - Cluster number and description
  - Distance to centroid
  - Centroid coordinates

### 📦 Batch Prediction
- **CSV Upload**: Process multiple customers at once
- **Data Preview**: Verify uploaded data
- **Bulk Prediction**: Classify all customers
- **Results Table**: View all predictions
- **CSV Export**: Download results with predictions
- **Distribution Chart**: Visualize cluster distribution

### 📈 Analytics
- **Summary Metrics**:
  - Total number of clusters
  - Income range across segments
  - Spending range across segments
- **Detailed Statistics**: Full cluster summary table
- **Visual Analytics**: Bar charts for income and spending by cluster

## 📂 Project Structure

```
Segmentum/
├── app.py                      # Streamlit dashboard application
├── main.py                     # Training pipeline entry point
├── requirements.txt            # Python dependencies
├── data/
│   └── Mall_Customers.csv     # Customer dataset (200 records)
├── segmentum/                 # Core package
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── data_processing.py     # Preprocessing functions
│   ├── clustering.py          # K-Means clustering logic
│   ├── evaluation.py          # Model evaluation metrics
│   └── prediction.py          # Prediction and model management
├── models/                    # Trained models (generated)
│   └── kmeans_model.pkl
└── graphs/                    # Visualization outputs (generated)
    ├── 2d_clusters.png        # Cluster scatter plot
    ├── elbow_method_wcss.png  # Elbow method graph
    └── silhouette_scores.png  # Silhouette score plot
```

### Module Descriptions

#### `app.py`
Main Streamlit dashboard with four pages:
- Overview: Cluster visualization and segment cards
- Predict: Single customer prediction
- Batch: CSV upload and bulk prediction
- Analytics: Statistical analysis and charts

#### `main.py`
End-to-end training pipeline:
1. Load and preprocess data
2. Evaluate optimal K
3. Train K-Means model
4. Generate visualizations
5. Save model for predictions

#### `segmentum/data_loader.py`
- `load_data()`: Loads CSV from data directory

#### `segmentum/data_processing.py`
- `apply_column_renames()`: Standardizes column names

#### `segmentum/clustering.py`
- `prepare_2d_data()`: Extracts features for clustering
- `run_kmeans_2d()`: Trains K-Means model
- `plot_2d_clusters()`: Creates cluster visualization
- `get_customers_by_cluster()`: Filters customers by segment

#### `segmentum/evaluation.py`
- `calculate_wcss_and_silhouette()`: Computes evaluation metrics
- `plot_cluster_evaluation()`: Generates elbow and silhouette plots

#### `segmentum/prediction.py`
- `CustomerSegmentPredictor`: Main prediction class
  - `save_model()`: Persist trained model
  - `load_model()`: Load saved model
  - `predict_single()`: Classify one customer
  - `predict_batch()`: Classify multiple customers
  - `get_cluster_summary()`: Get cluster statistics

## 📊 Dataset

### Mall Customers Dataset

**Source**: `data/Mall_Customers.csv`

**Description**: Customer data from a shopping mall with 200 records.

**Features**:
- `CustomerID`: Unique identifier (0001-0200)
- `Genre/Gender`: Male or Female
- `Age`: Customer age (18-70)
- `Annual Income (k$)`: Annual income in thousands (15-137)
- `Spending Score (1-100)`: Mall-assigned spending score (1-99)

**Sample Data**:
```csv
CustomerID,Genre,Age,Annual Income (k$),Spending Score (1-100)
0001,Male,19,15,39
0002,Male,21,15,81
0003,Female,20,16,6
...
```

## 🔍 How It Works

### 1. Data Preprocessing
```python
# Column standardization
RENAME_MAP = {
    'Genre': 'Gender',
    'Annual Income (k$)': 'Annual Income',
    'Spending Score (1-100)': 'Spending Score'
}
df = apply_column_renames(df, RENAME_MAP)
```

### 2. Feature Selection
```python
# Extract clustering features
data_2d = df[['Annual Income', 'Spending Score']]
```

### 3. Model Training
```python
# K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=2, n_init='auto')
kmeans.fit(data_2d)
```

### 4. Prediction
```python
# Classify new customer
predictor = CustomerSegmentPredictor()
predictor.load_model()
result = predictor.predict_single(income=75, spending=80)
# Output: Cluster 1 - Target Group (High income, high spending)
```

## 💡 Example Workflows

### Workflow 1: Train and Predict

```bash
# Step 1: Train the model
python main.py

# Step 2: Test predictions
python -m segmentum.prediction

# Step 3: Launch dashboard
streamlit run app.py
```

### Workflow 2: Single Customer Prediction

```python
from segmentum.prediction import CustomerSegmentPredictor

predictor = CustomerSegmentPredictor()
predictor.load_model()

# Predict for a customer
result = predictor.predict_single(
    annual_income=80,  # $80k income
    spending_score=85   # High spending
)

print(f"Cluster: {result['cluster']}")
print(f"Description: {result['description']}")
print(f"Distance to center: {result['distance_to_centroid']:.2f}")
```

**Output:**
```
Cluster: 1
Description: Target Group - High income, high spending
Distance to center: 3.45
```

### Workflow 3: Batch Prediction

Create `customers.csv`:
```csv
annual_income,spending_score
50,60
80,85
30,20
90,15
```

```python
import pandas as pd
from segmentum.prediction import CustomerSegmentPredictor

predictor = CustomerSegmentPredictor()
predictor.load_model()

# Load and predict
df = pd.read_csv('customers.csv')
results = predictor.predict_batch(df.to_dict('records'))

print(results)
results.to_csv('predictions.csv', index=False)
```

## 📈 Understanding the Clusters

### Cluster 0: Careful Spenders (Red)
- **Income**: $15k - $40k
- **Spending**: 1-40
- **Strategy**: Budget-friendly products, discounts, loyalty programs

### Cluster 1: Target Group (Blue)
- **Income**: $70k - $137k
- **Spending**: 60-99
- **Strategy**: Premium products, VIP treatment, exclusive offers

### Cluster 2: Sensible Customers (Green)
- **Income**: $70k - $120k
- **Spending**: 1-40
- **Strategy**: Quality focus, value propositions, savings programs

### Cluster 3: Careless Customers (Yellow)
- **Income**: $15k - $40k
- **Spending**: 60-99
- **Strategy**: Credit options, installment plans, financial education

### Cluster 4: Standard Customers (Magenta)
- **Income**: $40k - $70k
- **Spending**: 40-60
- **Strategy**: Balanced approach, seasonal promotions, mid-range products

## 🔧 Configuration

### Adjusting Number of Clusters

Edit `segmentum/clustering.py`:
```python
N_CLUSTERS = 5  # Change to desired number
```

### Customizing Colors

Edit `app.py`:
```python
COLOR_MAP = {
    0: '#FF6B6B',  # Red
    1: '#4ECDC4',  # Teal
    2: '#95E1D3',  # Mint
    3: '#FFE66D',  # Yellow
    4: '#C77DFF'   # Purple
}
```

### Updating Cluster Descriptions

Edit `segmentum/prediction.py`:
```python
self.cluster_descriptions = {
    0: "Your custom description",
    1: "Another description",
    # ... etc
}
```

## 🐛 Troubleshooting

### Model Not Found Error

**Error:**
```
⚠️ Model not found! Run `python main.py` first.
```

**Solution:**
```bash
python main.py  # Train the model first
```

### Streamlit Not Opening

**Error:**
```
streamlit: command not found
```

**Solution:**
```bash
pip install streamlit
streamlit run app.py
```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'segmentum'
```

**Solution:**
Ensure you're in the project root directory:
```bash
cd Segmentum
python main.py
```

### CSV Upload Format Error

**Error:**
```
KeyError: 'annual_income'
```

**Solution:**
Ensure your CSV has these exact column names:
```csv
annual_income,spending_score
50,60
```

## 🚀 Future Enhancements

- [ ] **Additional Features**: Include Age and Gender in clustering
- [ ] **3D Clustering**: Visualize multi-dimensional segments
- [ ] **Hierarchical Clustering**: Alternative clustering methods
- [ ] **DBSCAN**: Density-based clustering option
- [ ] **Time Series**: Track segment evolution over time
- [ ] **Customer Lifetime Value**: Predict CLV by segment
- [ ] **Recommendation Engine**: Product recommendations per segment
- [ ] **A/B Testing**: Test marketing strategies by segment
- [ ] **API Endpoint**: RESTful API for predictions
- [ ] **Database Integration**: Connect to live customer database
- [ ] **Automated Reporting**: Scheduled email reports
- [ ] **Multi-language Support**: Internationalization
- [ ] **Mobile App**: iOS/Android application
- [ ] **Real-time Updates**: Live dashboard updates
- [ ] **Export Options**: PDF reports, PowerPoint presentations

## 📊 Performance Metrics

### Model Evaluation

**Optimal K Selection:**
- Elbow Method: Identifies K=5 as optimal
- Silhouette Score: 0.55+ for K=5 (good separation)

**Clustering Quality:**
- Within-Cluster Sum of Squares (WCSS): Minimized at K=5
- Between-Cluster Variance: Maximized for distinct segments

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features
- Update documentation for changes

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Manmohit Singh**
- GitHub: [@Manmohit-24-Singh](https://github.com/Manmohit-24-Singh)
- Project: [https://github.com/Manmohit-24-Singh/Segmentum](https://github.com/Manmohit-24-Singh/Segmentum)

## 📞 Support

For support, issues, or feature requests:
- Open an issue on [GitHub Issues](https://github.com/Manmohit-24-Singh/Segmentum/issues)
- Contact the maintainer through GitHub

---
