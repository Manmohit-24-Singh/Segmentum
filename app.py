"""
🎯 Segmentum
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from segmentum.prediction import CustomerSegmentPredictor

# Page config
st.set_page_config(
    page_title="Segmentum",
    page_icon="🎯",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

COLOR_MAP = {
    0: '#FF6B6B', 1: '#4ECDC4', 2: '#95E1D3',
    3: '#FFE66D', 4: '#C77DFF'
}

@st.cache_resource
def load_predictor():
    predictor = CustomerSegmentPredictor()
    try:
        predictor.load_model()
        return predictor
    except FileNotFoundError:
        return None

def main():
    st.markdown('<div class="main-header">🎯 Customer Segmentation</div>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#666; font-size:1.2rem;">Predict customer segments using AI</p>', 
                unsafe_allow_html=True)
    
    predictor = load_predictor()
    
    if predictor is None:
        st.error("⚠️ Model not found! Run `python main.py` first.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("", 
        ["🏠 Overview", "🔮 Predict", "📦 Batch", "📈 Analytics"],
        label_visibility="collapsed"
    )
    
    if page == "🏠 Overview":
        show_overview(predictor)
    elif page == "🔮 Predict":
        show_prediction(predictor)
    elif page == "📦 Batch":
        show_batch(predictor)
    else:
        show_analytics(predictor)

def show_overview(predictor):
    st.header("Customer Segments Overview")
    
    summary = predictor.get_cluster_summary()
    
    # Visualization
    fig = go.Figure()
    for _, row in summary.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Avg_Income']],
            y=[row['Avg_Spending']],
            mode='markers+text',
            name=f"Cluster {row['Cluster']}",
            marker=dict(size=40, color=COLOR_MAP[row['Cluster']],
                       symbol='star', line=dict(width=2, color='white')),
            text=[f"C{row['Cluster']}"],
            textfont=dict(size=16, color='white', family='Arial Black'),
            hovertemplate=f"<b>{row['Description']}</b><br>"
                         f"Income: ${row['Avg_Income']:.0f}k<br>"
                         f"Spending: {row['Avg_Spending']:.0f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Cluster Centroids",
        xaxis_title="Annual Income (k$)",
        yaxis_title="Spending Score",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster cards
    st.subheader("📋 Segment Details")
    cols = st.columns(5)
    for idx, (col, (_, row)) in enumerate(zip(cols, summary.iterrows())):
        with col:
            st.markdown(f"""
                <div style='background:{COLOR_MAP[row['Cluster']]}; 
                            padding:1.5rem; border-radius:1rem; 
                            color:white; text-align:center;'>
                    <h2>C{row['Cluster']}</h2>
                    <p><b>{row['Description']}</b></p>
                    <p>💰 ${row['Avg_Income']:.0f}k</p>
                    <p>🛍️ {row['Avg_Spending']:.0f}</p>
                </div>
            """, unsafe_allow_html=True)

def show_prediction(predictor):
    st.header("🔮 Predict Customer Segment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Info")
        income = st.slider("Annual Income (k$)", 15, 150, 60, 5)
        spending = st.slider("Spending Score", 1, 100, 50, 1)
        
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            result = predictor.predict_single(income, spending)
            
            st.markdown(f"""
                <div style='background:{COLOR_MAP[result['cluster']]}; 
                            padding:2rem; border-radius:1rem; 
                            color:white; margin-top:1rem;'>
                    <h1 style='text-align:center;'>Cluster {result['cluster']}</h1>
                    <h3 style='text-align:center;'>{result['description']}</h3>
                    <hr style='border-color:rgba(255,255,255,0.3);'>
                    <p><b>Centroid:</b> ${result['centroid_income']}k, {result['centroid_spending']}</p>
                    <p><b>Distance:</b> {result['distance_to_centroid']:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Visual Position")
        if 'result' in locals():
            fig = create_position_plot(predictor, income, spending, result)
            st.plotly_chart(fig, use_container_width=True)

def create_position_plot(predictor, income, spending, result):
    summary = predictor.get_cluster_summary()
    fig = go.Figure()
    
    for _, row in summary.iterrows():
        opacity = 1.0 if row['Cluster'] == result['cluster'] else 0.3
        fig.add_trace(go.Scatter(
            x=[row['Avg_Income']], y=[row['Avg_Spending']],
            mode='markers', name=f"C{row['Cluster']}",
            marker=dict(size=30, color=COLOR_MAP[row['Cluster']],
                       symbol='star', opacity=opacity)
        ))
    
    fig.add_trace(go.Scatter(
        x=[income], y=[spending],
        mode='markers', name='Your Customer',
        marker=dict(size=25, color='black', symbol='diamond',
                   line=dict(width=3, color='white'))
    ))
    
    fig.update_layout(
        title="Position in Segment Space",
        xaxis_title="Income (k$)", yaxis_title="Spending",
        height=400
    )
    return fig

def show_batch(predictor):
    st.header("📦 Batch Prediction")
    
    st.info("Upload CSV with columns: annual_income, spending_score")
    
    uploaded = st.file_uploader("Choose CSV", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())
        
        if st.button("🎯 Predict All", type="primary"):
            results = predictor.predict_batch(df.to_dict('records'))
            
            st.subheader("✅ Results")
            st.dataframe(results)
            
            csv = results.to_csv(index=False)
            st.download_button("📥 Download", csv, 
                             "predictions.csv", "text/csv")
            
            # Chart
            counts = results['Cluster'].value_counts().sort_index()
            fig = px.bar(x=counts.index, y=counts.values,
                        labels={'x':'Cluster', 'y':'Count'},
                        color=counts.index, color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("📝 Sample Format")
        st.dataframe(pd.DataFrame({
            'annual_income': [50, 80, 30],
            'spending_score': [60, 85, 20]
        }))

def show_analytics(predictor):
    st.header("📈 Analytics")
    
    summary = predictor.get_cluster_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clusters", len(summary))
    with col2:
        st.metric("Income Range", 
                 f"${summary['Avg_Income'].min():.0f}k-${summary['Avg_Income'].max():.0f}k")
    with col3:
        st.metric("Spending Range",
                 f"{summary['Avg_Spending'].min():.0f}-{summary['Avg_Spending'].max():.0f}")
    
    st.dataframe(summary, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(summary, x='Cluster', y='Avg_Income',
                    color='Cluster', color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(summary, x='Cluster', y='Avg_Spending',
                    color='Cluster', color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
