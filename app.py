import streamlit as st
import pandas as pd
import json
import plotly.express as px
from sentiment_analysis import SentimentAnalyzer
import torch
import requests
import os

# Thiết lập giao diện
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextInput>div>div>input {
        padding: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        background-color: #4CAF50;
        color: white;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive {
        background-color: #ffffff;
        border: 2px solid #4CAF50;
    }
    .negative {
        background-color: #ffffff;
        border: 2px solid #F44336;
    }
    .sentiment-box h3 {
        color: #1a237e;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.1);
    }
    .sentiment-box p {
        margin: 0.8rem 0;
        font-size: 1.1rem;
        line-height: 1.5;
        background: rgba(255,255,255,0.9);
        padding: 0.5rem;
        border-radius: 4px;
    }
    .sentiment-box b {
        color: #1a237e;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model from Hugging Face"""
    model_path = 'phobert_sentiment.pth'
    if not os.path.exists(model_path):
        # Hiển thị trạng thái tải
        with st.spinner('Đang tải model từ Hugging Face...'):
            url = "https://huggingface.co/FoxCodes/GovService_SentimentAI/resolve/main/phobert_sentiment.pth?download=true"
            response = requests.get(url)
            
            # Lưu model
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success('✅ Đã tải xong model!')
    
    # Khởi tạo analyzer với model đã tải
    analyzer = SentimentAnalyzer()
    analyzer.load_model(model_path)
    return analyzer

def analyze_single_text(analyzer, text):
    result = analyzer.predict_phobert(text)
    return result

def analyze_file(analyzer, file):
    data = json.load(file)
    results = []
    
    for text in data.get('positive_feedback', []):
        result = analyze_single_text(analyzer, text)
        results.append({
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'expected': 'Positive'
        })
    
    for text in data.get('negative_feedback', []):
        result = analyze_single_text(analyzer, text)
        results.append({
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'expected': 'Negative'
        })
    
    return pd.DataFrame(results)

def main():
    st.title("🎯 Sentiment Analysis System")
    st.markdown("### Phân tích cảm xúc trong đánh giá dịch vụ công")
    
    # Kiểm tra CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        st.success(f"✅ CUDA available - Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ CUDA not available - Using CPU")
    
    # Load model
    analyzer = load_model()
    
    # Sidebar
    st.sidebar.title("📌 Chức năng")
    analysis_type = st.sidebar.radio(
        "Chọn chế độ phân tích:",
        ["Phân tích văn bản", "Phân tích file JSON"]
    )
    
    if analysis_type == "Phân tích văn bản":
        st.markdown("## 📝 Nhập văn bản cần phân tích")
        text_input = st.text_area("", height=100)
        
        if st.button("Phân tích"):
            if text_input:
                with st.spinner('Đang phân tích...'):
                    result = analyze_single_text(analyzer, text_input)
                
                # Hiển thị kết quả
                sentiment_class = 'positive' if result['sentiment'] == 'Positive' else 'negative'
                st.markdown(f"""
                    <div class='sentiment-box {sentiment_class}'>
                        <h3>Kết quả phân tích:</h3>
                        <p><b>Cảm xúc:</b> {result['sentiment']}</p>
                        <p><b>Độ tin cậy:</b> {result['confidence']:.2%}</p>
                        <p><b>Văn bản sau xử lý:</b> {result['processed_text']}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Vui lòng nhập văn bản cần phân tích!")
    
    else:  # Phân tích file JSON
        st.markdown("## 📁 Tải lên file JSON")
        uploaded_file = st.file_uploader("Chọn file JSON", type=['json'])
        
        if uploaded_file:
            with st.spinner('Đang phân tích...'):
                results_df = analyze_file(analyzer, uploaded_file)
                
                # Hiển thị thống kê
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total = len(results_df)
                    st.metric("Tổng số mẫu", total)
                
                with col2:
                    correct = sum(results_df['sentiment'] == results_df['expected'])
                    accuracy = correct / total
                    st.metric("Số dự đoán đúng", f"{correct} ({accuracy:.2%})")
                
                with col3:
                    avg_confidence = results_df['confidence'].mean()
                    st.metric("Độ tin cậy trung bình", f"{avg_confidence:.2%}")
                
                # Vẽ biểu đồ
                fig = px.bar(
                    results_df.groupby('expected')['sentiment'].value_counts().unstack(),
                    title="Phân phối dự đoán theo nhãn thực tế",
                    labels={'value': 'Số lượng', 'expected': 'Nhãn thực tế'}
                )
                st.plotly_chart(fig)
                
                # Hiển thị bảng kết quả
                st.markdown("### Chi tiết kết quả")
                st.dataframe(results_df)
                
                # Tải xuống kết quả
                csv = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Tải xuống kết quả (CSV)",
                    csv,
                    "results.csv",
                    "text/csv",
                    key='download-csv'
                )

if __name__ == "__main__":
    main() 