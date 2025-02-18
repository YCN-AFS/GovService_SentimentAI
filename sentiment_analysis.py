import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyvi import ViTokenizer, ViPosTagger
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
from tqdm import tqdm  # Thêm thanh tiến trình
from visualization import ModelVisualizer
import os
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.models = {
            'naive_bayes': MultinomialNB(),
            'svm': LinearSVC(random_state=42)
        }
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert_model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base", 
            num_labels=2
        )
        
    def load_data(self, excel_file):
        """Load and process data from Excel file with multiple sheets"""
        try:
            all_data = []
            xlsx = pd.ExcelFile(excel_file)
            
            expected_columns = [
                'reviewId', 'userName', 'userImage', 'content', 'score',
                'thumbsUpCount', 'reviewCreatedVersion', 'at', 
                'replyContent', 'repliedAt', 'appVersion'
            ]
            
            for sheet in xlsx.sheet_names:
                # Đọc từng sheet
                df = pd.read_excel(excel_file, sheet_name=sheet)
                
                # Kiểm tra và chuẩn hóa tên cột
                if not all(col in df.columns for col in expected_columns):
                    print(f"Warning: Sheet '{sheet}' missing some expected columns")
                    continue
                    
                # Thêm thông tin nguồn dữ liệu
                df['source'] = sheet
                
                # Đổi tên cột cho phù hợp với preprocessing
                df = df.rename(columns={
                    'content': 'Review',
                    'score': 'Rating'
                })
                
                # Chỉ giữ lại các dòng có nội dung review và rating
                df = df.dropna(subset=['Review', 'Rating'])
                
                # Thêm vào danh sách
                all_data.append(df)
                
                print(f"Loaded {len(df)} reviews from {sheet}")
                
            # Gộp tất cả dữ liệu
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal reviews loaded: {len(final_df)}")
            
            return final_df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def load_stopwords(self, filepath='vietnamese_stopwords.txt'):
        """Load Vietnamese stopwords from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: Stopwords file not found at {filepath}")
            return set()
    
    def preprocess_text(self, text):
        """Preprocess Vietnamese text data"""
        if not isinstance(text, str):
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove URLs and emails
        text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize and normalize Vietnamese text
        text = ViTokenizer.tokenize(text)
        tokens = text.split()
        
        # Remove stopwords
        vn_stop_words = self.load_stopwords()
        tokens = [t for t in tokens if t not in vn_stop_words]
        
        return ' '.join(tokens)
    
    def assign_sentiment_labels(self, rating):
        """Convert ratings to sentiment labels"""
        try:
            rating = float(rating)
            if rating <= 2:
                return 0  # Negative
            elif rating >= 4:
                return 1  # Positive
            else:
                return None  # Neutral/Ignore
        except:
            return None
            
    def prepare_data(self, df):
        """Prepare data for training"""
        # Preprocess reviews
        df['processed_text'] = df['Review'].apply(self.preprocess_text)
        
        # Assign sentiment labels
        df['sentiment'] = df['Rating'].apply(self.assign_sentiment_labels)
        
        # Remove neutral reviews and empty texts
        df = df.dropna(subset=['sentiment', 'processed_text'])
        df = df[df['processed_text'].str.strip() != '']
        
        return df
    
    def train_traditional_models(self, X_train, y_train):
        """Train traditional ML models"""
        # Transform text data to TF-IDF features
        print("Đang tạo TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train each model
        for name, model in self.models.items():
            print(f"Đang huấn luyện mô hình {name}...")
            model.fit(X_train_tfidf, y_train)
        
        print("Hoàn thành huấn luyện mô hình truyền thống!")
    
    def evaluate_traditional_models(self, X_test, y_test):
        """Evaluate traditional ML models"""
        print("Đang chuyển đổi dữ liệu test thành TF-IDF features...")
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        results = {}
        for name, model in self.models.items():
            print(f"\nĐang đánh giá mô hình {name}...")
            y_pred = model.predict(X_test_tfidf)
            try:
                # Lấy xác suất dự đoán nếu có
                y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
            except:
                # Nếu không có predict_proba (như LinearSVC), dùng decision_function
                y_pred_proba = model.decision_function(X_test_tfidf)
                # Chuẩn hóa về khoảng [0, 1]
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            
            results[name] = {
                'model_name': name,
                'accuracy': accuracy,
                'f1': f1,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'report': report,
                'confusion_matrix': conf_matrix,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Kết quả đánh giá cho {name}:")
            print(f"Độ chính xác: {accuracy:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"Độ đặc hiệu: {specificity:.4f}")
            print(f"Độ nhạy: {sensitivity:.4f}")
            print("\nBáo cáo phân loại chi tiết:")
            print(report)
            print("\nMa trận nhầm lẫn:")
            print(conf_matrix)
        
        return results
    
    def train_phobert(self, train_texts, train_labels, batch_size=32, epochs=3):
        """Train PhoBERT model"""
        from tqdm import tqdm
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng thiết bị: {device}")
        
        print("Chuẩn bị dữ liệu...")
        train_encodings = self.phobert_tokenizer(
            train_texts.tolist(),
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Chuyển labels thành tensor
        labels = torch.tensor(train_labels.tolist(), dtype=torch.long)
        
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            labels
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.phobert_model.to(device)
        optimizer = torch.optim.AdamW(self.phobert_model.parameters(), lr=5e-5)
        
        history = {
            'train_loss': [],
            'train_acc': []
        }
        
        for epoch in range(epochs):
            self.phobert_model.train()
            total_loss = 0
            correct = 0
            total = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = self.phobert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                # Tính accuracy
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)
            
            print(f"\nEpoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return history

    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model with detailed metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Tính toán thêm các metric chi tiết
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'report': report,
            'confusion_matrix': conf_matrix,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        print(f"\nKết quả đánh giá cho {model_name}:")
        print(f"Độ chính xác: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Độ đặc hiệu: {specificity:.4f}")
        print(f"Độ nhạy: {sensitivity:.4f}")
        print("\nBáo cáo phân loại chi tiết:")
        print(report)
        print("\nMa trận nhầm lẫn:")
        print(conf_matrix)
        
        return results

    def predict_phobert(self, text, device='cuda'):
        """Dự đoán cảm xúc cho một văn bản"""
        # Đảm bảo mô hình ở đúng thiết bị
        self.phobert_model = self.phobert_model.to(device)
        
        # Tiền xử lý văn bản
        processed_text = self.preprocess_text(text)
        
        # Chuyển văn bản thành tensor
        inputs = self.phobert_tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Chuyển tất cả tensor sang cùng thiết bị
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Dự đoán
        self.phobert_model.eval()
        with torch.no_grad():
            outputs = self.phobert_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
        
        # Chuyển kết quả về CPU để xử lý
        sentiment = 'Positive' if predicted_class.cpu().item() == 1 else 'Negative'
        confidence = predictions.cpu()[0][predicted_class.cpu().item()].item()
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'processed_text': processed_text
        }

    def predict_batch(self, texts, device='cuda', batch_size=32):
        """Dự đoán cảm xúc cho nhiều văn bản"""
        # Đảm bảo mô hình ở đúng thiết bị
        self.phobert_model = self.phobert_model.to(device)
        self.phobert_model.eval()
        
        # Tiền xử lý văn bản
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Khởi tạo mảng kết quả
        all_predictions = []
        all_probabilities = []
        
        # Xử lý theo batch
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.phobert_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            
            # Chuyển sang device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.phobert_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                
                # Chuyển về CPU và numpy
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy()[:, 1])
            
            # Giải phóng bộ nhớ
            del input_ids, attention_mask, outputs, logits, probabilities, predicted_classes
            torch.cuda.empty_cache()
            
            # Hiển thị tiến trình
            print(f"\rĐã xử lý {min(i + batch_size, len(texts))}/{len(texts)} mẫu", end="")
        
        print()  # Xuống dòng sau khi hoàn thành
        
        return np.array(all_predictions), np.array(all_probabilities)

    def save_model(self, path='phobert_sentiment.pth'):
        """Lưu mô hình đã huấn luyện"""
        torch.save(self.phobert_model.state_dict(), path)
        print(f"Đã lưu mô hình tại: {path}")

    def load_model(self, path='phobert_sentiment.pth'):
        """Tải mô hình đã huấn luyện"""
        try:
            # Tải state dict
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            
            # Khởi tạo classifier layers nếu chưa có
            self.phobert_model.classifier = torch.nn.Sequential(
                torch.nn.Linear(768, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(768, 2)
            )
            
            # Load state dict
            self.phobert_model.load_state_dict(state_dict)
            self.phobert_model.eval()
            print(f"Đã tải mô hình từ: {path}")
            return True
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {str(e)}")
            return False

def main():
    # Thiết lập cấu hình CUDA để tránh phân mảnh bộ nhớ
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Khởi tạo analyzer
    analyzer = SentimentAnalyzer()
    
    # Load và chuẩn bị dữ liệu
    print("\nĐang tải dữ liệu...")
    df_play = analyzer.load_data('google_play_reviews_all.xlsx')
    prepared_df = analyzer.prepare_data(df_play)
    
    # Chia dữ liệu
    print("\nĐang chia dữ liệu train/test...")
    X = prepared_df['processed_text']
    y = prepared_df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Khởi tạo visualizer
    visualizer = ModelVisualizer()
    
    # Vẽ biểu đồ phân bố dữ liệu
    print("\nĐang vẽ biểu đồ phân tích dữ liệu...")
    visualizer.plot_data_distribution(prepared_df)
    visualizer.plot_review_lengths(prepared_df)
    
    # Huấn luyện và đánh giá mô hình truyền thống
    print("\nĐang huấn luyện mô hình truyền thống...")
    analyzer.train_traditional_models(X_train, y_train)
    traditional_results = analyzer.evaluate_traditional_models(X_test, y_test)
    
    # Huấn luyện PhoBERT
    print("\nĐang huấn luyện mô hình PhoBERT...")
    history = analyzer.train_phobert(X_train, y_train, epochs=3)
    
    # Vẽ biểu đồ quá trình huấn luyện
    print("\nĐang vẽ biểu đồ kết quả huấn luyện...")
    visualizer.plot_training_history(history)
    
    # Đánh giá PhoBERT
    print("\nĐang đánh giá mô hình PhoBERT...")
    phobert_predictions, phobert_probabilities = analyzer.predict_batch(X_test)
    phobert_results = analyzer.evaluate_model(y_test, phobert_predictions, 'PhoBERT')
    phobert_results['y_pred_proba'] = phobert_probabilities
    
    # Tổng hợp kết quả đánh giá
    results = {
        'PhoBERT': phobert_results,
        'SVM': traditional_results['svm'],
        'Naive Bayes': traditional_results['naive_bayes']
    }
    
    # Vẽ biểu đồ so sánh và đánh giá
    print("\nĐang vẽ biểu đồ so sánh các mô hình...")
    visualizer.plot_model_comparison(results)
    visualizer.plot_confusion_matrices(results)
    visualizer.plot_roc_curves(results)
    
    # Lưu mô hình
    analyzer.save_model()
    print("\nĐã hoàn thành huấn luyện và đánh giá các mô hình!")

if __name__ == "__main__":
    main() 