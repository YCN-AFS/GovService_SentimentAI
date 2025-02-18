import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Default style settings
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

class ModelVisualizer:
    def __init__(self):
        # Thiết lập màu sắc cho biểu đồ
        self.colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
    def plot_data_distribution(self, df):
        """Plot data distribution"""
        plt.figure(figsize=(12, 5))
        
        # Sentiment distribution
        plt.subplot(121)
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=['Negative', 'Positive'], 
                autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        plt.title('Sentiment Distribution')
        
        # Source distribution
        plt.subplot(122)
        source_counts = df['source'].value_counts()
        sns.barplot(x=source_counts.index, y=source_counts.values)
        plt.title('Distribution by Data Source')
        plt.xlabel('Source')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_review_lengths(self, df):
        """Plot review length distribution"""
        plt.figure(figsize=(12, 5))
        
        # Calculate review lengths
        df['review_length'] = df['processed_text'].str.split().str.len()
        
        # Length distribution
        plt.subplot(121)
        sns.histplot(data=df, x='review_length', bins=50)
        plt.title('Review Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Number of Reviews')
        
        # Length by sentiment
        plt.subplot(122)
        sns.boxplot(data=df, x='sentiment', y='review_length')
        plt.title('Review Length by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Words')
        plt.xticks([0, 1], ['Negative', 'Positive'])
        
        plt.tight_layout()
        plt.savefig('review_lengths.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(121)
        plt.plot(history['train_loss'], label='Training')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(122)
        plt.plot(history['train_acc'], label='Training')
        plt.title('Training Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_comparison(self, results):
        """Plot model comparison"""
        plt.figure(figsize=(12, 5))
        
        # Accuracy comparison
        plt.subplot(121)
        accuracies = [res['accuracy'] for res in results.values()]
        plt.bar(results.keys(), accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # F1-score comparison
        plt.subplot(122)
        f1_scores = [res.get('f1', 0) for res in results.values()]
        plt.bar(results.keys(), f1_scores)
        plt.title('Model F1-Score Comparison')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, results):
        """Plot confusion matrices"""
        n_models = len(results)
        plt.figure(figsize=(15, 5))
        
        for i, (name, res) in enumerate(results.items(), 1):
            plt.subplot(1, n_models, i)
            sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, results):
        """Plot ROC curves"""
        plt.figure(figsize=(8, 6))
        
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(res['y_true'], res['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close() 