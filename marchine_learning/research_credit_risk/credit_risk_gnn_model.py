import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
class FinalDataPreprocessor:
    """最终数据预处理"""

    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, train_path, test_path):
        print("正在加载数据...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        self.train_df.fillna(self.train_df.median(), inplace=True)
        self.test_df.fillna(self.test_df.median(), inplace=True)

        print(f"训练集大小: {len(self.train_df)}")
        print(f"测试集大小: {len(self.test_df)}")

        return self.train_df, self.test_df

    def create_features(self, df):
        """特征工程 - 精选有效特征"""
        df_enhanced = df.copy()

        # 关键衍生特征
        df_enhanced['Audit_Score_Avg'] = (df['Internal_Audit_Score'] + df['External_Audit_Score']) / 2
        df_enhanced['Risk_Ratio'] = df['Loss_score'] / (df['Fin_Score'] + 1)
        df_enhanced['Audit_Consistency'] = abs(df['Internal_Audit_Score'] - df['External_Audit_Score'])
        df_enhanced['Financial_Strength'] = df['Fin_Score'] - df['Loss_score']

        return df_enhanced

    def split_features(self, df, is_train=True):
        df_enhanced = self.create_features(df)

        # 9个特征
        structural_features = [
            'Location_Score', 'Internal_Audit_Score',
            'External_Audit_Score', 'Fin_Score', 'Loss_score',
            'Audit_Score_Avg', 'Risk_Ratio', 'Audit_Consistency', 'Financial_Strength'
        ]

        temporal_features = ['Past_Results']
        other_features = ['City']

        X_structural = df_enhanced[structural_features].values
        X_temporal = df[temporal_features].values
        X_city = df[other_features].values

        if is_train:
            y = df['IsUnderRisk'].values
            return X_structural, X_temporal, X_city, y
        else:
            return X_structural, X_temporal, X_city, None

    def normalize_features(self, X_train_structural, X_test_structural):
        print("正在进行特征标准化...")
        X_train_scaled = self.scaler.fit_transform(X_train_structural)
        X_test_scaled = self.scaler.transform(X_test_structural)
        return X_train_scaled, X_test_scaled

    def create_risk_scores(self, y):
        risk_scores = y.astype(float) + np.random.normal(0, 0.05, size=y.shape)
        risk_scores = np.clip(risk_scores, 0, 1)
        return risk_scores


# 图构建
class OptimalGraphBuilder:
    """优化的图构建"""

    def __init__(self, k_neighbors=6, use_weights=True):
        self.k_neighbors = k_neighbors
        self.use_weights = use_weights

    def build_graph(self, X_structural, X_city):
        print("正在构建图结构...")
        n_nodes = len(X_structural)

        city_edges = self._build_city_edges(X_city)
        knn_edges = self._build_knn_edges(X_structural)

        all_edges = np.vstack([city_edges, knn_edges])
        all_edges = np.unique(all_edges, axis=0)

        if self.use_weights:
            edge_weights = self._compute_edge_weights(X_structural, all_edges)
            adj_matrix = self._create_weighted_adjacency(n_nodes, all_edges, edge_weights)
        else:
            adj_matrix = self._create_adjacency_matrix(n_nodes, all_edges)

        adj_matrix = adj_matrix + np.eye(n_nodes)
        adj_normalized = self._normalize_adjacency(adj_matrix)

        print(f"图节点数: {n_nodes}, 边数: {len(all_edges)}, 平均度: {len(all_edges)*2/n_nodes:.2f}")

        return adj_normalized, all_edges

    def _build_city_edges(self, X_city):
        edges = []
        n_nodes = len(X_city)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if X_city[i, 0] == X_city[j, 0]:
                    edges.append([i, j])
                    edges.append([j, i])
        return np.array(edges) if edges else np.empty((0, 2), dtype=int)

    def _build_knn_edges(self, X_structural):
        distances = cdist(X_structural, X_structural, metric='euclidean')
        edges = []
        n_nodes = len(X_structural)
        for i in range(n_nodes):
            nearest_indices = np.argsort(distances[i])[1:self.k_neighbors+1]
            for j in nearest_indices:
                edges.append([i, j])
        return np.array(edges) if edges else np.empty((0, 2), dtype=int)

    def _compute_edge_weights(self, X_structural, edges):
        if len(edges) == 0:
            return np.array([])
        weights = []
        for edge in edges:
            i, j = edge
            dist = np.linalg.norm(X_structural[i] - X_structural[j])
            weight = np.exp(-dist / 2)
            weights.append(weight)
        return np.array(weights)

    def _create_adjacency_matrix(self, n_nodes, edges):
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for edge in edges:
            i, j = edge
            adj_matrix[i, j] = 1.0
        return adj_matrix

    def _create_weighted_adjacency(self, n_nodes, edges, weights):
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for idx, edge in enumerate(edges):
            i, j = edge
            adj_matrix[i, j] = weights[idx]
        return adj_matrix

    def _normalize_adjacency(self, adj_matrix):
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        degree_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix + 1e-6))
        adj_normalized = degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt
        return adj_normalized


# GNN模型
class ImprovedGraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ImprovedGraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        if self.training:
            adj = F.dropout(adj, self.dropout, training=True)
        output = torch.mm(adj, support)
        output = output + self.bias
        return output


class EnhancedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.35):
        super(EnhancedGNN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(ImprovedGraphConvolutionLayer(input_dim, hidden_dims[0], dropout))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(ImprovedGraphConvolutionLayer(hidden_dims[i], hidden_dims[i+1], dropout))

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in hidden_dims])

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class ImprovedTemporalModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(ImprovedTemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        output = F.relu(output)
        return output


class FinalMultiTaskGNN(nn.Module):
    """最终优化的多任务模型"""

    def __init__(self, structural_dim, temporal_dim, gnn_hidden_dims,
                 lstm_hidden_dim, shared_dim):
        super(FinalMultiTaskGNN, self).__init__()

        self.gnn = EnhancedGNN(structural_dim, gnn_hidden_dims, dropout=0.35)
        self.temporal_module = ImprovedTemporalModule(temporal_dim, lstm_hidden_dim)

        fusion_input_dim = gnn_hidden_dims[-1] + lstm_hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(shared_dim)
        )

        # 分类头 - 增加一层
        self.classification_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(shared_dim // 2, 2)
        )

        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(shared_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_structural, x_temporal, adj):
        gnn_features = self.gnn(x_structural, adj)
        temporal_features = self.temporal_module(x_temporal)
        combined_features = torch.cat([gnn_features, temporal_features], dim=1)
        shared_features = self.fusion_layer(combined_features)

        risk_score = self.regression_head(shared_features)
        classification_logits = self.classification_head(shared_features)

        return risk_score, classification_logits, shared_features


# 数据集、训练器
class CreditRiskDataset(Dataset):
    def __init__(self, X_structural, X_temporal, y_class, y_reg, adj_matrix):
        self.X_structural = torch.FloatTensor(X_structural)
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.y_class = torch.LongTensor(y_class) if y_class is not None else None
        self.y_reg = torch.FloatTensor(y_reg) if y_reg is not None else None
        self.adj_matrix = torch.FloatTensor(adj_matrix)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.y_class is not None and self.y_reg is not None:
            return (self.X_structural, self.X_temporal,
                    self.y_class, self.y_reg, self.adj_matrix)
        else:
            return (self.X_structural, self.X_temporal, self.adj_matrix)


class FinalTrainer:
    def __init__(self, model, device, learning_rate=0.0008,
                 class_weight=0.65, reg_weight=0.35):
        self.model = model.to(device)
        self.device = device
        self.class_weight = class_weight
        self.reg_weight = reg_weight

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=5e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )

        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_class_acc': [],
            'val_class_acc': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        batch = next(iter(train_loader))
        x_structural, x_temporal, y_class, y_reg, adj = batch

        x_structural = x_structural.squeeze(0).to(self.device)
        x_temporal = x_temporal.squeeze(0).to(self.device)
        y_class = y_class.squeeze(0).to(self.device)
        y_reg = y_reg.squeeze(0).to(self.device).unsqueeze(1)
        adj = adj.squeeze(0).to(self.device)

        self.optimizer.zero_grad()
        risk_score, class_logits, _ = self.model(x_structural, x_temporal, adj)

        class_loss = self.classification_criterion(class_logits, y_class)
        reg_loss = self.regression_criterion(risk_score, y_reg)
        total_loss = self.class_weight * class_loss + self.reg_weight * reg_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        preds = torch.argmax(class_logits, dim=1)
        accuracy = accuracy_score(y_class.cpu().numpy(), preds.cpu().numpy())

        return total_loss.item(), accuracy

    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            x_structural, x_temporal, y_class, y_reg, adj = batch

            x_structural = x_structural.squeeze(0).to(self.device)
            x_temporal = x_temporal.squeeze(0).to(self.device)
            y_class = y_class.squeeze(0).to(self.device)
            y_reg = y_reg.squeeze(0).to(self.device).unsqueeze(1)
            adj = adj.squeeze(0).to(self.device)

            risk_score, class_logits, _ = self.model(x_structural, x_temporal, adj)

            class_loss = self.classification_criterion(class_logits, y_class)
            reg_loss = self.regression_criterion(risk_score, y_reg)
            total_loss = self.class_weight * class_loss + self.reg_weight * reg_loss

            preds = torch.argmax(class_logits, dim=1)
            accuracy = accuracy_score(y_class.cpu().numpy(), preds.cpu().numpy())

        return total_loss.item(), accuracy

    def train(self, train_loader, val_loader, epochs):
        print("\n开始训练最终优化模型...")
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        patience = 12

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_class_acc'].append(train_acc)
            self.history['val_class_acc'].append(val_acc)

            self.scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 同时考虑loss和accuracy
            if val_loss < best_val_loss or val_acc > best_val_acc:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model_final.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(torch.load('best_model_final.pth'))
        print("训练完成！")


# 评估和可视化类
class ComprehensiveEvaluator:
    """完整的评估器，包含所有可视化"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def evaluate(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(data_loader))
            x_structural, x_temporal, y_class, y_reg, adj = batch

            x_structural = x_structural.squeeze(0).to(self.device)
            x_temporal = x_temporal.squeeze(0).to(self.device)
            adj = adj.squeeze(0).to(self.device)

            risk_score, class_logits, _ = self.model(x_structural, x_temporal, adj)

            class_preds = torch.argmax(class_logits, dim=1).cpu().numpy()
            class_labels = y_class.squeeze(0).numpy()

            reg_preds = risk_score.cpu().numpy().flatten()
            reg_labels = y_reg.squeeze(0).numpy()

        accuracy = accuracy_score(class_labels, class_preds)
        precision = precision_score(class_labels, class_preds, average='binary')
        recall = recall_score(class_labels, class_preds, average='binary')
        f1 = f1_score(class_labels, class_preds, average='binary')

        mse = mean_squared_error(reg_labels, reg_preds)
        r2 = r2_score(reg_labels, reg_preds)

        cm = confusion_matrix(class_labels, class_preds)

        results = {
            'classification': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': class_preds,
                'labels': class_labels
            },
            'regression': {
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'predictions': reg_preds,
                'labels': reg_labels
            }
        }

        return results

    def print_results(self, results):
        print("\n" + "="*60)
        print("最终优化模型评估结果")
        print("="*60)

        print("\n分类任务 (IsUnderRisk 预测):")
        print(f"  准确率 (Accuracy):  {results['classification']['accuracy']:.4f}")
        print(f"  精确率 (Precision): {results['classification']['precision']:.4f}")
        print(f"  召回率 (Recall):    {results['classification']['recall']:.4f}")
        print(f"  F1分数 (F1-Score):  {results['classification']['f1_score']:.4f}")

        print("\n回归任务 (Risk Score 预测):")
        print(f"  均方误差 (MSE):     {results['regression']['mse']:.4f}")
        print(f"  均方根误差 (RMSE):  {results['regression']['rmse']:.4f}")
        print(f"  R²分数:             {results['regression']['r2']:.4f}")
        print("="*60 + "\n")

    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - GNN Model', fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
        plt.close()

    def plot_training_history(self, history, save_path='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#3498db')
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(history['train_class_acc'], label='Train Accuracy', linewidth=2, color='#2ecc71')
        axes[1].plot(history['val_class_acc'], label='Validation Accuracy', linewidth=2, color='#f39c12')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存: {save_path}")
        plt.close()

    def plot_model_comparison(self, gnn_results, baseline_results, save_path='model_comparison.png'):
        """绘制模型对比图（不含SVM）"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 准备数据
        models = ['Original\nGNN', 'Logistic\nRegression', 'Final\nGNN']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        original_scores = [0.7706, 0.8923, 0.7632, 0.8227]
        lr_scores = [baseline_results['LR'][m] for m in metrics]
        gnn_scores = [gnn_results['classification'][m] for m in metrics]

        x = np.arange(len(metric_names))
        width = 0.25

        # 柱状图
        axes[0].bar(x - width, original_scores, width, label='Original GNN', color='#95a5a6', alpha=0.8)
        axes[0].bar(x, lr_scores, width, label='Logistic Regression', color='#3498db', alpha=0.8)
        axes[0].bar(x + width, gnn_scores, width, label='Final GNN', color='#2ecc71', alpha=0.8)

        axes[0].set_xlabel('Metrics', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metric_names)
        axes[0].legend(fontsize=10, loc='lower right')
        axes[0].set_ylim([0.6, 1.0])
        axes[0].grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bars in axes[0].containers:
            axes[0].bar_label(bars, fmt='%.3f', fontsize=8, padding=3)

        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        original_scores_radar = original_scores + [original_scores[0]]
        lr_scores_radar = lr_scores + [lr_scores[0]]
        gnn_scores_radar = gnn_scores + [gnn_scores[0]]
        angles += angles[:1]

        ax = plt.subplot(1, 2, 2, projection='polar')
        ax.plot(angles, original_scores_radar, 'o-', linewidth=2, label='Original GNN', color='#95a5a6')
        ax.fill(angles, original_scores_radar, alpha=0.15, color='#95a5a6')
        ax.plot(angles, lr_scores_radar, 'o-', linewidth=2, label='Logistic Regression', color='#3498db')
        ax.fill(angles, lr_scores_radar, alpha=0.15, color='#3498db')
        ax.plot(angles, gnn_scores_radar, 'o-', linewidth=2, label='Final GNN', color='#2ecc71')
        ax.fill(angles, gnn_scores_radar, alpha=0.2, color='#2ecc71')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0.6, 1.0)
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存: {save_path}")
        plt.close()

    def plot_risk_distribution(self, predictions, labels, save_path='risk_distribution.png'):
        """绘制风险分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 预测风险分数分布
        axes[0].hist(predictions[labels == 0], bins=30, alpha=0.6, label='Low Risk (True)', color='#2ecc71', edgecolor='black')
        axes[0].hist(predictions[labels == 1], bins=30, alpha=0.6, label='High Risk (True)', color='#e74c3c', edgecolor='black')
        axes[0].set_xlabel('Predicted Risk Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Risk Score Distribution by True Label', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)

        # 预测类别分布
        pred_classes = (predictions > 0.5).astype(int)
        unique, counts = np.unique(pred_classes, return_counts=True)
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(counts, labels=['Low Risk', 'High Risk'], autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title('Predicted Risk Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"风险分布图已保存: {save_path}")
        plt.close()

    def create_results_table(self, gnn_results, baseline_results, save_path='results_table.png'):
        """创建结果对比表"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')

        # 表格数据
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        original_gnn = [0.7706, 0.8923, 0.7632, 0.8227]
        lr_scores = [baseline_results['LR'][m.lower().replace('-', '_')] for m in metrics]
        final_gnn = [gnn_results['classification'][m.lower().replace('-', '_')] for m in metrics]

        table_data = []
        for i, metric in enumerate(metrics):
            row = [metric,
                   f"{original_gnn[i]:.4f}",
                   f"{lr_scores[i]:.4f}",
                   f"{final_gnn[i]:.4f}"]
            table_data.append(row)

        # 添加提升幅度
        improvements = [f"+{(final_gnn[i] - original_gnn[i])*100:.2f}%" for i in range(len(metrics))]
        table_data.append(['Improvement vs Original'] + [''] * 2 + [improvements[0]])

        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Original GNN', 'Logistic Reg.', 'Final GNN'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # 设置表头样式
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 设置最后一列（Final GNN）高亮
        for i in range(1, len(table_data)):
            table[(i, 3)].set_facecolor('#d5f4e6')
            table[(i, 3)].set_text_props(weight='bold')

        # 标题
        plt.title('Model Performance Comparison Table',
                 fontsize=16, fontweight='bold', pad=20)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果对比表已保存: {save_path}")
        plt.close()

    def plot_risk_propagation_path(self, edges, risk_scores, predicted_classes,
                                   X_structural, save_path='risk_propagation.png'):
        """绘制风险传导路径图"""
        # 创建图
        G = nx.Graph()

        # 添加节点
        n_nodes = len(risk_scores)
        for i in range(n_nodes):
            G.add_node(i)

        # 添加边（限制边数，避免图太密）
        for edge in edges[:min(len(edges), n_nodes * 3)]:
            G.add_edge(int(edge[0]), int(edge[1]))

        # 选择高风险节点和它们的邻居进行重点展示
        high_risk_nodes = np.where(predicted_classes == 1)[0]

        # 如果高风险节点太多，只选择前20个
        if len(high_risk_nodes) > 20:
            # 选择风险分数最高的20个
            high_risk_indices = np.argsort(risk_scores)[-20:]
            high_risk_nodes = high_risk_indices

        # 创建子图：包含高风险节点及其邻居
        subgraph_nodes = set(high_risk_nodes)
        for node in high_risk_nodes:
            if node in G:
                neighbors = list(G.neighbors(node))
                subgraph_nodes.update(neighbors[:5])  # 每个高风险节点最多5个邻居

        # 子图太小，添加一些额外节点
        if len(subgraph_nodes) < 30:
            additional_nodes = set(range(min(30, n_nodes))) - subgraph_nodes
            subgraph_nodes.update(list(additional_nodes)[:30-len(subgraph_nodes)])

        # 创建子图
        subgraph_nodes = list(subgraph_nodes)
        G_sub = G.subgraph(subgraph_nodes).copy()

        # 绘制
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        # === 左图：风险传导网络图 ===
        ax1 = axes[0]

        # 使用spring布局
        pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)

        # 节点颜色：根据风险分数
        node_colors = [risk_scores[node] for node in G_sub.nodes()]

        # 节点大小：根据度中心性
        node_sizes = []
        for node in G_sub.nodes():
            degree = G_sub.degree(node)
            node_sizes.append(300 + degree * 50)

        # 绘制边
        nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=1, edge_color='gray', ax=ax1)

        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            G_sub, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap='RdYlGn_r',
            vmin=0, vmax=1,
            alpha=0.8,
            edgecolors='black',
            linewidths=1.5,
            ax=ax1
        )

        # 高亮高风险节点
        high_risk_in_subgraph = [n for n in G_sub.nodes() if n in high_risk_nodes]
        if high_risk_in_subgraph:
            nx.draw_networkx_nodes(
                G_sub, pos,
                nodelist=high_risk_in_subgraph,
                node_color='red',
                node_size=[300 + G_sub.degree(n) * 50 for n in high_risk_in_subgraph],
                alpha=0.9,
                edgecolors='darkred',
                linewidths=3,
                ax=ax1
            )

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Risk Score', fontsize=12, fontweight='bold')

        ax1.set_title('Risk Propagation Network', fontsize=14, fontweight='bold', pad=15)
        ax1.axis('off')

        # === 右图：风险传导路径示例 ===
        ax2 = axes[1]

        # 找出一条典型的风险传导路径（从高风险节点开始的最短路径）
        if len(high_risk_in_subgraph) >= 2:
            # 选择两个高风险节点
            source = high_risk_in_subgraph[0]
            target = high_risk_in_subgraph[-1]

            try:
                # 找最短路径
                path = nx.shortest_path(G_sub, source=source, target=target)

                # 创建路径的子图
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]

                # 使用层次布局
                pos2 = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)

                # 绘制所有边（浅灰色）
                nx.draw_networkx_edges(G_sub, pos2, alpha=0.1, width=1, edge_color='lightgray', ax=ax2)

                # 绘制所有节点（小的灰色）
                nx.draw_networkx_nodes(
                    G_sub, pos2,
                    node_color='lightgray',
                    node_size=200,
                    alpha=0.3,
                    ax=ax2
                )

                # 高亮路径上的边
                nx.draw_networkx_edges(
                    G_sub, pos2,
                    edgelist=path_edges,
                    edge_color='red',
                    width=3,
                    alpha=0.8,
                    arrows=True,
                    arrowsize=20,
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.1',
                    ax=ax2
                )

                # 高亮路径上的节点
                path_node_colors = [risk_scores[node] for node in path]
                nx.draw_networkx_nodes(
                    G_sub, pos2,
                    nodelist=path,
                    node_color=path_node_colors,
                    node_size=600,
                    cmap='RdYlGn_r',
                    vmin=0, vmax=1,
                    alpha=0.9,
                    edgecolors='darkred',
                    linewidths=2,
                    ax=ax2
                )

                # 添加节点标签（显示风险分数）
                labels = {node: f"{risk_scores[node]:.2f}" for node in path}
                nx.draw_networkx_labels(
                    G_sub, pos2,
                    labels=labels,
                    font_size=10,
                    font_weight='bold',
                    font_color='white',
                    ax=ax2
                )

                # 添加路径说明
                path_info = f"Path Length: {len(path)} nodes\n"
                path_info += f"Start Risk: {risk_scores[source]:.3f}\n"
                path_info += f"End Risk: {risk_scores[target]:.3f}\n"
                path_info += f"Avg Path Risk: {np.mean([risk_scores[n] for n in path]):.3f}"

                ax2.text(0.02, 0.98, path_info,
                        transform=ax2.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                ax2.set_title('Example Risk Propagation Path', fontsize=14, fontweight='bold', pad=15)

            except nx.NetworkXNoPath:
                # 如果没有路径，显示单个高风险节点的邻居
                source = high_risk_in_subgraph[0]
                neighbors = list(G_sub.neighbors(source))[:8]
                subgraph_nodes_small = [source] + neighbors
                G_small = G_sub.subgraph(subgraph_nodes_small)

                pos2 = nx.spring_layout(G_small, k=3, iterations=50, seed=42)

                # 绘制
                node_colors_small = [risk_scores[node] for node in G_small.nodes()]
                nx.draw_networkx_edges(G_small, pos2, alpha=0.3, width=2, edge_color='gray', ax=ax2)
                nx.draw_networkx_nodes(
                    G_small, pos2,
                    node_color=node_colors_small,
                    node_size=600,
                    cmap='RdYlGn_r',
                    vmin=0, vmax=1,
                    alpha=0.9,
                    edgecolors='black',
                    linewidths=2,
                    ax=ax2
                )

                # 标注
                labels = {node: f"{risk_scores[node]:.2f}" for node in G_small.nodes()}
                nx.draw_networkx_labels(G_small, pos2, labels=labels, font_size=10,
                                       font_weight='bold', font_color='white', ax=ax2)

                ax2.set_title('High-Risk Node and Neighbors', fontsize=14, fontweight='bold', pad=15)
        else:
            ax2.text(0.5, 0.5, 'Insufficient high-risk nodes\nfor path visualization',
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_title('Risk Propagation Path', fontsize=14, fontweight='bold', pad=15)

        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"风险传导路径图已保存: {save_path}")
        plt.close()


# 主函数

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 步骤1: 数据预处理
    print("\n数据预处理")
    print("-" * 70)

    preprocessor = FinalDataPreprocessor()
    train_df, test_df = preprocessor.load_data('Train.csv', 'Test.csv')

    X_train_structural, X_train_temporal, X_train_city, y_train = \
        preprocessor.split_features(train_df, is_train=True)

    X_test_structural, X_test_temporal, X_test_city, _ = \
        preprocessor.split_features(test_df, is_train=False)

    X_train_structural_scaled, X_test_structural_scaled = \
        preprocessor.normalize_features(X_train_structural, X_test_structural)

    y_train_risk_scores = preprocessor.create_risk_scores(y_train)

    n_train = int(len(X_train_structural_scaled) * 0.8)
    indices = np.random.permutation(len(X_train_structural_scaled))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train_s = X_train_structural_scaled[train_indices]
    X_val_s = X_train_structural_scaled[val_indices]
    X_train_t = X_train_temporal[train_indices]
    X_val_t = X_train_temporal[val_indices]
    X_train_c = X_train_city[train_indices]
    X_val_c = X_train_city[val_indices]
    y_train_class = y_train[train_indices]
    y_val_class = y_train[val_indices]
    y_train_reg = y_train_risk_scores[train_indices]
    y_val_reg = y_train_risk_scores[val_indices]

    print(f"特征数量: {X_train_s.shape[1]} (原始5个 + 衍生4个)")

    # 步骤2: 图构建
    print("\n图结构构建")
    print("-" * 70)

    graph_builder = OptimalGraphBuilder(k_neighbors=6, use_weights=True)
    adj_train, edges_train = graph_builder.build_graph(X_train_s, X_train_c)
    adj_val, edges_val = graph_builder.build_graph(X_val_s, X_val_c)
    adj_test, edges_test = graph_builder.build_graph(X_test_structural_scaled, X_test_city)

    # 步骤3: 创建数据集
    print("\n创建数据加载器")
    print("-" * 70)

    train_dataset = CreditRiskDataset(X_train_s, X_train_t, y_train_class, y_train_reg, adj_train)
    val_dataset = CreditRiskDataset(X_val_s, X_val_t, y_val_class, y_val_reg, adj_val)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"训练集节点数: {len(X_train_s)}")
    print(f"验证集节点数: {len(X_val_s)}")

    # 步骤4: 构建模型
    print("\n构建最终优化GNN模型")
    print("-" * 70)

    model = FinalMultiTaskGNN(
        structural_dim=X_train_s.shape[1],
        temporal_dim=X_train_t.shape[1],
        gnn_hidden_dims=[128, 192, 128],  # 稍大一点
        lstm_hidden_dim=64,
        shared_dim=128
    )

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

    # 步骤5: 训练
    print("\n模型训练")
    print("-" * 70)

    trainer = FinalTrainer(
        model=model,
        device=device,
        learning_rate=0.0008,
        class_weight=0.65,
        reg_weight=0.35
    )

    trainer.train(train_loader, val_loader, epochs=120)

    # 步骤6: 评估
    print("\n模型评估")
    print("-" * 70)

    evaluator = ComprehensiveEvaluator(model, device)
    gnn_results = evaluator.evaluate(val_loader)
    evaluator.print_results(gnn_results)

    # 步骤7: 基准模型对比
    print("\n基准模型对比")
    print("-" * 70)

    X_train_baseline = np.hstack([X_train_s, X_train_t])
    X_val_baseline = np.hstack([X_val_s, X_val_t])

    # 只对比Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_baseline, y_train_class)
    lr_pred = lr.predict(X_val_baseline)

    baseline_results = {
        'LR': {
            'accuracy': accuracy_score(y_val_class, lr_pred),
            'precision': precision_score(y_val_class, lr_pred, average='binary'),
            'recall': recall_score(y_val_class, lr_pred, average='binary'),
            'f1_score': f1_score(y_val_class, lr_pred, average='binary')
        }
    }

    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)

    print(f"\n原始GNN模型:")
    print(f"  准确率:  77.06%")
    print(f"  F1分数:  82.27%")

    print(f"\nLogistic Regression:")
    print(f"  准确率:  {baseline_results['LR']['accuracy']:.4f}")
    print(f"  F1分数:  {baseline_results['LR']['f1_score']:.4f}")

    print(f"\n最终优化GNN:")
    print(f"  准确率:  {gnn_results['classification']['accuracy']:.4f}")
    print(f"  F1分数:  {gnn_results['classification']['f1_score']:.4f}")

    # 判断提升
    if gnn_results['classification']['accuracy'] > 0.77:
        improvement = (gnn_results['classification']['accuracy'] - 0.77) * 100
        print(f"\n✅ 相比原始GNN提升了 {improvement:.2f}%!")

    if gnn_results['classification']['accuracy'] > baseline_results['LR']['accuracy']:
        print(f"🎉 已超越Logistic Regression基准!")

    print("="*60)

    # 步骤8: 生成所有可视化
    print("\n生成可视化图表")
    print("-" * 70)

    evaluator.plot_training_history(trainer.history)
    evaluator.plot_confusion_matrix(gnn_results['classification']['confusion_matrix'])
    evaluator.plot_model_comparison(gnn_results, baseline_results)
    evaluator.plot_risk_distribution(
        gnn_results['regression']['predictions'],
        gnn_results['classification']['labels']
    )
    evaluator.create_results_table(gnn_results, baseline_results)

    # 绘制风险传导路径图
    evaluator.plot_risk_propagation_path(
        edges_val,
        gnn_results['regression']['predictions'],
        gnn_results['classification']['predictions'],
        X_val_s
    )

    # 步骤9: 测试集预测
    print("\n测试集预测")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        X_test_s_tensor = torch.FloatTensor(X_test_structural_scaled).to(device)
        X_test_t_tensor = torch.FloatTensor(X_test_temporal).to(device)
        adj_test_tensor = torch.FloatTensor(adj_test).to(device)

        risk_scores, class_logits, _ = model(X_test_s_tensor, X_test_t_tensor, adj_test_tensor)

        predicted_classes = torch.argmax(class_logits, dim=1).cpu().numpy()
        predicted_risk_scores = risk_scores.cpu().numpy().flatten()

    predictions_df = pd.DataFrame({
        'Predicted_IsUnderRisk': predicted_classes,
        'Predicted_Risk_Score': predicted_risk_scores
    })

    predictions_df.to_csv('predictions_final.csv', index=False)
    print("测试集预测结果已保存: predictions_final.csv")
    print(f"预测为高风险的样本数: {(predicted_classes == 1).sum()}")
    print(f"预测为低风险的样本数: {(predicted_classes == 0).sum()}")


if __name__ == "__main__":
    main()