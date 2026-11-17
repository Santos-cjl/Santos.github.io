import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# 导入MinMaxScaler和LabelEncoder，用于数据预处理
import tensorflow as tf  # 导入tensorflow库，用于构建神经网络模型
from sklearn.metrics import (precision_score, recall_score,  # 导入评估指标函数
                             f1_score, accuracy_score, roc_auc_score)
# 精确率、召回率、F1分数、准确率和ROC曲线下面积
import matplotlib.pyplot as plt
import warnings  # 导入warnings库，用于处理警告信息

warnings.filterwarnings('ignore')  # 忽略警告信息

# 加载和预处理数据集
data = pd.read_csv('../../data/adult.csv')

# 识别分类列和目标变量
dis_cols = data.select_dtypes(include=["object"]).columns.tolist()  # 获取所有分类列
label_col = 'income'  # 目标变量列名为'income'

# 使用标签编码器将分类变量转换为数值
for col in dis_cols:
    data[col] = LabelEncoder().fit_transform(data[col])  # 对每个分类列进行标签编码

# 分割特征和目标变量
x = data.drop(columns=[label_col])  # 特征集，包含除目标变量外的所有列
y = data[label_col]  # 目标变量

# 预处理特征，使用Min-Max缩放
x = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)  # 将特征缩放到0-1范围

# 分割数据为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 构建神经网络模型
model = tf.keras.Sequential()  # 初始化一个顺序模型

# 添加第一个隐藏层，包含80个神经元，使用ReLU激活函数
model.add(tf.keras.layers.Dense(80, input_dim=x.shape[1],activation='relu'))

# 添加第二个隐藏层，包含20个神经元，使用ReLU激活函数
model.add(tf.keras.layers.Dense(20, activation='relu'))

# 添加输出层，包含1个神经元，使用sigmoid激活函数（二分类问题）
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# 编译模型，使用Adam优化器和二元交叉熵损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

model.summary()  # 显示模型架构摘要

# 定义训练回调函数
early_stop = tf.keras.callbacks.EarlyStopping(  # 早停回调函数
    monitor='val_loss', patience=10, restore_best_weights=True)
# 监控验证损失，如果连续10个epoch没有改善则停止训练，并恢复到最佳权重

reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(  # 学习率减少回调函数
    monitor='val_loss', factor=0.1, patience=5)
# 监控验证损失，如果损失没有改善，则将学习率减少为原来的0.1倍

# 训练模型
history = model.fit(x_train, y_train, epochs=100, batch_size=150, validation_split=0.2,
                    verbose=1, callbacks=[early_stop, reduce_on_plateau])
# 使用早停和学习率减少回调函数

# 评估模型
y_pred_prob = model.predict(x_test)  # 对测试集进行预测，得到概率值
y_pred = (y_pred_prob > 0.5).astype(int)  # 将概率值转换为二分类结果（阈值为0.5）

# 计算评估指标
print("Accuracy:", accuracy_score(y_test, y_pred))  # 计算准确率
print("Precision:", precision_score(y_test, y_pred))  # 计算精确率
print("Recall:", recall_score(y_test, y_pred))  # 计算召回率
print("F1 Score:", f1_score(y_test, y_pred))  # 计算F1分数
print("AUC:", roc_auc_score(y_test, y_pred_prob))  # 计算ROC曲线下面积

# 绘制训练和验证的准确率与损失曲线
plt.rcParams['font.size'] = 15  # 设置全局字体大小为15
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置全局字体为Times New Roman

plt.figure(figsize=(12, 6))  # 创建一个12x6英寸的图形

# 创建两个子图：一个用于准确率，一个用于损失
plt.subplot(1, 2, 1)  # 第一个子图（准确率）
plt.plot(history.history['accuracy'], label='Train Accuracy')  # 绘制训练准确率曲线
plt.plot(history.history['val_accuracy'], label='Val Accuracy')  # 绘制验证准确率曲线
plt.grid()  # 添加网格线
plt.legend(loc='best')  # 将图例放置在最佳位置

plt.subplot(1, 2, 2)  # 第二个子图（损失）
plt.plot(history.history['loss'], label='Train Loss')  # 绘制训练损失曲线
plt.plot(history.history['val_loss'], label='Val Loss')  # 绘制验证损失曲线
plt.grid()  # 添加网格线
plt.legend(loc='best')  # 将图例放置在最佳位置

plt.show()  # 显示图形
