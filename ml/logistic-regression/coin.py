import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 生成模拟数据
# 正面用1表示，反面用0表示
np.random.seed(42)
num_samples = 10000  # 增加数据集规模
bias = 0.7  # 正面出现的概率
coin_flips = np.random.binomial(1, bias, num_samples)

# 将数据转换为特征矩阵X和标签y
X = coin_flips.reshape(-1, 1)  # 特征是投掷次数
y = coin_flips  # 标签是结果

# 将数据集分为训练集和测试集
# 这里将测试集的比例增加到50%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(X_test)), y_test, color='blue', label='True values', s=10)
plt.scatter(np.arange(len(X_test)), y_pred, color='red', alpha=0.5, label='Predicted values', s=10)
plt.title('Logistic Regression: Bent Coin Prediction')
plt.xlabel('Sample Index')
plt.ylabel('Outcome (1 for Heads, 0 for Tails)')
plt.legend()
plt.show()

