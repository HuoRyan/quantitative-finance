import baostock as bs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# 登录baostock
bs.login()

# 获取股票累计收益率数据
def get_stock_data(stock_code, start_date, end_date):
    """
    获取股票的累计收益率及特征数据（PE, PB）
    """
    # 获取每日行情数据，包括 PE 和 PB
    rs_daily = bs.query_history_k_data_plus(
        stock_code,
        "date, close, peTTM, pbMRQ",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    daily_list = []
    while (rs_daily.error_code == '0') & rs_daily.next():
        daily_list.append(rs_daily.get_row_data())
    daily_data = pd.DataFrame(daily_list, columns=rs_daily.fields)
    
    # 转换列数据类型
    daily_data['close'] = pd.to_numeric(daily_data['close'], errors='coerce')
    daily_data['peTTM'] = pd.to_numeric(daily_data['peTTM'], errors='coerce')
    daily_data['pbMRQ'] = pd.to_numeric(daily_data['pbMRQ'], errors='coerce')
    
    # 删除空值
    daily_data.dropna(inplace=True)
    
    # 计算累计收益率
    start_price = daily_data.iloc[0]['close']
    end_price = daily_data.iloc[-1]['close']
    cumulative_return = (end_price - start_price) / start_price

    # 提取特征均值作为整体特征
    avg_pe = daily_data['peTTM'].mean()
    avg_pb = daily_data['pbMRQ'].mean()
    
    return {
        'Stock': stock_code,
        'Cumulative Return': cumulative_return,
        'PE': avg_pe,
        'PB': avg_pb
    }

# 获取数据
stock_list = ["sh.601688","sh.600703","sh.601318","sh.601398","sh.600000", "sz.000001","sh.601628"]  # 示例股票代码
train_data = pd.DataFrame([get_stock_data(stock, "2023-01-01", "2023-12-31") for stock in stock_list])
test_data = pd.DataFrame([get_stock_data(stock_list[0], "2024-01-01", "2024-12-31") ])

# 关闭baostock
bs.logout()

# 模型训练
X_train = train_data[['PE', 'PB']]
y_train = train_data['Cumulative Return']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 线性回归模型
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# 随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

#神经网络模型
ml_model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=1000, random_state=42)
ml_model.fit(X_train_scaled, y_train)

# 模型预测
X_test = test_data[['PE', 'PB']]
y_test = test_data['Cumulative Return']
X_test_scaled = scaler.transform(X_test)
test_data['Predicted Cumulative Return (Linear)'] = linear_model.predict(X_test_scaled)
test_data['Predicted Cumulative Return (RF)'] = rf_model.predict(X_test_scaled)
test_data['Predicted Cumulative Return (MLP)'] = ml_model.predict(X_test_scaled)
test_data['Cumulative Return'] = test_data['Cumulative Return']

# 计算误差
error_linear = abs(test_data['Predicted Cumulative Return (Linear)'] - test_data['Cumulative Return']).mean()
error_rf = abs(test_data['Predicted Cumulative Return (RF)'] - test_data['Cumulative Return']).mean()
error_mlp = abs(test_data['Predicted Cumulative Return (MLP)'] - test_data['Cumulative Return']).mean()

# 比较三种模型的误差
errors = {
    'Linear Regression': error_linear,
    'Random Forest': error_rf,
    'MLP Regressor': error_mlp
}
 # 找出误差最小的模型
best_model = min(errors, key=errors.get) 



# 输出结果
print("All test stock predictions:")
print(test_data)
# 打印最优模型
print("Best model:", best_model)

# 保存为 CSV 文件
test_data.to_csv('D:/2025-winter/output2.csv', index=False)  
