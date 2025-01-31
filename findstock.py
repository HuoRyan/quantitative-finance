import baostock as bs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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
    start_price = daily_data.iloc[0]['close']  # 起始收盘价
    end_price = daily_data.iloc[-1]['close']  # 结束收盘价
    cumulative_return = (end_price - start_price) / start_price  # 累计收益率

    # 提取特征均值作为整体特征
    avg_pe = daily_data['peTTM'].mean()
    avg_pb = daily_data['pbMRQ'].mean()
    
    return {
        'Stock': stock_code,
        'Cumulative Return': cumulative_return,
        'PE': avg_pe,
        'PB': avg_pb
    }

# 获取训练数据
stock_list = ["sh.600000", "sz.000001"]
train_data = pd.DataFrame([get_stock_data(stock, "2023-01-01", "2023-12-31") for stock in stock_list])

# 获取测试数据
future_stock_list = ["sh.601398", "sz.000001"]
test_data = pd.DataFrame([get_stock_data(stock, "2024-01-01", "2024-12-31") for stock in future_stock_list])

# 关闭baostock
bs.logout()

# 模型训练
X_train = train_data[['PE', 'PB']]
y_train = train_data['Cumulative Return']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestRegressor(n_estimators=100, random_state=42)  # 使用100棵树的随机森林
model.fit(X_train_scaled, y_train)

# 模型预测
X_test = test_data[['PE', 'PB']]
X_test_scaled = scaler.transform(X_test)
test_data['Predicted Cumulative Return'] = model.predict(X_test_scaled)

# 筛选高收益股票
selected_stocks = test_data[test_data['Predicted Cumulative Return'] > 0.1]

# 输出结果
print("All test stock predictions:")
print(test_data)

print("\nSelected stocks with high predicted cumulative return:")
print(selected_stocks)
