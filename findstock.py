import baostock as bs
import pandas as pd

# 登录 Baostock
lg = bs.login()
print("login respond error_code:", lg.error_code)
print("login respond error_msg:", lg.error_msg)

# 获取估值数据
valuation_rs = bs.query_history_k_data_plus(
    "sz.000001",
    "date, code, peTTM, pbMRQ, psTTM",
    start_date="2023-01-01",
    end_date="2023-12-31",
    frequency="d",
    adjustflag="3"
)

valuation_list = []
while valuation_rs.next():
    valuation_list.append(valuation_rs.get_row_data())
valuation_df = pd.DataFrame(valuation_list, columns=valuation_rs.fields)

# 获取财务指标数据
profit_rs = bs.query_profit_data(
    code="sz.000001", 
    year=2023, 
    quarter=3
)
profit_list = []
while profit_rs.next():
    profit_list.append(profit_rs.get_row_data())
profit_df = pd.DataFrame(profit_list, columns=profit_rs.fields)

# 合并数据
merged_df = pd.merge(valuation_df, profit_df, on=["code"], how="left")

# 输出结果

merged_df.to_csv("D:/2025-winter/history_A_stock_k_data.csv", index=False)

print(merged_df)

# 登出 Baostock
bs.logout()
