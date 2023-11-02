# streamlit 不能寫 %reset -f
# streamlit 註解直接用 #，不要用 """ """。
# 檔名可以有中文也可以有 "."。 #shiny不行
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

def VaR(returns, alpha=0.05, window=252, VaR_method="Normal"):
    data = returns.copy()
    if VaR_method == "Historical Simulation":
        data["Rolling_VaR"] = -data.rolling(window=window).quantile(quantile=alpha)

    if VaR_method == "Normal":
        mu = data.rolling(window=window).mean()
        sigma = data.rolling(window=window).std()
        # 使用常態分配計算 VaR
        z_alpha = norm.ppf(alpha)
        # norm.ppf(alpha) 為 N(0, 1) 左尾機率 alpha 之分位數
        data["Rolling_VaR"] = -(mu+sigma*z_alpha)
    return data["Rolling_VaR"]


#%%
# 臺灣證券交易所發行量加權股價指數成分股暨市值比重
# google => python ead html table =>
# Reading HTML tables with Pandas =>
# https://pbpython.com/pandas-html-table.html
# or
# How to read HTML tables using Python? =>
# https://www.askpython.com/python-modules/pandas/read-html-tables

url = "https://www.taifex.com.tw/cht/9/futuresQADetail"
table = pd.read_html(url)
stocks1 = table[0].iloc[:,1:4]
stocks2 = table[0].iloc[:,5:8]
stocks1.columns = ["代號", "證券名稱", "市值佔大盤比重"]
stocks2.columns = ["代號", "證券名稱", "市值佔大盤比重"]
stocks1 = stocks1.dropna()
stocks2 = stocks2.dropna()
stocks1["代號"] = stocks1["代號"].astype(str)
stocks2["代號"] = [str(int(stocks_代號)) for stocks_代號 in stocks2["代號"]]
# stocks2["代號"] 有 .0。
stocks = pd.concat([stocks1, stocks2], axis=0)
stocks = stocks.reset_index(drop=True)
# stocks["市值佔大盤比重"] 最後為 % 符號。
stocks["市值佔大盤比重"] = stocks["市值佔大盤比重"].str[:-1].astype(float)/100
stocks["代號"] = [stocks["代號"][i]+".TW" for i in range(len(stocks))]
stocks["代號_證券名稱_市值佔大盤比重"] = [stocks["代號"][i]+" "+stocks["證券名稱"][i]+" "+str(round(stocks["市值佔大盤比重"][i], 6)) for i in range(len(stocks))]


#%%
# pip install streamlit
# google => python streamlit input =>
# Input widgets - Streamlit Docs =>
# https://docs.streamlit.io/library/api-reference/widgets

import streamlit as st 

#在原本程式的基礎下把參數數字改為st.
#有寫sidebar選項會顯示在左邊 沒寫的話會在上面
#selectbox為下拉視窗
stock_ticker_name = st.sidebar.selectbox("Select asset", stocks["代號_證券名稱_市值佔大盤比重"])
ticker = stock_ticker_name.split(" ")[0] #用空格做分割找第0個 #原2330.TW 台積電 0.265781會被割成3個

start = st.sidebar.date_input(label="start date",
                              value=dt.date(2020, 1, 1),
                              format="YYYY-MM-DD")
end = st.sidebar.date_input(label="end date",
                            value=dt.datetime.now(),
                            format="YYYY-MM-DD")
interval = "1d"
df = yf.download(ticker,
                 start=start,
                 end=end,
                 interval=interval,
                 auto_adjust=True)
# auto_adjust=True => "Close" 即為 "Adj Close"，
# "Open", "High", "Low", "Close" 均做除權息的調整。
# auto_adjust 內建值為 False。

# 計算 log-return：
# df["returns"] = np.log(1+df["Close"].pct_change())
df["returns"] = np.log(df["Close"]/df["Close"].shift(1))

alpha = st.sidebar.slider("alpha",
                          min_value=0.001,
                          max_value=0.1,
                          value=0.05, #初始顯示值
                          step=0.001, #滑標移動一次改變多少
                          format="%.3f")
# google => st.sidebar.slider decimal 3 point =>
# Increasing Decimal Places with st.number_input() =>
# https://discuss.streamlit.io/t/increasing-decimal-places-with-st-number-input/12874
window = st.sidebar.slider("window",
                           min_value=10,
                           max_value=250,
                           value=50,
                           step=10)
df[f"Rolling_VaR_HS_{alpha}"] = VaR(df[["returns"]],
                                    alpha=alpha,
                                    window=window,
                                    VaR_method="Historical Simulation")
df[f"Rolling_VaR_Normal_{alpha}"] = VaR(df[["returns"]],
                                        alpha=alpha,
                                        window=window,
                                        VaR_method="Normal")
df[f"Rolling_VaR_HS_{alpha}_predict"] = df[f"Rolling_VaR_HS_{alpha}"].shift(1)
df[f"Rolling_VaR_Normal_{alpha}_predict"] = df[f"Rolling_VaR_Normal_{alpha}"].shift(1)


#%%
# matplotlib 畫回溯測試 (backtesting) 圖
import matplotlib.pyplot as plt

fontsize = 16
plt.rc("font", size=fontsize)
# plt.rcParams.update({"font.size": fontsize})
# controls default text sizes

# 二條線之 linestyle 不相同：
fig, ax = plt.subplots(figsize=(16, 12))
df[["returns"]].plot.line(color="blue",
                          linestyle="solid",
                          linewidth=1,
                          alpha=0.6,
                          xlabel="Date",
                          ylabel="log-return",
                          title=f"Rolling VaR ({ticker}, alpha={alpha}, window={window})",
                          ax=ax)
(-df[[f"Rolling_VaR_HS_{alpha}_predict"]]).plot.line(color="red",
                                                     linestyle="dashed",
                                                     linewidth=1,
                                                     ax=ax)
(-df[[f"Rolling_VaR_Normal_{alpha}_predict"]]).plot.line(color="brown",
                                                         linestyle="dashdot",
                                                         linewidth=1,
                                                         ax=ax)
# 二個都是 DataFrame.plot() => legend 都會出現，
# 若第一個是 Series.plot() => 這個線的 legend 不會出現。
st.pyplot(fig, transparent=True)


#%%
# plotly 畫回溯測試 (backtesting) 圖
# pip install plotly
import plotly
import plotly.graph_objects as go

# google => python plotly browser =>
# Displaying figures in Python =>
# https://plotly.com/python/renderers/
plotly.io.renderers.default = "browser"

# google => python plotly scatter line =>
# Scatter plots in Python =>
# https://plotly.com/python/line-and-scatter/

# google => plotly line color =>
# Line charts in Python =>
# https://plotly.com/python/line-charts/

fig_plotly = go.Figure()
fig_plotly.add_traces(
    go.Scatter(x=df.index,
               y=df["returns"],
               mode="lines",
               line=dict(color="blue", width=1, dash="solid"),
               name="returns")
    )
fig_plotly.add_traces(
    go.Scatter(x=df.index,
               y=-df[f"Rolling_VaR_HS_{alpha}_predict"],
               mode="lines",
               line=dict(color="red", width=1, dash="dash"),
               name=f"Rolling_VaR_HS_{alpha}_predict")
    )
fig_plotly.add_traces(
    go.Scatter(x=df.index,
               y=-df[f"Rolling_VaR_Normal_{alpha}_predict"],
               mode="lines",
               line=dict(color="brown", width=1, dash="dashdot"),
               name=f"Rolling_VaR_Normal_{alpha}_predict")
    )

# google => plotly line scatter x label y label =>
# Setting the font, title, legend entries, and axis titles in Python =>
# https://plotly.com/python/figure-labels/

# google => plotly legend position =>
# Legends in Python =>
# https://plotly.com/python/legend/

fig_plotly.update_layout(
    title=f"Rolling VaR ({ticker}, alpha={alpha}, window={window})",
    xaxis_title="Date",
    yaxis_title="log-return",
    xaxis=dict(title="Date", showline=True, linecolor="black"),
    yaxis=dict(title="log-return", showline=True, linecolor="black"),
    legend=dict(xanchor="left",
                x=0.04,
                yanchor="top",
                y=0.96),
    width=700,
    height=550)
st.plotly_chart(fig_plotly)

#%%
# 打開 "命令提示字元" 或 "終端機"，開始執行程式 (或進入 Command Prompt 視窗)：
# (a) Anaconda => Anaconda Prompt (or Anaconda Powershell Prompt)
# (b) WinPython => WinPython Command Prompt.exe
# Streamlit run "C:\Anaconda3\test\07-Value-at-Risk-rolling().quantile()-VaR_Normal-DataFrame.plot()-matplotlib-plotly-臺灣證券交易所發行量加權股價指數成分股暨市值比重-streamlit1(1).py"
# Google Chrome => http://localhost:8501/
# Google Chrome 不關掉，程式修改存檔後，可直接執行 Google Chrome output 畫面右上角的 Return。
# 指定 localhost port => streamlit run app.py --server.port 8502
# Mac 不能使用 MacOS 內建的 Safari 瀏覽器，改用 Google Chrome 就沒問題了。