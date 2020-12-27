import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression


# In[]:
# データ読み込み
weather_data_df = pd.read_csv("weather_data_Osaka.csv",encoding='cp932', header = 0)
df = weather_data_df.copy()
# 1980年 情報表示
df.head(12)

#8月のデータのみを抽出
AUG_df = df.query('年月.str.contains("Aug")', engine='python')
SEP_df = df.query('年月.str.contains("Sep")', engine='python')
OCT_df = df.query('年月.str.contains("Oct")', engine='python')
#df.describe()
# In[]:
plt.figure(figsize=(10,8))
df['日照時間(時間)'].plot(legend=True)

# In[]:
#plt.scatter(df['平均気温(℃)'], df['日照時間(時間)']) # 平均部屋数と住宅価格の散布図をプロット
plt.scatter(df['日照時間(時間)'], df['降水量の合計(mm)']) # 平均部屋数と住宅価格の散布図をプロット


plt.title('Scatter Plot of RM vs MEDV')    # 図のタイトル
plt.xlabel('Average number of rooms [RM]') # x軸のラベル
plt.ylabel('Prices in $1000\'s [MEDV]')    # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()
# In[]:
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#X = df[['平均気温(℃)']].values         # 説明変数（Numpyの配列）
#X = df[['降水量の合計(mm)']].values         # 説明変数（Numpyの配列）
X = df[['日照時間(時間)']].values         # 目的変数（Numpyの配列）
Y = df['平均気温(℃)'].values         # 目的変数（Numpyの配列）


lr.fit(X, Y)

# In[]:
print('coefficient = ', lr.coef_[0]) # 説明変数の係数を出力
print('intercept = ', lr.intercept_) # 切片を出力
# In[]:
plt.scatter(X, Y, color = 'blue')         # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X, lr.predict(X), color = 'red') # 回帰直線をプロット

plt.title('Regression Line')               # 図のタイトル
plt.xlabel('Average number of rooms [RM]') # x軸のラベル
plt.ylabel('Prices in $1000\'s [MEDV]')    # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()
AUG_df[['年月','日照時間(時間)']]
AUG_df[['年月','降水量の合計(mm)']]
AUG_df['年月']
AUG_df['年月'].str.split("-", expand=True)[1]

ABC = []
for year in AUG_df['年月'].str.split("-", expand=True)[1]:
    #AAA = []
    #ABC = []
    #print(year)
    if int(year) >= 80:
        #print('19' + year)
        #print(year)
        AAA.append(year)
        ABC.append('19' + year)
    else:
        ABC.append('20' + year)

ABC

AUG_df.head()
pd.Series(ABC).astype(int)

AUG_df = AUG_df.reset_index()
AUG_df['年'] = pd.Series(ABC).astype(int).copy()
AUG_df[['年','降水量の合計(mm)']]
# In[]:
plt.scatter(AUG_df['年'], AUG_df['降水量の合計(mm)']) # プロット


plt.title('Scatter Plot of Year vs Monthly Rainfall')    # 図のタイトル
plt.xlabel('year') # x軸のラベル
plt.ylabel('monthly rainfall(mm)')    # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()
# In[]:
AUG_df[['年','降水量の合計(mm)']]
# In[]:
#線形2乗回帰を行い,説明変数の係数と切片を出力
lr = LinearRegression()

X = AUG_df[['年']].values         # 目的変数（Numpyの配列）
Y = AUG_df['降水量の合計(mm)'].values         # 目的変数（Numpyの配列）

lr.fit(X, Y)

print('coefficient = ', lr.coef_[0]) # 説明変数の係数を出力
print('intercept = ', lr.intercept_) # 切片を出力
# In[]:
#データと回帰直線を表示
plt.scatter(X, Y)         # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X, lr.predict(X), color = 'red') # 回帰直線をプロット

plt.title('Scatter Plot of Year vs Monthly Rainfall')    # 図のタイトル
plt.xlabel('year')                         # x軸のラベル
plt.ylabel('monthly rainfall(mm)')         # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()
# In[]:
#残差プロット
Y_pred = lr.predict(X) # データを用いて目的変数を予測
plt.scatter(X, Y_pred - Y, color = 'blue')      # 残差をプロット
plt.hlines(y = 0, xmin = X.min(), xmax = X.max(), color = 'black') # x軸に沿った直線をプロット
plt.title('Residual Plot')                                # 図のタイトル
plt.xlabel('Predicted Values')                            # x軸のラベル
plt.ylabel('Residuals')                                   # y軸のラベル
plt.grid()                                                # グリッド線を表示

plt.show()
# In[]:
from sklearn.metrics import r2_score

print('r^2 決定係数: ', r2_score(Y, lr.predict(X)))


r2_score(Y, lr.predict(X))
# In[]:
# 2060年をデータに追加
X2 = X.copy()
X2 = np.append(X2, 2060)
X.shape
X2.shape
lr.predict([[2060]])[0]
X2 = X2.reshape([41, 1])
# In[]:
# 2060年の予測を合わせたデータと回帰直線を表示
plt.scatter(X, Y)         # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X2, lr.predict(X2), color = 'red') # 回帰直線をプロット
plt.scatter(2060, lr.predict([[2060]]))
plt.title('Scatter Plot of Year vs Monthly Rainfall')    # 図のタイトル
plt.xlabel('year') # x軸のラベル
plt.ylabel('monthly rainfall(mm)') # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()
# In[]:
ABC = []
for year in OCT_df['年月'].str.split("-", expand=True)[1]:
    if int(year) >= 80:
        ABC.append('19' + year)
    else:
        ABC.append('20' + year)

ABC
pd.Series(ABC).astype(int)

OCT_df = OCT_df.reset_index()
OCT_df['年'] = pd.Series(ABC).astype(int).copy()

# In[]:
#plt.scatter(df['日照時間(時間)'], df['平均気温(℃)']) # 平均部屋数と住宅価格の散布図をプロット
#plt.scatter(SEP_df['年'], SEP_df['平均気温(℃)']) # 平均部屋数と住宅価格の散布図をプロット
plt.scatter(OCT_df['年'], OCT_df['日照時間(時間)']) # 平均部屋数と住宅価格の散布図をプロット
#plt.scatter(SEP_df['年'], SEP_df['降水量の合計(mm)']) # 平均部屋数と住宅価格の散布図をプロット


plt.title('Scatter Plot of RM vs MEDV')    # 図のタイトル
plt.xlabel('year') # x軸のラベル
plt.ylabel('')    # y軸のラベル
plt.grid()                                 # グリッド線を表示

plt.show()
