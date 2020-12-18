import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
%matplotlib inline


# In[]:
# データ読み込み
weather_data_df = pd.read_csv("weather_data_Osaka.csv",encoding='cp932', header = 0)
df = weather_data_df.copy()
# 1980年 情報表示
df['年月']
# In[]:
#8月のデータのみを抽出
AUG_df = df.query('年月.str.contains("Aug")', engine='python')

#8月の年月, 降水量の合計(mm)データのみを抽出
AUG_df[['年月','降水量の合計(mm)']]

# In[]:
#年月から, 年だけを取り出す
Tmp = []
for year in AUG_df['年月'].str.split("-", expand=True)[1]:
    if int(year) >= 80:
        Tmp.append('19' + year)
    else:
        Tmp.append('20' + year)

AUG_df = AUG_df.reset_index()
AUG_df['年'] = pd.Series(Tmp).astype(int).copy()
AUG_df[['年','降水量の合計(mm)']]
# In[]:
#########
