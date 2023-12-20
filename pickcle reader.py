import pickle
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

with open("C:/Users/HOME/문서/한양대/3-2/산업공학연구실현장실습2/order_info_frequency.pickle","rb") as f:
    data=pickle.load(f)

print(data)

# X가 tensor로 바뀌지 않아서 time string data를 float로 바꾸기
data["ORDER_DATE"]=pd.to_datetime(data["ORDER_DATE"],format="%Y-%m-%d %H:%M:%S")
data["ORDER_DATE"]=data["ORDER_DATE"].apply(lambda x : x.timestamp())

# GENDER도 바꾸기
le=LabelEncoder()
gen=le.fit_transform(data["GENDER"])
data.drop("GENDER",axis=1,inplace=True)
data["GENDER"]=gen

# fq 계산
data["total_fq"]=0.5*data["user_total_fq"]*0.5*data["item_total_fq"]
data["total_fq"]=data["total_fq"].astype(int)
data["C"]=data["total_fq"]/data["total_fq"].sum()
data["target"]=1

# DEPTH column 생성
data["PRODUCT_CODE"]=data["PRODUCT_CODE"].astype(str)
data["DEPTH1"]=data["PRODUCT_CODE"].str[:2]
data["DEPTH2"]=data["PRODUCT_CODE"].str[:4]
data["DEPTH3"]=data["PRODUCT_CODE"].str[:6]
data["DEPTH4"]=data["PRODUCT_CODE"].str[:8]
print(data)
# encoded_df=pd.get_dummies(data,columns=["DEPTH1"])

data.to_csv("C:/Users/HOME/문서/한양대/3-2/산업공학연구실현장실습2/data_new.csv")