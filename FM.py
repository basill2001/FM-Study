import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.optim as optim
from skopt import gp_minimize

class FM_Preprocessing:

    # 생성자
    def __init__(self, df, target_col='target', num_epochs=10):
        self.df = df
        self.target_col = target_col
        self.num_epochs = num_epochs
        self.X_tensor, self.y_tensor, self.c_values_tensor, self.user_feature_tensor, self.item_feature_tensor, self.all_item_ids, self.num_features = self.prepare_data()
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The df parameter should be a pandas DataFrame.")

        if target_col not in df.columns:
            raise ValueError(f"The target column {target_col} is not in the DataFrame.")

    # target=0인 데이터 만들기
    def generate_not_purchased_data(self,df):
        # customer_frequency, product_frequency ; 컬럼 추가

        df['customer_frequency'] = df.groupby('AUTH_CUSTOMER_ID')['AUTH_CUSTOMER_ID'].transform('count')
        df['product_frequency'] = df.groupby('PRODUCT_CODE')['PRODUCT_CODE'].transform('count')
        unique_customers = df['AUTH_CUSTOMER_ID'].unique()
        unique_products = df['PRODUCT_CODE'].unique()


        not_purchased_products_list = []


        for customer in unique_customers:
            customer_frequency = df[df['AUTH_CUSTOMER_ID'] == customer]['customer_frequency'].iloc[0]
            purchased_products = df[df['AUTH_CUSTOMER_ID'] == customer]['PRODUCT_CODE'].unique()

            birth_year = df[df['AUTH_CUSTOMER_ID'] == customer]['BIRTH_YEAR'].iloc[0]
            gender = df[df['AUTH_CUSTOMER_ID'] == customer]['GENDER'].iloc[0]

            # customer가 구매하지 않은 products의 리스트
            not_purchased_products = [product for product in unique_products if product not in purchased_products]
            not_purchased_products_data = [{'AUTH_CUSTOMER_ID': customer,
                                            'PRODUCT_CODE': product,
                                            'Birth_Category': birth_year,
                                            'gender_category': gender,
                                            'customer_frequency': customer_frequency,
                                            'product_frequency': df[df['PRODUCT_CODE'] == product]['product_frequency'].iloc[0]}
                                        for product in not_purchased_products]

            not_purchased_products_list.extend(not_purchased_products_data)

        not_purchased_df = pd.DataFrame(not_purchased_products_list)
        not_purchased_df['target'] = 0

        return not_purchased_df

    # X_tensor, y_tensor, c_values_tensor, user_feature_tensor, item_feature_tensor, all_item_ids, num_features 만들기
    def prepare_data(self):
        X = self.df.drop(columns=[self.target_col, 'PRODUCT_CODE', 'C', 'AUTH_CUSTOMER_ID'])
        y = self.df[self.target_col]
        c = self.df['C']

        # tensor로 바꾸기
        X_tensor = torch.tensor(X.values,dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1)

        c_values_tensor = torch.tensor(c, dtype=torch.float32)
        c_values_tensor = torch.where(c_values_tensor < 1, c_values_tensor * 100, c_values_tensor)

        # unique user에 대한 df 생성
        unique_user_df = self.df.drop_duplicates(subset=['AUTH_CUSTOMER_ID']).sort_values('AUTH_CUSTOMER_ID')
        # user들의 feature에 관한 tensor 생성
        # 생년과 성별에 관한 tensor
        user_features_df = unique_user_df[['BIRTH_YEAR', 'GENDER']]
        user_feature_tensor = torch.tensor(pd.get_dummies(user_features_df).values, dtype=torch.float32)

        # unique itme에 대한 df 생성
        unique_item_df = self.df.drop_duplicates(subset=['PRODUCT_CODE']).sort_values('PRODUCT_CODE')
        # item들의 feature에 관한 tensor 생성
        item_features_df = unique_item_df.filter(like='DEPTH') # DEPTH column들 가져오기
        item_feature_tensor = torch.tensor(item_features_df.values, dtype=torch.float32)

        # 모든 item들의 id
        all_item_ids = list(self.df.PRODUCT_CODE.unique())

        num_features = X.shape[1]

        # Unnamed:0, ORDER_DATE, BIRTH_YEAR, GENDER, user_total_fq, item_total_fq, total_fq, DEPTH1, DEPTH2, DEPTH3, DEPTH4
        return X_tensor, y_tensor, c_values_tensor, user_feature_tensor, item_feature_tensor, all_item_ids, num_features
    
# FM 모델 만들기
class FactorizationMachine(nn.Module):
    # 생성자
    def __init__(self, num_features, num_factors, lr=0.01, weight_decay=0.01):
        super(FactorizationMachine, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.w = nn.Parameter(torch.randn(num_features))              # num_features 길이의 난수 텐서를 모델의 파라미터로 등록
        self.v = nn.Parameter(torch.randn(num_features, num_factors)) # num_features X num_facotrs 크기의 난수 텐서를 모델의 파라미터로 등록
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        #self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func =  nn.MSELoss()
        self.weight_decay = weight_decay

    # forward 연산 ; 입력층부터 출력층까지 계산
    def forward(self, x):
      linear_terms = torch.matmul(x, self.w) # x와 w의 행렬곱
      interactions = 0.5 * torch.sum(
        torch.matmul(x, self.v) ** 2 - torch.matmul(x ** 2, self.v ** 2),
        dim=1,
        keepdim=True
      )
      return linear_terms + interactions.squeeze()

    def forward_for_recommendation(self, x):
        w=torch.tensor([model.w[2],model.w[3],model.w[7],model.w[8],model.w[9],model.w[10]])
        v=torch.stack([model.v[2],model.v[3],model.v[7],model.v[8],model.v[9],model.v[10]])
        linear_terms = torch.matmul(x, w) # x와 w의 행렬곱
        interactions = 0.5 * torch.sum(
          torch.matmul(x, v) ** 2 - torch.matmul(x ** 2, v ** 2),
          dim=1,
          keepdim=True
      )
        return linear_terms + interactions.squeeze()

    def loss(self, y_pred, y_true, c_values):
        mse = (y_pred - y_true.float()) ** 2
        weighted_mse = c_values * mse
        l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2  # L2 regularization
        return torch.mean(weighted_mse) + self.weight_decay * l2_reg

    def train_step(self, x, y, c_values):
        self.optimizer.zero_grad()  # gradient를 0으로 만들어주기
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y, c_values)
        loss.backward()
        self.optimizer.step() # optimizer에 loss function이 최소화되도록 파라미터 수정
        return loss.item()

    # 각 user마다 n개의 item 추천해주기
    def recommend_top_n_items(self, user_features, all_item_features, all_item_ids, top_n=5):
        combined_features = torch.cat([user_features.expand(all_item_features.shape[0], -1), all_item_features], dim=1) # 텐서 합쳐주기
        combined_features=normalize(combined_features)
        with torch.no_grad():        # 더이상 학습 X
            scores = self.forward_for_recommendation(combined_features)
        sorted_indices = torch.argsort(scores, descending=True)[:top_n]
        return [all_item_ids[i] for i in sorted_indices]

    def recommend_top_n_items_for_all_users(self, user_features_list, all_item_features, all_item_ids, top_n=5):
        for i, user_features in enumerate(user_features_list):
            user_id = i  # can replace with actual user ID if I have
            top_n_items = self.recommend_top_n_items(user_features, all_item_features, all_item_ids, top_n)
            recommendations[user_id] = top_n_items
            print(user_id,top_n_items)
        return recommendations
    
if __name__ == '__main__':

    try:
        df = pd.read_csv('C:/Users/HOME/문서/한양대/3-2/산업공학연구실현장실습2/data_new.csv')
        preprocess = FM_Preprocessing(df)
        # need to be done
        # not_purchased_df = preprocess.generate_not_purchased_data(df)
        # ...
    except Exception as e:
        print(f"An error occurred: {e}")

model=FactorizationMachine(11,2)
num_epochs=5
X_tensor=normalize(preprocess.X_tensor)

for epoch in range(num_epochs):
    loss = model.train_step(X_tensor, preprocess.y_tensor,preprocess.c_values_tensor)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

recommendations = {}
recommendations["Loss"]=loss
recommendations["number of factors"]=model.num_factors
# Make recommendations
recommendations = model.recommend_top_n_items_for_all_users(preprocess.user_feature_tensor, preprocess.item_feature_tensor, preprocess.all_item_ids, top_n=5)


import csv
with open("recommendations.csv","w") as file:
  writer=csv.writer(file)
  for k,v, in recommendations.items():
    writer.writerow([k,v])