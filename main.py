import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#データのファイルの指定
#filepath = './data/u.data.txt'

class SBDRecommender:
    def __init__(self,filepath):
        self.filepath = filepath
        self.data = None
        self.user_map = None
        self.item_map = None
        self.R = None
        self.predict_matrix = None
    
    def dataload(self):
        columns = ["user_id","movie_id","rating","timestamp"]
        self.data = pd.read_csv(self.filepath,sep="\t",names=columns)
        df = self.data

        #remove timestamp raw
        df.drop(columns="timestamp",inplace=True)

        items = df["movie_id"].unique()

        #self.train_data, self.test_data = train_test_split(self.data,test_size=0.2,random_state=42)

        interact_matrix = df.pivot_table(index="user_id",columns="movie_id",values="rating",fill_value=np.nan)
        self.user_map = {idx: user for idx, user in enumerate(interact_matrix.index)}
        self.item_map = {idx: item for idx, item in enumerate(items)}

        R = interact_matrix.fillna(0).values

        self.R = R

    def learn(self):
        #ハイパーパラメータ
        latent_facters = 2
        learning_rate = 0.01
        num_epochs = 50
        lambda_reg = 0.02
        R = self.R
        num_users,num_movies = self.R.shape

        print("latent_facters:", latent_facters)
        print("num_users:", num_users)

        #ユーザ行列Uとアイテム行列Vの初期化
        U = np.random.normal(scale=1.0/latent_facters, size=(num_users,latent_facters))
        V = np.random.normal(scale=1.0/latent_facters, size=(latent_facters,num_movies))
        Y = np.dot(U,V)

        #SGDによる行列分解
        for epoch in range(num_epochs):
            g = (((R-Y)**2).sum() + lambda_reg * ((U**2).sum() + (V**2).sum()).sum())/2
            grad_u = -np.dot((R-Y),V.T) + lambda_reg*U
            grad_v = -np.dot((R-Y).T,U).T + lambda_reg*V

            U += learning_rate*grad_u
            V += learning_rate*grad_v

            loss = abs(g-(((R-Y)**2).sum() + lambda_reg * ((U**2).sum() + (V**2).sum()).sum())/2)/(((R-Y)**2).sum() + lambda_reg * ((U**2).sum() + (V**2).sum()))/2

            if loss < 0.01:
                break

        self.prediction_matrix = np.dot(U,V)
            

    def predict(self,user_id,movie_id):
        user_idx = list(self.user_map.values()).index(user_id)
        item_idx = list(self.item_map.values()).index(movie_id)

        rating = self.prediction_matrix[user_idx,item_idx]

        return rating
    
    def recommend(self,user_id,n=5):
        user_idx = list(self.user_map.values()).index(user_id)

        rating = self.prediction_matrix[user_idx,:]
        top_movies_idx = np.argsort(rating)[::-1][:n]
        top_movies = [int(self.item_map[idx]) for idx in top_movies_idx]

        return top_movies
    
"""
def main():
    filepath = ".data/u.data.txt"
    recommender = SBDRecommender(filepath)
    recommender.dataload()
    recommender.learn()

    target_user = 1
    target_movie = 1

    predicted_rating = recommender.predict(target_user,target_movie)
    recommend_movie = recommender.recommend(target_user)

    print(f"Predict rating for (User,Item) = {(target_user,target_movie)}:{predicted_rating}")
    print(f"Top recommended movies for User {target_user}: {recommend_movie}")


if __name__ == "__main__":
    main()
  
"""

