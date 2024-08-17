import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, likes_file):
        self.likes_file = likes_file
        self.user_item_matrix = None
        self.user_similarity = None
        self.svd = None
        self._load_data()
        self._train_model()
    
    def _load_data(self):
        """Load data from CSV file and create user-item matrix."""
        likes = pd.read_csv(self.likes_file)
        self.user_item_matrix = pd.pivot_table(likes, index='user_id', columns='post_id', values='like', fill_value=0)
    
    def _train_model(self):
        """Train the collaborative filtering model using matrix factorization."""
        # Perform matrix factorization
        self.svd = TruncatedSVD(n_components=10)  # Number of latent features
        user_item_matrix_svd = self.svd.fit_transform(self.user_item_matrix)
        
        # Compute user similarity
        self.user_similarity = cosine_similarity(user_item_matrix_svd)
    
    def recommend_posts(self, user_id, top_n=5):
        """Recommend posts for a given user based on collaborative filtering."""
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the data.")
        
        user_index = self.user_item_matrix.index.get_loc(user_id)
        
        # Compute scores for all posts
        scores = self.user_similarity[user_index].dot(self.user_item_matrix.fillna(0)) / np.array([np.abs(self.user_similarity[user_index]).sum()])
        
        # Get top N posts
        top_posts = scores.argsort()[::-1][:top_n]
        return top_posts

# Example usage
if __name__ == "__main__":
    recommender = Recommender('data/likes.csv')
    user_id = 1
    recommended_posts = recommender.recommend_posts(user_id)
    print(f"Recommended posts for user {user_id}: {recommended_posts}")
