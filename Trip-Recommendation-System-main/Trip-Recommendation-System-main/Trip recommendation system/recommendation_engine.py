import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

class HybridRecommender:
    def __init__(self):
        """
        Initialize the recommender system without evaluation mode
        as it will be handled by the separate evaluator
        """
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load data from CSV files"""
        self.items = pd.read_csv('items.csv')
        self.all_ratings = pd.read_csv('ratings.csv')
        self.users = pd.read_csv('users.csv')
        self.train_ratings = self.all_ratings  # Will be updated by evaluator

    def preprocess_data(self):
        """Preprocess data for recommendations"""
        # Collaborative Filtering Matrix
        self.user_item_matrix = self.train_ratings.pivot_table(
            index='userId', 
            columns='itemId', 
            values='rating'
        ).fillna(0)
        self.matrix = csr_matrix(self.user_item_matrix.values)
        
        # Content-Based Features
        self.items['features'] = self.items['category'] + ' ' + \
                                self.items.get('tags', '') + ' ' + \
                                self.items.get('description', '')
        
        self.tfidf = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2),
            max_features=5000  # Limit features for better performance
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.items['features'])
        
        # Weighted Ratings with configurable parameters
        self.calculate_weighted_ratings(quantile=0.85)

    def calculate_weighted_ratings(self, quantile=0.85):
        """
        Calculate weighted ratings with configurable parameters
        
        Args:
            quantile: Quantile for minimum votes threshold
        """
        C = self.items['p_rating'].mean()
        m = self.items['count'].quantile(quantile)
        self.items['weighted_rating'] = self.items.apply(
            lambda x: (x['count']/(x['count']+m) * x['p_rating']) + 
                     (m/(m+x['count']) * C),
            axis=1
        )

    def collaborative_recommendations(self, user_id, num_rec=5):
        """
        Generate collaborative filtering recommendations
        
        Args:
            user_id: User ID to get recommendations for
            num_rec: Number of recommendations to generate
            
        Returns:
            pandas.Series: Recommended items with scores
        """
        try:
            model = NearestNeighbors(
                metric='cosine',
                algorithm='brute',
                n_neighbors=min(6, len(self.matrix.indptr)-1)
            )
            model.fit(self.matrix)
            
            user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
            distances, indices = model.kneighbors(user_vector)
            
            similar_users = indices.flatten()[1:]
            rec_items = self.user_item_matrix.iloc[similar_users].mean(0).sort_values(ascending=False)
            
            seen_items = self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index
            
            return rec_items.drop(seen_items).head(num_rec)
            
        except (KeyError, ValueError) as e:
            print(f"Error in collaborative recommendations for user {user_id}: {str(e)}")
            return pd.Series()

    def content_based_recommendations(self, query, num_rec=5):
        """
        Generate content-based recommendations
        
        Args:
            query: Search query for content matching
            num_rec: Number of recommendations to generate
            
        Returns:
            tuple: (recommendations DataFrame, similarity scores)
        """
        try:
            query_vec = self.tfidf.transform([query])
            sim_scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()
            sim_indices = sim_scores.argsort()[::-1][:num_rec*2]
            return self.items.iloc[sim_indices], sim_scores[sim_indices]
        except Exception as e:
            print(f"Error in content recommendations: {str(e)}")
            return pd.DataFrame(), []

    def hybrid_recommendations(self, user_id, query=None, num_rec=5, eval_user=False):
        """
        Generate hybrid recommendations combining multiple approaches
        
        Args:
            user_id: User ID to get recommendations for
            query: Optional search query for content matching
            num_rec: Number of recommendations to generate
            eval_user: Whether this is for evaluation
            
        Returns:
            pandas.DataFrame: Recommended items with scores
        """
        try:
            is_new_user = user_id not in self.user_item_matrix.index
            
            if is_new_user:
                return self._new_user_recommendations(query, num_rec)
            else:
                return self._existing_user_recommendations(
                    user_id, query, num_rec, eval_user
                )
                
        except Exception as e:
            print(f"Error in hybrid recommendations: {str(e)}")
            return self.items.sort_values('weighted_rating', ascending=False).head(num_rec)

    def _new_user_recommendations(self, query, num_rec):
        """Handle recommendations for new users"""
        if not query:
            return self.items.sort_values('weighted_rating', ascending=False).head(num_rec)
            
        content_rec, content_scores = self.content_based_recommendations(query, num_rec*2)
        if content_rec.empty:
            return self.items.sort_values('weighted_rating', ascending=False).head(num_rec)
            
        content_rec = content_rec.copy()
        content_rec['combined_score'] = 0.7*content_scores + 0.3*content_rec['weighted_rating']
        return content_rec.sort_values('combined_score', ascending=False).head(num_rec)

    def _existing_user_recommendations(self, user_id, query, num_rec, eval_user):
        """Handle recommendations for existing users"""
        collab_rec = self.collaborative_recommendations(user_id, num_rec)
        seen_items = self.user_item_matrix.loc[user_id][
            self.user_item_matrix.loc[user_id] > 0
        ].index
        
        content_rec, content_scores = (
            self.content_based_recommendations(query, num_rec) 
            if query else (pd.DataFrame(), [])
        )
        
        # Combine recommendations
        combined_items = pd.concat([
            collab_rec.rename('score'),
            pd.Series(
                content_rec['itemId'].tolist() if not content_rec.empty else [],
                name='itemId'
            )
        ], ignore_index=True).drop_duplicates().tolist()
        
        hybrid_df = self.items[self.items['itemId'].isin(combined_items)]
        hybrid_df = hybrid_df[~hybrid_df['itemId'].isin(seen_items)]
        
        # Merge scores
        hybrid_df = hybrid_df.merge(
            collab_rec.rename('collab_score'),
            left_on='itemId',
            right_index=True,
            how='left'
        )
        
        if query and not content_rec.empty:
            content_scores_df = pd.DataFrame({
                'itemId': content_rec['itemId'],
                'content_score': content_scores
            })
            hybrid_df = hybrid_df.merge(content_scores_df, on='itemId', how='left')
            hybrid_df['content_score'] = hybrid_df['content_score'].fillna(0)
        else:
            hybrid_df['content_score'] = 0
        
        # Normalize scores
        scaler = MinMaxScaler()
        score_columns = ['collab_score', 'content_score', 'weighted_rating']
        hybrid_df[score_columns] = scaler.fit_transform(
            hybrid_df[score_columns].fillna(0)
        )
        
        # Calculate final score
        hybrid_df['final_score'] = (
            0.5 * hybrid_df['collab_score'] + 
            0.3 * hybrid_df['content_score'] + 
            0.2 * hybrid_df['weighted_rating']
        )
        
        return hybrid_df.sort_values('final_score', ascending=False).head(num_rec)