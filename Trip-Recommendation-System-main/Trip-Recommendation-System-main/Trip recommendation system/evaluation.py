import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from recommendation_engine import HybridRecommender  # Import your recommender class

class RecommenderEvaluator:
    def __init__(self, recommender=None, ratings_df=None, test_size=0.2, random_state=42):
        """
        Initialize the evaluator with a recommender system and dataset
        
        Args:
            recommender: HybridRecommender instance (if None, will be created)
            ratings_df: DataFrame of ratings (if None, will be loaded)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Load recommender if not provided
        self.recommender = recommender if recommender else HybridRecommender()
        
        # Load data if not provided
        self.ratings_df = ratings_df if ratings_df is not None else pd.read_csv('ratings.csv')
        self.test_size = test_size
        self.random_state = random_state
        
    def train_test_split(self):
        """Split ratings into train and test sets"""
        # Split by user stratification to ensure all users have some data in both sets
        # This is important for evaluating how well the system works for existing users
        train, test = train_test_split(
            self.ratings_df, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.ratings_df['userId']
        )
        return train, test
    
    def calculate_mae(self, true_ratings, predicted_ratings):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(np.array(true_ratings) - np.array(predicted_ratings)))
    
    def calculate_rmse(self, true_ratings, predicted_ratings):
        """Calculate Root Mean Squared Error"""
        return np.sqrt(np.mean(np.square(np.array(true_ratings) - np.array(predicted_ratings))))
    
    def precision_recall_at_k(self, user_id, actual_items, k=10):
        """
        Calculate precision and recall at k for a user
        
        Args:
            user_id: User ID to evaluate
            actual_items: List of items the user has actually rated in test set
            k: Number of recommendations to consider
            
        Returns:
            tuple: (precision, recall) values
        """
        # Get top k recommendations for the user
        user_recs = self.recommender.hybrid_recommendations(user_id, num_rec=k, eval_user=True)
        recommended_items = set(user_recs['itemId'].tolist())
        
        # Calculate metrics
        relevant_items = set(actual_items)
        true_positives = len(relevant_items.intersection(recommended_items))
        
        precision = true_positives / len(recommended_items) if recommended_items else 0
        recall = true_positives / len(relevant_items) if relevant_items else 0
        
        return precision, recall
    
    def calculate_ndcg(self, recommended_items, relevant_items, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of items the user has actually rated
            k: Number of recommendations to consider
            
        Returns:
            float: NDCG value
        """
        # Get DCG (Discounted Cumulative Gain)
        dcg = 0
        for i, item_id in enumerate(recommended_items[:k]):
            if item_id in relevant_items:
                # Using binary relevance (1 if relevant)
                dcg += 1 / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # Get IDCG (Ideal DCG)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
        
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate(self, k=10):
        """
        Run full evaluation
        
        Args:
            k: Number of recommendations to consider
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("Starting recommender system evaluation...")
        
        # Split data
        train, test = self.train_test_split()
        print(f"Split data into {len(train)} training samples and {len(test)} test samples")
        
        # Update recommender with training data
        self.recommender.train_ratings = train
        self.recommender.preprocess_data()
        print("Recommender system updated with training data")
        
        # Setup metrics
        results = {
            'precision': [],
            'recall': [],
            'ndcg': [],
            'user_coverage': 0,
            'catalog_coverage': set()
        }
        
        # Get unique users in test set
        test_users = test['userId'].unique()
        covered_users = 0
        
        print(f"Evaluating recommendations for {len(test_users)} users...")
        for i, user_id in enumerate(test_users):
            if i % 50 == 0 and i > 0:
                print(f"Processed {i}/{len(test_users)} users")
                
            # Get items the user actually rated in test set
            actual_items = test[test['userId'] == user_id]['itemId'].tolist()
            
            try:
                # Get recommendations for this user
                user_recs = self.recommender.hybrid_recommendations(user_id, num_rec=k, eval_user=True)
                
                if not user_recs.empty:
                    covered_users += 1
                    recommended_items = user_recs['itemId'].tolist()
                    
                    # Add to catalog coverage
                    results['catalog_coverage'].update(recommended_items)
                    
                    # Calculate precision and recall
                    precision, recall = self.precision_recall_at_k(user_id, actual_items, k)
                    results['precision'].append(precision)
                    results['recall'].append(recall)
                    
                    # Calculate NDCG
                    ndcg = self.calculate_ndcg(recommended_items, actual_items, k)
                    results['ndcg'].append(ndcg)
            
            except Exception as e:
                print(f"Error evaluating user {user_id}: {str(e)}")
        
        print("Calculating final metrics...")
        # Calculate averages and coverage
        results['avg_precision'] = sum(results['precision']) / len(results['precision']) if results['precision'] else 0
        results['avg_recall'] = sum(results['recall']) / len(results['recall']) if results['recall'] else 0
        results['avg_ndcg'] = sum(results['ndcg']) / len(results['ndcg']) if results['ndcg'] else 0
        results['f1_score'] = 2 * (results['avg_precision'] * results['avg_recall']) / (results['avg_precision'] + results['avg_recall']) if (results['avg_precision'] + results['avg_recall']) > 0 else 0
        
        results['user_coverage'] = covered_users / len(test_users)
        results['catalog_coverage_pct'] = len(results['catalog_coverage']) / len(self.recommender.items)
        
        print("Evaluation complete!")
        return results
    
    def evaluate_diversity(self, num_users=20, k=10):
        """
        Evaluate diversity of recommendations
        
        Args:
            num_users: Number of users to sample for diversity evaluation
            k: Number of recommendations to consider
            
        Returns:
            dict: Dictionary containing diversity metrics
        """
        print("Evaluating recommendation diversity...")
        
        # Sample users
        all_users = self.recommender.user_item_matrix.index.tolist()
        if len(all_users) > num_users:
            sample_users = random.sample(all_users, num_users)
        else:
            sample_users = all_users
        
        intra_list_diversity = []
        category_diversity = []
        
        for user_id in sample_users:
            try:
                # Get recommendations for this user
                user_recs = self.recommender.hybrid_recommendations(user_id, num_rec=k)
                
                if len(user_recs) <= 1:
                    continue
                
                # Get item IDs
                rec_item_ids = user_recs['itemId'].tolist()
                
                # Get content features for these items
                items_idx = [
                    self.recommender.items[self.recommender.items['itemId'] == item_id].index[0] 
                    for item_id in rec_item_ids 
                    if item_id in self.recommender.items['itemId'].values
                ]
                
                if len(items_idx) <= 1:
                    continue
                
                # Get TF-IDF vectors for these items
                item_vectors = self.recommender.tfidf_matrix[items_idx]
                
                # Calculate pairwise cosine similarity
                sim_matrix = cosine_similarity(item_vectors)
                
                # Calculate diversity (1 - average similarity)
                # Remove diagonal elements (self-similarity)
                np.fill_diagonal(sim_matrix, 0)
                avg_sim = sim_matrix.sum() / (sim_matrix.shape[0] * (sim_matrix.shape[0] - 1))
                diversity = 1 - avg_sim
                
                intra_list_diversity.append(diversity)
                
                # Category diversity (if available)
                rec_items_df = self.recommender.items[self.recommender.items['itemId'].isin(rec_item_ids)]
                if 'category' in rec_items_df.columns:
                    unique_categories = rec_items_df['category'].nunique()
                    category_diversity.append(unique_categories / len(rec_items_df))
            
            except Exception as e:
                print(f"Error calculating diversity for user {user_id}: {str(e)}")
        
        results = {
            'avg_diversity': sum(intra_list_diversity) / len(intra_list_diversity) if intra_list_diversity else 0
        }
        
        if category_diversity:
            results['category_diversity'] = sum(category_diversity) / len(category_diversity)
            
        print("Diversity evaluation complete!")
        return results
    
    def evaluate_novelty(self, num_users=20, k=10):
        """
        Evaluate novelty of recommendations
        
        Args:
            num_users: Number of users to sample for novelty evaluation
            k: Number of recommendations to consider
            
        Returns:
            dict: Dictionary containing novelty metrics
        """
        print("Evaluating recommendation novelty...")
        
        # Calculate popularity of each item (how many users rated it)
        item_popularity = self.ratings_df['itemId'].value_counts()
        total_ratings = len(self.ratings_df)
        
        # Convert to probability
        item_prob = item_popularity / total_ratings
        
        # Calculate the self-information (novelty) for each item: -log2(popularity)
        item_novelty = item_prob.apply(lambda x: -np.log2(x))
        
        # Sample users
        all_users = self.recommender.user_item_matrix.index.tolist()
        if len(all_users) > num_users:
            sample_users = random.sample(all_users, num_users)
        else:
            sample_users = all_users
        
        novelty_scores = []
        
        for user_id in sample_users:
            try:
                # Get recommendations for this user
                user_recs = self.recommender.hybrid_recommendations(user_id, num_rec=k)
                
                if user_recs.empty:
                    continue
                
                # Get item IDs
                rec_item_ids = user_recs['itemId'].tolist()
                
                # Calculate average novelty for this user's recommendations
                user_items_novelty = [
                    item_novelty.get(item_id, 0)  # Default to 0 if item not found
                    for item_id in rec_item_ids
                    if item_id in item_novelty.index
                ]
                
                if user_items_novelty:
                    novelty_scores.append(sum(user_items_novelty) / len(user_items_novelty))
            
            except Exception as e:
                print(f"Error calculating novelty for user {user_id}: {str(e)}")
        
        results = {
            'avg_novelty': sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
        }
        
        print("Novelty evaluation complete!")
        return results
    
    def compare_recommendation_methods(self, num_users=20, k=10):
        """
        Compare different recommendation methods
        
        Args:
            num_users: Number of users to sample for comparison
            k: Number of recommendations to consider
            
        Returns:
            dict: Dictionary containing comparison metrics
        """
        print("Comparing recommendation methods...")
        
        methods = {
            "Collaborative": [],
            "Content-Based": [],
            "Hybrid": []
        }
        
        # Track items recommended by each method for overlap analysis
        method_recommendations = {
            "Collaborative": set(),
            "Content-Based": set(),
            "Hybrid": set()
        }
        
        # Track performance per user
        user_metrics = []
        
        # Sample users
        all_users = self.recommender.user_item_matrix.index.tolist()
        if len(all_users) > num_users:
            sample_users = random.sample(all_users, num_users)
        else:
            sample_users = all_users
        
        for user_id in sample_users:
            # Get user info for content query
            user_info = self.recommender.users[self.recommender.users['userId'] == user_id]
            query = user_info['Category'].values[0] if not user_info.empty and 'Category' in user_info.columns else None
            
            user_record = {"userId": user_id}
            
            try:
                # Collaborative filtering
                collab_recs = self.recommender.collaborative_recommendations(user_id, k)
                if isinstance(collab_recs, pd.DataFrame) and not collab_recs.empty:
                    collab_items = collab_recs.index.tolist()
                    method_recommendations["Collaborative"].update(collab_items)
                    methods["Collaborative"].append(len(collab_items))
                    user_record["collab_count"] = len(collab_items)
                else:
                    user_record["collab_count"] = 0
                
                # Content-based
                if query:
                    content_recs, _ = self.recommender.content_based_recommendations(query, k)
                    if not content_recs.empty:
                        content_items = content_recs['itemId'].tolist()
                        method_recommendations["Content-Based"].update(content_items)
                        methods["Content-Based"].append(len(content_items))
                        user_record["content_count"] = len(content_items)
                    else:
                        user_record["content_count"] = 0
                else:
                    user_record["content_count"] = 0
                
                # Hybrid
# Hybrid
                hybrid_recs = self.recommender.hybrid_recommendations(user_id, num_rec=k)
                if not hybrid_recs.empty:
                    hybrid_items = hybrid_recs['itemId'].tolist()
                    method_recommendations["Hybrid"].update(hybrid_items)
                    methods["Hybrid"].append(len(hybrid_items))
                    user_record["hybrid_count"] = len(hybrid_items)
                else:
                    user_record["hybrid_count"] = 0
                
                # Compare overlaps
                if user_record["collab_count"] > 0 and user_record["content_count"] > 0:
                    collab_items = set(collab_recs.index.tolist())
                    content_items = set(content_recs['itemId'].tolist())
                    overlap = len(collab_items.intersection(content_items))
                    user_record["overlap_count"] = overlap
                    user_record["overlap_pct"] = overlap / len(collab_items.union(content_items)) if len(collab_items.union(content_items)) > 0 else 0
                
                user_metrics.append(user_record)
            
            except Exception as e:
                print(f"Error comparing methods for user {user_id}: {str(e)}")
        
        # Calculate recommendation method metrics
        avg_metrics = {
            "avg_collab_count": sum([u["collab_count"] for u in user_metrics]) / len(user_metrics) if user_metrics else 0,
            "avg_content_count": sum([u["content_count"] for u in user_metrics if "content_count" in u]) / len(user_metrics) if user_metrics else 0,
            "avg_hybrid_count": sum([u["hybrid_count"] for u in user_metrics]) / len(user_metrics) if user_metrics else 0,
        }
        
        # Calculate overlap metrics
        overlap_metrics = [u for u in user_metrics if "overlap_count" in u]
        if overlap_metrics:
            avg_metrics["avg_overlap_count"] = sum([u["overlap_count"] for u in overlap_metrics]) / len(overlap_metrics)
            avg_metrics["avg_overlap_pct"] = sum([u["overlap_pct"] for u in overlap_metrics]) / len(overlap_metrics)
        
        # Calculate Jaccard similarity between recommendation methods
        all_methods = list(method_recommendations.keys())
        jaccard_similarities = {}
        for i, method1 in enumerate(all_methods):
            for method2 in all_methods[i+1:]:
                intersection = len(method_recommendations[method1].intersection(method_recommendations[method2]))
                union = len(method_recommendations[method1].union(method_recommendations[method2]))
                similarity = intersection / union if union > 0 else 0
                jaccard_similarities[f"{method1}_{method2}"] = similarity
        
        # Combine results
        results = {
            "recommendation_counts": methods,
            "avg_metrics": avg_metrics,
            "jaccard_similarities": jaccard_similarities,
            "user_metrics": user_metrics
        }
        
        print("Method comparison complete!")
        return results
    
    def visualize_results(self, results):
        """
        Visualize evaluation results
        
        Args:
            results: Dictionary containing evaluation metrics
        """
        print("Generating visualizations...")
        
        # Set up plot style
        plt.style.use('seaborn-whitegrid')
        
        # Create a figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precision, Recall, F1
        axes[0, 0].bar(['Precision', 'Recall', 'F1 Score'], 
                    [results['avg_precision'], results['avg_recall'], results['f1_score']],
                    color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Recommendation Quality Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        # NDCG
        axes[0, 1].bar(['NDCG'], [results['avg_ndcg']], color='#d62728')
        axes[0, 1].set_title('Normalized Discounted Cumulative Gain')
        axes[0, 1].set_ylim(0, 1)
        
        # Coverage
        axes[1, 0].bar(['User Coverage', 'Catalog Coverage'], 
                    [results['user_coverage'], results['catalog_coverage_pct']],
                    color=['#9467bd', '#8c564b'])
        axes[1, 0].set_title('Coverage Metrics')
        axes[1, 0].set_ylim(0, 1)
        
        # Histogram of precision scores
        axes[1, 1].hist(results['precision'], bins=10, color='#e377c2', alpha=0.7)
        axes[1, 1].set_title('Distribution of Precision Scores')
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        plt.close()
        
        # If we have diversity and novelty results, plot those too
        if 'avg_diversity' in results and 'avg_novelty' in results:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].bar(['Diversity'], [results['avg_diversity']], color='#1f77b4')
            axes[0].set_title('Average Intra-list Diversity')
            axes[0].set_ylim(0, 1)
            
            axes[1].bar(['Novelty'], [results['avg_novelty']], color='#ff7f0e')
            axes[1].set_title('Average Novelty')
            
            plt.tight_layout()
            plt.savefig('diversity_novelty_results.png')
            plt.close()
        
        print("Visualizations saved!")