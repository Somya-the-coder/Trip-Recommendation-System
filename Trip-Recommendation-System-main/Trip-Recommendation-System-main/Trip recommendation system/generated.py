import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Set the style for better visualizations
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Load datasets
def load_data():
    # Load your datasets - adjust file paths as needed
    items_df = pd.read_csv('items.csv')
    users_df = pd.read_csv('users.csv')
    ratings_df = pd.read_csv('ratings.csv')
    
    print(f"Loaded {len(items_df)} items, {len(users_df)} users, and {len(ratings_df)} ratings")
    return items_df, users_df, ratings_df

# Create evaluation results - assuming you've already run the evaluator
def generate_sample_results():
    """
    This function simulates evaluation results if you haven't run the actual evaluator yet.
    Replace this with your actual evaluation results when available.
    """
    return {
        'evaluation_stats': {
            'total_users': 1000,
            'valid_users': 850,
            'min_ratings_threshold': 5,
            'train_size': 8500,
            'test_size': 2125,
            'avg_ratings_per_user': 10
        },
        'precision@5': 0.32,
        'recall@5': 0.28,
        'ndcg@5': 0.41,
        'precision@5_std': 0.15,
        'recall@5_std': 0.14,
        'precision@10': 0.29,
        'recall@10': 0.38,
        'ndcg@10': 0.39,
        'precision@10_std': 0.13,
        'recall@10_std': 0.16,
        'coverage': 0.72
    }

# Plotting functions
def plot_metric_comparison(metrics, k_values):
    """Plot precision, recall and NDCG for different k values"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.25
    x = np.arange(len(k_values))
    
    precision_values = [metrics[f'precision@{k}'] for k in k_values]
    recall_values = [metrics[f'recall@{k}'] for k in k_values]
    ndcg_values = [metrics[f'ndcg@{k}'] for k in k_values]
    
    # Add bars
    ax.bar(x - width, precision_values, width, label='Precision', color='#1f77b4', alpha=0.8)
    ax.bar(x, recall_values, width, label='Recall', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width, ndcg_values, width, label='NDCG', color='#2ca02c', alpha=0.8)
    
    # Add error bars
    precision_std = [metrics[f'precision@{k}_std'] for k in k_values]
    recall_std = [metrics[f'recall@{k}_std'] for k in k_values]
    
    ax.errorbar(x - width, precision_values, yerr=precision_std, fmt='none', ecolor='black', capsize=5)
    ax.errorbar(x, recall_values, yerr=recall_std, fmt='none', ecolor='black', capsize=5)
    
    # Customize plot
    ax.set_ylabel('Score')
    ax.set_title('Recommendation Performance Metrics at Different k Values')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(precision_values):
        ax.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    for i, v in enumerate(recall_values):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    for i, v in enumerate(ndcg_values):
        ax.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_item_category_distribution(items_df):
    """Plot distribution of items by category"""
    plt.figure(figsize=(10, 6))
    
    # Count items by category
    category_counts = items_df['category'].value_counts()
    
    # Create horizontal bar chart
    ax = sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
    
    # Add counts as labels
    for i, v in enumerate(category_counts.values):
        ax.text(v + 0.5, i, str(v), va='center')
    
    plt.title('Distribution of Items by Category')
    plt.xlabel('Number of Items')
    plt.ylabel('Category')
    plt.tight_layout()
    
    return plt.gcf()

def plot_rating_distribution(ratings_df):
    """Plot distribution of ratings"""
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    ax = sns.histplot(ratings_df['rating'], bins=5, kde=True)
    
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating Value')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()

def plot_user_activity(ratings_df):
    """Plot user activity (number of ratings per user)"""
    plt.figure(figsize=(12, 6))
    
    # Count ratings per user
    user_activity = ratings_df['userId'].value_counts()
    
    # Create histogram
    ax = sns.histplot(user_activity.values, bins=30, kde=True)
    
    plt.title('Distribution of User Activity')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()

def plot_item_popularity(ratings_df, items_df):
    """Plot item popularity (number of ratings per item)"""
    plt.figure(figsize=(12, 6))
    
    # Count ratings per item
    item_popularity = ratings_df['itemId'].value_counts().reset_index()
    item_popularity.columns = ['itemId', 'rating_count']
    
    # Merge with items to get categories
    if 'category' in items_df.columns:
        item_popularity = item_popularity.merge(
            items_df[['itemId', 'category']], 
            on='itemId', 
            how='left'
        )
        
        # Top 20 most popular items with their categories
        top_items = item_popularity.sort_values('rating_count', ascending=False).head(20)
        
        # Create bar chart
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='rating_count', y='itemId', data=top_items, hue='category', dodge=False)
        
        plt.title('Top 20 Most Popular Items')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Item ID')
        plt.tight_layout()
    else:
        # Just show top 20 most popular items
        top_items = item_popularity.sort_values('rating_count', ascending=False).head(20)
        
        # Create bar chart
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='rating_count', y='itemId', data=top_items)
        
        plt.title('Top 20 Most Popular Items')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Item ID')
        plt.tight_layout()
    
    return plt.gcf()

def plot_evaluation_stats(metrics):
    """Plot evaluation statistics"""
    stats = metrics['evaluation_stats']
    
    # Create a pie chart for user stats
    plt.figure(figsize=(18, 6))
    
    # First subplot: User stats
    plt.subplot(1, 3, 1)
    labels = ['Valid Users', 'Excluded Users']
    sizes = [stats['valid_users'], stats['total_users'] - stats['valid_users']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('User Coverage in Evaluation')
    
    # Second subplot: Train/Test split
    plt.subplot(1, 3, 2)
    labels = ['Training Set', 'Test Set']
    sizes = [stats['train_size'], stats['test_size']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ffcc99'])
    plt.title('Train/Test Split')
    
    # Third subplot: Recommendation coverage bar
    plt.subplot(1, 3, 3)
    coverage = metrics.get('coverage', 0)
    plt.barh(['Coverage'], [coverage], color='#c2c2f0')
    plt.barh(['Coverage'], [1], color='#f0c2c2', alpha=0.3)
    plt.xlim(0, 1)
    plt.title('Item Catalog Coverage')
    for i, v in enumerate([coverage]):
        plt.text(v/2, i, f'{v*100:.1f}%', va='center', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return plt.gcf()

def plot_metrics_radar(metrics, k_values):
    """Create a radar chart for metrics visualization"""
    # Prepare data
    categories = ['Precision', 'Recall', 'NDCG']
    
    # Collect metrics for each k
    k_metrics = []
    for k in k_values:
        k_metrics.append([
            metrics[f'precision@{k}'],
            metrics[f'recall@{k}'],
            metrics[f'ndcg@{k}']
        ])
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Spider plot for each k value
    for i, k in enumerate(k_values):
        values = k_metrics[i]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'k={k}')
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Set y axis limits
    ax.set_ylim(0, max([max(metrics) for metrics in k_metrics]) * 1.2)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Recommendation Performance Metrics Comparison', size=15, y=1.1)
    plt.tight_layout()
    
    return fig

def create_evaluation_dashboard(items_df, users_df, ratings_df, metrics):
    """Create all evaluation plots"""
    # Create metrics comparison plots
    k_values = [5, 10]  # Extract k values from metrics
    
    # Create and save plots
    plots = {}
    
    plots['metrics_comparison'] = plot_metric_comparison(metrics, k_values)
    plots['metrics_radar'] = plot_metrics_radar(metrics, k_values)
    plots['evaluation_stats'] = plot_evaluation_stats(metrics)
    plots['rating_distribution'] = plot_rating_distribution(ratings_df)
    plots['user_activity'] = plot_user_activity(ratings_df)
    
    if 'category' in items_df.columns:
        plots['item_category'] = plot_item_category_distribution(items_df)
    
    plots['item_popularity'] = plot_item_popularity(ratings_df, items_df)
    
    return plots

# Define a function to run the entire evaluation visualization pipeline
def run_recsys_evaluation_visualization():
    """Main function to run the recommendation system evaluation visualization"""
    try:
        # Load data
        items_df, users_df, ratings_df = load_data()
        
        # Create or load evaluation metrics
        # If you have already run the evaluator, load those results instead
        metrics = generate_sample_results()
        
        # Create evaluation dashboard
        plots = create_evaluation_dashboard(items_df, users_df, ratings_df, metrics)
        
        # Display all plots
        for name, plot in plots.items():
            plt.figure(plot.number)
            plt.tight_layout()
            plt.show()
            
        return "Evaluation visualization complete!"
        
    except Exception as e:
        print(f"Error in evaluation visualization: {str(e)}")
        return None

# Run the visualization if this script is executed directly
if __name__ == "__main__":
    run_recsys_evaluation_visualization()