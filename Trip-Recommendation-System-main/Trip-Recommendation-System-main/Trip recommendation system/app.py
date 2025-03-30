import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, jsonify
from io import BytesIO
import base64
import json
from recommendation_engine import HybridRecommender
from evaluation import RecommenderEvaluator

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

from flask.json.provider import DefaultJSONProvider
import numpy as np

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)

# Set custom JSON provider in Flask app
app.json = CustomJSONProvider(app)




# Initialize the recommender and evaluator
recommender = HybridRecommender()
evaluator = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('dashboard.html')

@app.route('/metrics')#evaluation.py
def metrics():
    """Render the metrics page"""
    return render_template('metrics.html')

@app.route('/index')
def index_page():
    """Render the main recommendation page"""
    return render_template('index.html')


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import traceback



class CustomJSONEncoder(json.JSONEncoder):
    """ Custom encoder to handle NumPy and Pandas types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder  # Set custom JSON encoder globally

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Support both form-data and JSON requests
        data = request.json if request.is_json else request.form.to_dict()

        if not data:
            return jsonify({'recommendations': [], 'error': "No data received"}), 400

        user_type = data.get('user_type', 'new')
        interests = data.get('interests', '')

        if not interests:
            return jsonify({'recommendations': [], 'error': "Interests field is required"}), 400

        # Validate user_id for existing users
        try:
            user_id = int(data.get('user_id', -1)) if user_type == 'existing' else -1
        except ValueError:
            return jsonify({'recommendations': [], 'error': "User ID must be a number"}), 400

        print(f"Processing recommendation request: user_type={user_type}, user_id={user_id}, interests={interests}")

        # Fetch recommendations
        try:
            recommendations = recommender.hybrid_recommendations(user_id=user_id, query=interests)

            if recommendations is None:
                return jsonify({'recommendations': [], 'error': "No recommendations returned"}), 200

            if isinstance(recommendations, pd.DataFrame):
                recommendations_json = recommendations.replace({np.nan: None}).to_dict(orient='records')
            else:
                recommendations_json = list(recommendations) if hasattr(recommendations, '__iter__') else [{"result": str(recommendations)}]

            return jsonify({'recommendations': recommendations_json, 'error': None})

        except Exception as e:
            traceback.print_exc()
            return jsonify({'recommendations': [], 'error': f"Recommendation engine error: {str(e)}"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({'recommendations': [], 'error': f"Server error: {str(e)}"}), 500

@app.route("/trending_destinations")
def trending_destinations():
    try:
        # Example: Fetch from a travel API (Replace with real API key)
        api_url = "https://api.example.com/trending-destinations"
        headers = {"Authorization": "Bearer YOUR_API_KEY"}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            destinations = response.json()  # Assuming API returns JSON
        else:
            destinations = [
                {"name": "Bali, Indonesia", "description": "Beautiful beaches and vibrant culture."},
                {"name": "Kyoto, Japan", "description": "Ancient temples and traditional gardens."},
                {"name": "Santorini, Greece", "description": "Whitewashed buildings and blue seas."},
            ]

        return jsonify({"success": True, "data": destinations})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Add these imports at the top of app.py
from trending import TrendingEngine
import json
import os

# Initialize the trending engine
trending_engine = TrendingEngine()

# Add this route to your app.py file
@app.route('/api/trending')
def get_trending():
    """API endpoint to get trending destinations."""
    trending = trending_engine.get_trending_destinations()
    return json.dumps(trending)

# Optional: Schedule trending updates
def schedule_trending_updates():
    """Set up a scheduler to update trending data periodically."""
    from apscheduler.schedulers.background import BackgroundScheduler
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=trending_engine.update_trending_data,
        trigger="interval",
        hours=6
    )
    scheduler.start()
    
    # Shut down the scheduler when the app is stopping
    import atexit
    atexit.register(lambda: scheduler.shutdown())

# Add this to your app initialization code
# schedule_trending_updates()  # Uncomment when ready to use


@app.route('/run_evaluation', methods=['POST'])
def run_evaluation():
    """Run the evaluation and return results"""
    global recommender, evaluator
    
    # Get parameters from the form
    k = int(request.form.get('k', 10))
    test_size = float(request.form.get('test_size', 0.2))
    
    try:
        # Initialize if not already done
        if recommender is None:
            recommender = HybridRecommender()
            
        if evaluator is None:
            evaluator = RecommenderEvaluator(recommender=recommender, test_size=test_size)
            
        # Run the evaluation
        results = evaluator.evaluate(k=k)
        
        # Add diversity and novelty metrics
        diversity_results = evaluator.evaluate_diversity(k=k)
        novelty_results = evaluator.evaluate_novelty(k=k)
        
        # Combine all results
        all_results = {**results, **diversity_results, **novelty_results}
        
        # Convert catalog_coverage to JSON serializable format
        if 'catalog_coverage' in all_results:
            all_results['catalog_coverage'] = len(all_results['catalog_coverage'])
        
        # Generate visualizations
        visualization_data = generate_visualizations(all_results)
        
        # Prepare data for the template
        metrics_data = {
            'precision': round(all_results['avg_precision'], 4),
            'recall': round(all_results['avg_recall'], 4),
            'f1_score': round(all_results['f1_score'], 4),
            'ndcg': round(all_results['avg_ndcg'], 4),
            'user_coverage': round(all_results['user_coverage'], 4),
            'catalog_coverage': round(all_results['catalog_coverage_pct'], 4),
            'diversity': round(all_results.get('avg_diversity', 0), 4),
            'novelty': round(all_results.get('avg_novelty', 0), 4)
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Evaluation completed successfully',
            'metrics': metrics_data,
            'visualizations': visualization_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error running evaluation: {str(e)}'
        })

@app.route('/compare_methods', methods=['POST'])
def compare_methods():
    """Compare different recommendation methods"""
    global recommender, evaluator
    
    # Get parameters from the form
    k = int(request.form.get('k', 10))
    num_users = int(request.form.get('num_users', 20))
    
    try:
        # Initialize if not already done
        if recommender is None:
            recommender = HybridRecommender()
            
        if evaluator is None:
            evaluator = RecommenderEvaluator(recommender=recommender)
        
        # Run the comparison
        comparison_results = evaluator.compare_recommendation_methods(num_users=num_users, k=k)
        
        # Generate visualizations for comparison
        visualization_data = generate_comparison_visualizations(comparison_results)
        
        # Prepare data for the template
        comparison_data = {
            'avg_collab_count': round(comparison_results['avg_metrics'].get('avg_collab_count', 0), 2),
            'avg_content_count': round(comparison_results['avg_metrics'].get('avg_content_count', 0), 2),
            'avg_hybrid_count': round(comparison_results['avg_metrics'].get('avg_hybrid_count', 0), 2),
            'avg_overlap_count': round(comparison_results['avg_metrics'].get('avg_overlap_count', 0), 2),
            'avg_overlap_pct': round(comparison_results['avg_metrics'].get('avg_overlap_pct', 0) * 100, 2),
            'jaccard_similarities': {k: round(v, 4) for k, v in comparison_results['jaccard_similarities'].items()}
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Method comparison completed successfully',
            'comparison': comparison_data,
            'visualizations': visualization_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error comparing methods: {str(e)}'
        })

def generate_visualizations(results):
    """Generate visualizations for the evaluation results"""
    visualization_data = {}
    
    # Create a figure for metrics
    plt.figure(figsize=(10, 6))
    metrics = ['avg_precision', 'avg_recall', 'f1_score', 'avg_ndcg']
    values = [results[metric] for metric in metrics]
    labels = ['Precision', 'Recall', 'F1 Score', 'NDCG']
    
    plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1)
    plt.title('Recommendation Quality Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    visualization_data['metrics'] = base64.b64encode(image_png).decode('utf-8')
    
    # Create a figure for coverage
    plt.figure(figsize=(8, 5))
    coverage_metrics = ['user_coverage', 'catalog_coverage_pct']
    coverage_values = [results[metric] for metric in coverage_metrics]
    coverage_labels = ['User Coverage', 'Catalog Coverage']
    
    plt.bar(coverage_labels, coverage_values, color=['#9467bd', '#8c564b'])
    plt.ylim(0, 1)
    plt.title('Coverage Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    visualization_data['coverage'] = base64.b64encode(image_png).decode('utf-8')
    
    # Create a figure for diversity and novelty if available
    if 'avg_diversity' in results and 'avg_novelty' in results:
        plt.figure(figsize=(8, 5))
        diversity_metrics = ['avg_diversity', 'avg_novelty']
        diversity_values = [results[metric] for metric in diversity_metrics]
        diversity_labels = ['Diversity', 'Novelty']
        
        plt.bar(diversity_labels, diversity_values, color=['#1f77b4', '#ff7f0e'])
        plt.title('Diversity and Novelty Metrics')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        visualization_data['diversity_novelty'] = base64.b64encode(image_png).decode('utf-8')
    
    return visualization_data

def generate_comparison_visualizations(comparison_results):
    """Generate visualizations for method comparison"""
    visualization_data = {}
    
    # Create a figure for recommendation counts
    plt.figure(figsize=(10, 6))
    method_names = list(comparison_results['recommendation_counts'].keys())
    method_counts = [np.mean(comparison_results['recommendation_counts'][method]) 
                     if comparison_results['recommendation_counts'][method] else 0 
                     for method in method_names]
    
    plt.bar(method_names, method_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Average Number of Recommendations by Method')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    visualization_data['method_counts'] = base64.b64encode(image_png).decode('utf-8')
    
    # Create a figure for Jaccard similarities
    plt.figure(figsize=(10, 6))
    jaccard_keys = list(comparison_results['jaccard_similarities'].keys())
    jaccard_values = [comparison_results['jaccard_similarities'][key] for key in jaccard_keys]
    
    plt.bar(jaccard_keys, jaccard_values, color=['#d62728', '#9467bd', '#8c564b'])
    plt.title('Jaccard Similarity Between Methods')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    visualization_data['jaccard_similarities'] = base64.b64encode(image_png).decode('utf-8')
    
    return visualization_data

if __name__ == '__main__':
    app.run(debug=True)