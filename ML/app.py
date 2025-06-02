from flask import Flask, render_template, jsonify, request
from numba_ml import HousingAnalyzer
import json
import os
import sys

# Create necessary directories
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

app = Flask(__name__)

# Initialize the analyzer
analyzer = HousingAnalyzer("california_housing_train.csv")

# Ensure model is trained before accessing features requiring it
@app.before_request
def before_request():
    if request.endpoint in ['shap_analysis', 'feature_importance', 'get_importance_plot', 
                           'get_shap_plot', 'get_prediction_plot', 'model_training']:
        # Train model if not already trained
        if analyzer.model is None:
            analyzer.train_model()

@app.route('/')
def index():
    """Render the main dashboard"""
    dataset_info = analyzer.get_dataset_info()
    return render_template('index.html', dataset_info=dataset_info)

@app.route('/data-exploration')
def data_exploration():
    """Render data exploration page"""
    dataset_info = analyzer.get_dataset_info()
    summary_stats = analyzer.get_summary_stats()
    missing_values = analyzer.get_missing_values()
    return render_template(
        'data_exploration.html', 
        dataset_info=dataset_info, 
        summary_stats=summary_stats,
        missing_values=missing_values
    )

@app.route('/api/correlation-plot')
def get_correlation_plot():
    """API endpoint to get correlation plot JSON"""
    plot_json = analyzer.create_correlation_plot()
    return plot_json

@app.route('/api/target-distribution')
def get_target_distribution():
    """API endpoint to get target distribution plot JSON"""
    plot_json = analyzer.create_target_distribution()
    return plot_json

@app.route('/api/feature-distributions')
def get_feature_distributions():
    """API endpoint to get feature distributions plot JSON"""
    plot_json = analyzer.create_feature_distributions()
    return plot_json

@app.route('/api/scatter-matrix')
def get_scatter_matrix():
    """API endpoint to get scatter matrix plot JSON"""
    plot_json = analyzer.create_scatter_matrix()
    return plot_json

@app.route('/api/geo-plot')
def get_geo_plot():
    """API endpoint to get geographical plot JSON"""
    plot_json = analyzer.create_geo_plot()
    return plot_json

@app.route('/feature-engineering')
def feature_engineering():
    """Render feature engineering page"""
    # First ensure feature engineering is performed
    feature_eng_results = analyzer.perform_feature_engineering()
    return render_template('feature_engineering.html', feature_eng_results=feature_eng_results)

@app.route('/model-training')
def model_training():
    """Render model training and evaluation page"""
    model_results = analyzer.train_model()
    return render_template('model_training.html', model_results=model_results)

@app.route('/api/prediction-plot')
def get_prediction_plot():
    """API endpoint to get prediction plot JSON"""
    plot_json = analyzer.create_prediction_plot()
    return plot_json

@app.route('/feature-importance')
def feature_importance():
    """Render feature importance page"""
    importance_results = analyzer.get_feature_importance()
    return render_template('feature_importance.html', importance_results=importance_results)

@app.route('/api/importance-plot')
def get_importance_plot():
    """API endpoint to get feature importance plot JSON"""
    importance_results = analyzer.get_feature_importance()
    return importance_results["plot"]

@app.route('/shap-analysis')
def shap_analysis():
    """Render SHAP analysis page"""
    shap_results = analyzer.calculate_shap_values()
    return render_template('shap_analysis.html', shap_results=shap_results)

@app.route('/api/shap-plot')
def get_shap_plot():
    """API endpoint to get SHAP plot JSON"""
    shap_results = analyzer.calculate_shap_values()
    return shap_results["plot"]

if __name__ == '__main__':
    app.run(debug=True)
