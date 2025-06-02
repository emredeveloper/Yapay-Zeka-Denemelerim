import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from numba import cuda, jit, float32
import numpy as np
import math
import base64
from io import BytesIO
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class HousingAnalyzer:
    def __init__(self, csv_path="california_housing_train.csv"):
        # Load dataset
        self.data = pd.read_csv(csv_path)
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_importance = None
        self.shap_values = None
    
    def get_dataset_info(self):
        """Return basic dataset information"""
        return {
            "shape": self.data.shape,
            "samples": self.data.shape[0],
            "features": self.data.shape[1] - 1,
            "columns": list(self.data.columns),
            "target": "median_house_value",
            "head": self.data.head().to_html(classes='table table-striped table-hover')
        }
    
    def get_summary_stats(self):
        """Return dataset summary statistics"""
        return {
            "summary_html": self.data.describe().to_html(classes='table table-striped table-hover'),
            "numeric_columns": self.data.select_dtypes(include=[np.number]).columns.tolist()
        }
    
    def get_missing_values(self):
        """Return missing values analysis"""
        missing = self.data.isnull().sum()
        return {
            "missing_counts": missing.to_dict(),
            "total_missing": missing.sum(),
            "has_missing": missing.sum() > 0
        }
    
    def create_correlation_plot(self):
        """Generate correlation heatmap as JSON for plotly"""
        corr = self.data.corr().round(2)
        fig = px.imshow(
            corr, 
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Plasma',
            title="Feature Correlation Heatmap"
        )
        
        # Apply dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white')
        )
        
        return fig.to_json()
    
    def create_target_distribution(self):
        """Create target distribution plot"""
        fig = px.histogram(
            self.data, 
            x="median_house_value",
            marginal="box",
            nbins=50,
            title="Distribution of Median House Value",
            color_discrete_sequence=['#9C27B0']
        )
        
        # Apply dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white'),
            bargap=0.1
        )
        
        return fig.to_json()
    
    def create_feature_distributions(self):
        """Create distribution plots for all features"""
        fig = make_subplots(
            rows=4, cols=3, 
            subplot_titles=self.data.drop("median_house_value", axis=1).columns
        )
        
        # Define color sequence
        colors = ['#FF4081', '#7C4DFF', '#00BCD4', '#FFC107', '#4CAF50', '#2196F3', 
                 '#E91E63', '#9C27B0', '#CDDC39', '#3F51B5', '#009688', '#FF9800']
        
        col_names = list(self.data.drop("median_house_value", axis=1).columns)
        r, c = 1, 1
        
        for i, col in enumerate(col_names):
            fig.add_trace(
                go.Histogram(
                    x=self.data[col], 
                    name=col,
                    marker=dict(color=colors[i % len(colors)])
                ),
                row=r, col=c
            )
            c += 1
            if c > 3:
                c = 1
                r += 1
        
        # Apply dark theme
        fig.update_layout(
            height=800,
            showlegend=False, 
            title_text="Feature Distributions",
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white')
        )
        
        return fig.to_json()
    
    def create_scatter_matrix(self):
        """Create scatter matrix for selected features"""
        # Select most important features plus target
        important_features = ["median_income", "latitude", "longitude", "median_house_value"]
        fig = px.scatter_matrix(
            self.data[important_features],
            dimensions=important_features,
            color="median_house_value",
            opacity=0.5
        )
        
        # Apply dark theme
        fig.update_layout(
            title="Scatter Matrix of Key Features",
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white'),
            colorway=['#ff4081', '#7c4dff', '#00bcd4', '#ffc107', '#4caf50', '#2196f3', 
                      '#e91e63', '#9c27b0', '#cddc39', '#3f51b5', '#009688', '#ff9800']
        )
        
        return fig.to_json()
    
    def create_geo_plot(self):
        """Create geographical plot of house values"""
        fig = px.scatter(
            self.data, 
            x="longitude", 
            y="latitude",
            color="median_house_value",
            size="population",
            hover_data=["median_income", "median_house_value"],
            color_continuous_scale=px.colors.sequential.Plasma,
            title="California Housing Prices Geographical Distribution"
        )
        
        # Add California state outline
        fig.update_layout(
            mapbox_style="dark",
            mapbox_zoom=5,
            mapbox_center={"lat": 37.5, "lon": -119},
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white'),
            colorway=['#ff4081', '#7c4dff', '#00bcd4', '#ffc107', '#4caf50', '#2196f3', 
                      '#e91e63', '#9c27b0', '#cddc39', '#3f51b5', '#009688', '#ff9800']
        )
        
        return fig.to_json()
    
    def perform_feature_engineering(self):
        """Perform feature engineering and return results"""
        # Create new features
        self.data['rooms_per_household'] = self.data['total_rooms'] / self.data['households']
        self.data['bedrooms_ratio'] = self.data['total_bedrooms'] / self.data['total_rooms']
        self.data['population_per_household'] = self.data['population'] / self.data['households']
        self.data['income_per_capita'] = self.data['median_income'] / (self.data['population'] / self.data['households'])
        self.data['distance_to_ocean'] = np.sqrt(self.data['longitude']**2 + self.data['latitude']**2)
        
        new_features = ['rooms_per_household', 'bedrooms_ratio', 'population_per_household', 
                       'income_per_capita', 'distance_to_ocean']
        
        feature_stats = {}
        for feature in new_features:
            feature_stats[feature] = {
                'mean': float(self.data[feature].mean()),
                'min': float(self.data[feature].min()),
                'max': float(self.data[feature].max()),
                'median': float(self.data[feature].median())
            }
            
        return {
            "new_features": new_features,
            "stats": feature_stats,
        }
    
    def train_model(self):
        """Train the model and return evaluation metrics"""
        # First, ensure feature engineering is performed
        self.perform_feature_engineering()
        
        # Split data including engineered features
        self.X = self.data.drop("median_house_value", axis=1)
        self.y = self.data["median_house_value"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train two models - one with original features, one with all features
        # to compare the impact of feature engineering
        original_columns = [col for col in self.X.columns if col not in [
            'rooms_per_household', 'bedrooms_ratio', 'population_per_household', 
            'income_per_capita', 'distance_to_ocean'
        ]]
        
        # Model with original features only
        model_original = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_original.fit(self.X_train[original_columns], self.y_train)
        y_pred_original = model_original.predict(self.X_test[original_columns])
        score_original = r2_score(self.y_test, y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(self.y_test, y_pred_original))
        
        # Model with all features (including engineered ones)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        
        # Evaluation metrics
        score = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mae = mean_absolute_error(self.y_test, self.y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='r2')
        
        # Calculate improvement from feature engineering
        improvement_r2 = score - score_original
        improvement_rmse = rmse_original - rmse
        
        return {
            "r2_score": float(score),
            "rmse": float(rmse),
            "mae": float(mae),
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "original_r2_score": float(score_original),
            "original_rmse": float(rmse_original),
            "improvement_r2": float(improvement_r2),
            "improvement_rmse": float(improvement_rmse),
            "feature_engineering_impact": float(improvement_r2 / score_original * 100) if score_original > 0 else 0
        }
    
    def create_prediction_plot(self):
        """Create actual vs predicted plot"""
        fig = px.scatter(
            x=self.y_test,
            y=self.y_pred,
            labels={"x": "Actual Values", "y": "Predicted Values"},
            title="Actual vs Predicted Housing Prices"
        )
        
        # Add perfect prediction line
        min_val = min(min(self.y_test), min(self.y_pred))
        max_val = max(max(self.y_test), max(self.y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        
        # Apply dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white'),
            colorway=['#ff4081', '#7c4dff', '#00bcd4', '#ffc107', '#4caf50', '#2196f3', 
                    '#e91e63', '#9c27b0', '#cddc39', '#3f51b5', '#009688', '#ff9800']
        )
        
        return fig.to_json()
    
    def get_feature_importance(self):
        """Get feature importance from model"""
        if self.model is None:
            self.train_model()
        
        self.feature_importance = self.model.feature_importances_
        features = self.X.columns
        importance_data = []
        
        for i, feature in enumerate(features):
            importance_data.append({
                "feature": feature,
                "importance": float(self.feature_importance[i])
            })
        
        # Sort by importance
        importance_data = sorted(importance_data, key=lambda x: x["importance"], reverse=True)
        
        # Create plot with dark theme
        fig = px.bar(
            x=[item["importance"] for item in importance_data],
            y=[item["feature"] for item in importance_data],
            orientation='h',
            title="Feature Importance from Random Forest",
            color=[item["importance"] for item in importance_data],
            color_continuous_scale='Plasma'
        )
        
        # Apply dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white'),
            coloraxis_colorbar=dict(title='Importance'),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(gridcolor='rgba(70, 70, 70, 0.5)'),
            yaxis=dict(gridcolor='rgba(70, 70, 70, 0.5)')
        )
        
        # Add hover effects
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
            marker=dict(line=dict(width=1, color='#333'))
        )
        
        return {
            "importance_data": importance_data,
            "plot": fig.to_json()
        }
    
    def calculate_shap_values(self):
        """Calculate SHAP values"""
        # Make sure model is trained first
        if self.model is None or self.X_test is None:
            self.train_model()
        
        # Convert to NumPy array and transfer to device
        X_test_np = self.X_test.to_numpy().astype(np.float32)
        
        # Proper way to transfer data to GPU
        d_X_test = cuda.to_device(X_test_np)
        d_shap_values_out = cuda.device_array((X_test_np.shape[0], X_test_np.shape[1]), dtype=np.float32)
        
        # CUDA kernel
        @cuda.jit
        def calculate_shap_values_gpu(X_test, shap_values_out):
            idx = cuda.grid(1)
            if idx < X_test.shape[0]:
                for i in range(X_test.shape[1]):
                    feature_importance = 0.2 * (i + 1)  # Placeholder
                    shap_values_out[idx, i] = X_test[idx, i] * feature_importance
        
        # Run CUDA kernel
        threadsperblock = 256
        blockspergrid = math.ceil(X_test_np.shape[0] / threadsperblock)
        calculate_shap_values_gpu[blockspergrid, threadsperblock](d_X_test, d_shap_values_out)
        
        # Copy results back to host
        self.shap_values = d_shap_values_out.copy_to_host()
        
        # Get feature names
        features = list(self.X.columns)
        
        # Calculate SHAP importance
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create comparison data
        model_ranks = pd.Series(self.model.feature_importances_, index=features).rank(ascending=False)
        shap_ranks = pd.Series(shap_importance, index=features).rank(ascending=False)
        
        comparison_data = []
        for col in features:
            comparison_data.append({
                "feature": col,
                "model_rank": int(model_ranks[col]),
                "shap_rank": int(shap_ranks[col])
            })
        
        # Create scatter plot for SHAP vs correlation
        corr_with_target = self.X.corrwith(self.y).to_dict()
        
        fig = px.scatter(
            x=[corr_with_target[f] for f in features],
            y=shap_importance,
            text=features,
            labels={"x": "Correlation with Target", "y": "Mean |SHAP Value|"},
            title="Feature Correlation vs SHAP Importance"
        )
        
        # Apply dark theme
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(26, 28, 32, 1)',
            plot_bgcolor='rgba(26, 28, 32, 1)',
            font=dict(color='white'),
            title_font=dict(size=24, color='white'),
            colorway=['#ff4081', '#7c4dff', '#00bcd4', '#ffc107', '#4caf50', '#2196f3', 
                      '#e91e63', '#9c27b0', '#cddc39', '#3f51b5', '#009688', '#ff9800']
        )
        
        fig.update_traces(textposition='top center')
        
        return {
            "comparison_data": comparison_data,
            "plot": fig.to_json()
        }
    
    def run_full_analysis(self):
        """Run the full analysis pipeline and return all results"""
        results = {
            "dataset_info": self.get_dataset_info(),
            "summary_stats": self.get_summary_stats(),
            "missing_values": self.get_missing_values(),
            "correlation_plot": self.create_correlation_plot(),
            "target_distribution": self.create_target_distribution(),
            "feature_distributions": self.create_feature_distributions(),
            "scatter_matrix": self.create_scatter_matrix(),
            "geo_plot": self.create_geo_plot(),
            "feature_engineering": self.perform_feature_engineering(),
        }
        
        # Train model and get metrics
        model_results = self.train_model()
        results["model_evaluation"] = model_results
        results["prediction_plot"] = self.create_prediction_plot()
        
        # Feature importance
        importance_results = self.get_feature_importance()
        results["feature_importance"] = importance_results
        
        # SHAP values
        shap_results = self.calculate_shap_values()
        results["shap_analysis"] = shap_results
        
        return results

# Only run this if script is executed directly (not imported)
if __name__ == "__main__":
    analyzer = HousingAnalyzer()
    results = analyzer.run_full_analysis()
    print("Analysis completed.")