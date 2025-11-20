from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from io import StringIO
import json

app = Flask(__name__)
CORS(app)

# Global data store
data_store = {}

@app.route('/api/analyze-dataset', methods=['POST'])
def analyze_dataset():
    """
    Comprehensive dataset analysis endpoint
    Handles CSV/Excel uploads and performs EDA
    """
    try:
        file = request.files.get('file')
        project_type = request.form.get('project_type', 'general')
        
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Read file based on extension
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV or Excel'}), 400
        
        # Store dataframe
        data_store['df'] = df
        data_store['project_type'] = project_type
        
        # Perform comprehensive analysis
        analysis = perform_eda(df)
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def perform_eda(df):
    """
    Perform Exploratory Data Analysis
    """
    analysis = {}
    
    # Basic shape information
    analysis['shape'] = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1])
    }
    
    # Column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    analysis['numeric_cols'] = len(numeric_cols)
    analysis['categorical_cols'] = len(categorical_cols)
    analysis['column_names'] = df.columns.tolist()
    
    # Statistical summary for numeric columns
    if numeric_cols:
        summary_stats = df[numeric_cols].describe()
        summary_list = []
        for col in numeric_cols[:10]:  # Limit to first 10 columns
            if col in summary_stats.columns:
                summary_list.append({
                    'Metric': col,
                    'Count': int(summary_stats.loc['count', col]),
                    'Mean': float(summary_stats.loc['mean', col]),
                    'Std': float(summary_stats.loc['std', col]),
                    'Min': float(summary_stats.loc['min', col]),
                    '25%': float(summary_stats.loc['25%', col]),
                    '50%': float(summary_stats.loc['50%', col]),
                    '75%': float(summary_stats.loc['75%', col]),
                    'Max': float(summary_stats.loc['max', col])
                })
        analysis['summary'] = summary_list
    
    # Missing data analysis
    missing = df.isnull().sum()
    missing_data = []
    for col in missing[missing > 0].index[:15]:  # Top 15 columns with missing data
        missing_data.append({
            'column': col,
            'missing_count': int(missing[col]),
            'missing_percent': float((missing[col] / len(df)) * 100)
        })
    analysis['missing_data'] = missing_data
    
    # Feature distributions (mean and std for numeric columns)
    if numeric_cols:
        distributions = []
        for col in numeric_cols[:10]:  # First 10 numeric columns
            distributions.append({
                'feature': col,
                'mean': float(df[col].mean()) if pd.notna(df[col].mean()) else 0,
                'std': float(df[col].std()) if pd.notna(df[col].std()) else 0,
                'min': float(df[col].min()) if pd.notna(df[col].min()) else 0,
                'max': float(df[col].max()) if pd.notna(df[col].max()) else 0
            })
        analysis['distributions'] = distributions
    
    # Correlation analysis (top correlations)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        correlations = []
        
        # Get upper triangle of correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if pd.notna(corr_value) and abs(corr_value) > 0.3:  # Only significant correlations
                    correlations.append({
                        'pair': f'{col1[:15]} - {col2[:15]}',
                        'correlation': float(corr_value)
                    })
        
        # Sort by absolute correlation and take top 15
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        analysis['correlations'] = correlations[:15]
    
    # Categorical value counts (for top categorical columns)
    if categorical_cols:
        category_analysis = []
        for col in categorical_cols[:5]:  # First 5 categorical columns
            value_counts = df[col].value_counts().head(10)
            category_analysis.append({
                'column': col,
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.to_dict()
            })
        analysis['categorical_analysis'] = category_analysis
    
    # Data quality metrics
    analysis['data_quality'] = {
        'total_missing': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'completeness': float((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
    }
    
    return analysis

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """
    Train ML model on uploaded dataset
    """
    try:
        if 'df' not in data_store:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        df = data_store['df']
        params = request.json
        
        target_column = params.get('target_column')
        model_type = params.get('model_type', 'random_forest')
        task_type = params.get('task_type', 'classification')
        
        if not target_column or target_column not in df.columns:
            return jsonify({'error': 'Invalid target column'}), 400
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if task_type == 'classification':
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            result = {
                'task': 'classification',
                'model': model_type,
                'accuracy': float(accuracy),
                'test_samples': int(len(y_test))
            }
        else:
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            result = {
                'task': 'regression',
                'model': model_type,
                'r2_score': float(r2),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'test_samples': int(len(y_test))
            }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = [
                {'feature': col, 'importance': float(imp)}
                for col, imp in zip(X.columns, importances)
            ]
            feature_imp.sort(key=lambda x: x['importance'], reverse=True)
            result['feature_importance'] = feature_imp[:15]
        
        # Store model
        data_store['model'] = model
        data_store['X_test'] = X_test
        data_store['y_test'] = y_test
        data_store['y_pred'] = y_pred
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clustering', methods=['POST'])
def perform_clustering():
    """
    Perform K-Means clustering
    """
    try:
        if 'df' not in data_store:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        df = data_store['df']
        params = request.json
        n_clusters = params.get('n_clusters', 3)
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_df.shape[1] < 2:
            return jsonify({'error': 'Need at least 2 numeric columns for clustering'}), 400
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Prepare response
        result = {
            'data': [
                {'x': float(pca_data[i, 0]), 'y': float(pca_data[i, 1]), 'cluster': int(clusters[i])}
                for i in range(min(1000, len(pca_data)))  # Limit to 1000 points
            ],
            'explained_variance': [float(v) for v in pca.explained_variance_ratio_],
            'n_clusters': n_clusters
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pca-analysis', methods=['POST'])
def pca_analysis():
    """
    Perform PCA analysis
    """
    try:
        if 'df' not in data_store:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        df = data_store['df']
        params = request.json
        n_components = params.get('n_components', 2)
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_df.shape[1] < n_components:
            return jsonify({'error': f'Need at least {n_components} numeric columns'}), 400
        
        # Standardize and perform PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(scaled_data)
        
        result = {
            'pca_data': pca_data[:1000].tolist(),  # Limit to 1000 points
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-columns', methods=['GET'])
def get_columns():
    """
    Get column names from loaded dataset
    """
    if 'df' not in data_store:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    df = data_store['df']
    
    return jsonify({
        'columns': df.columns.tolist(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Flask ML API is running',
        'datasets_loaded': 'df' in data_store
    })

if __name__ == '__main__':
    print("Starting Flask ML Data Analysis API...")
    print("Server running on http://localhost:5000")
    print("Available endpoints:")
    print("  POST /api/analyze-dataset - Upload and analyze dataset")
    print("  POST /api/train-model - Train ML model")
    print("  POST /api/clustering - Perform clustering")
    print("  POST /api/pca-analysis - PCA analysis")
    print("  GET  /api/get-columns - Get dataset columns")
    print("  GET  /api/health - Health check")
    app.run(debug=True, port=5000, host='0.0.0.0')