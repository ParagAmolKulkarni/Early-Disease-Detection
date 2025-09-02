# Early Disease Prediction - Machine Learning Pipeline
# Part C: Early Disease Detection

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Setting style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("HEART DISEASE PREDICTION - MACHINE LEARNING PIPELINE")
print("=" * 80)
print("\n🔬 Part C: Early Disease Detection Analysis")
print("📊 Comprehensive ML solution for cardiovascular disease prediction")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
def quick_test_mode(df, sample_size=10000):
    """Use smaller sample for faster testing"""
    if len(df) > sample_size:
        print(f"⚡ Quick mode: Using {sample_size} samples for faster execution")
        return df.sample(n=sample_size, random_state=42)
    return df

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50),  
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

def load_and_explore_data(file_path):
    """Load data and perform initial exploration"""
    print("\n" + "="*50)
    print("1. DATA LOADING AND INITIAL EXPLORATION")
    print("="*50)
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Display basic information
        print(f"\n📋 Dataset Overview:")
        print(f"   • Number of samples: {df.shape[0]:,}")
        print(f"   • Number of features: {df.shape[1]:,}")
        print(f"   • Target variable: 'disease'")
        
        # Display first few rows
        print(f"\n🔍 First 5 rows of the dataset:")
        print(df.head())
        
        # Data types and info
        print(f"\n📊 Data Types and Non-null Counts:")
        print(df.info())
        
        # Statistical summary
        print(f"\n📈 Statistical Summary:")
        print(df.describe())
        
        return df
    
    except FileNotFoundError:
        print("❌ Error: Data file not found. Please ensure 'datafile.csv' is in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    print("\n" + "="*50)
    print("2. EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    # Check for missing values
    print("\n🔍 Missing Values Analysis:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing_df['Missing Count'].sum() == 0:
        print("✅ No missing values found in the dataset!")
    
    # Target variable distribution
    print("\n🎯 Target Variable Distribution:")
    disease_counts = df['disease'].value_counts()
    disease_percent = df['disease'].value_counts(normalize=True) * 100
    
    print(f"   • No Disease (0): {disease_counts[0]:,} ({disease_percent[0]:.1f}%)")
    print(f"   • Disease (1): {disease_counts[1]:,} ({disease_percent[1]:.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Heart Disease Dataset - Initial Analysis', fontsize=16, fontweight='bold')
    
    # Target distribution
    axes[0,0].pie(disease_counts.values, labels=['No Disease', 'Disease'], 
                  autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
    axes[0,0].set_title('Disease Distribution')
    
    # Age distribution
    axes[0,1].hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].set_title('Age Distribution (in days)')
    axes[0,1].set_xlabel('Age (days)')
    axes[0,1].set_ylabel('Frequency')
    
    # Blood pressure analysis
    axes[1,0].scatter(df['ap_lo'], df['ap_hi'], alpha=0.6, c=df['disease'], 
                      cmap='RdYlBu', s=20)
    axes[1,0].set_title('Blood Pressure Distribution')
    axes[1,0].set_xlabel('Diastolic BP (ap_lo)')
    axes[1,0].set_ylabel('Systolic BP (ap_hi)')
    
    # BMI calculation and distribution
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    axes[1,1].boxplot([df[df['disease']==0]['bmi'], df[df['disease']==1]['bmi']], 
                      labels=['No Disease', 'Disease'])
    axes[1,1].set_title('BMI Distribution by Disease Status')
    axes[1,1].set_ylabel('BMI')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis for numerical features
    print("\n🔗 Correlation Analysis:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()
    
    return df

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("\n" + "="*50)
    print("3. DATA PREPROCESSING")
    print("="*50)
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Convert age from days to years for better interpretability
    df_processed['age_years'] = df_processed['age'] / 365.25
    print(f"✅ Converted age from days to years")
    
    # Convert height from cm to meters (assuming height is in cm, not days as per data dict)
    if df_processed['height'].mean() > 50:  # Likely in cm if mean > 50
        df_processed['height_m'] = df_processed['height'] / 100
        print(f"✅ Converted height from cm to meters")
    
    # Calculate BMI
    df_processed['bmi'] = df_processed['weight'] / (df_processed['height_m'] ** 2)
    print(f"✅ Calculated BMI feature")
    
    # Handle outliers in blood pressure
    print(f"\n🔍 Handling outliers:")
    
    # Remove extreme blood pressure values (likely data entry errors)
    bp_outliers_before = len(df_processed)
    df_processed = df_processed[
        (df_processed['ap_hi'] >= 70) & (df_processed['ap_hi'] <= 250) &
        (df_processed['ap_lo'] >= 40) & (df_processed['ap_lo'] <= 150)
    ]
    bp_outliers_after = len(df_processed)
    print(f"   • Removed {bp_outliers_before - bp_outliers_after} extreme BP outliers")
    
    # Handle BMI outliers
    bmi_q1 = df_processed['bmi'].quantile(0.01)
    bmi_q99 = df_processed['bmi'].quantile(0.99)
    bmi_outliers_before = len(df_processed)
    df_processed = df_processed[(df_processed['bmi'] >= bmi_q1) & (df_processed['bmi'] <= bmi_q99)]
    bmi_outliers_after = len(df_processed)
    print(f"   • Removed {bmi_outliers_before - bmi_outliers_after} BMI outliers (1st-99th percentile)")
    
    # Encode categorical variables
    print(f"\n🔤 Encoding categorical variables:")
    le_gender = LabelEncoder()
    df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])
    print(f"   • Gender encoded: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
    
    le_country = LabelEncoder()
    df_processed['country_encoded'] = le_country.fit_transform(df_processed['country'])
    print(f"   • Country encoded (first 5): {dict(list(zip(le_country.classes_[:5], le_country.transform(le_country.classes_[:5]))))}...")
    
    le_occupation = LabelEncoder()
    df_processed['occupation_encoded'] = le_occupation.fit_transform(df_processed['occupation'])
    print(f"   • Occupation encoded (first 5): {dict(list(zip(le_occupation.classes_[:5], le_occupation.transform(le_occupation.classes_[:5]))))}...")
    
    # Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age_years'], 
                                     bins=[0, 30, 40, 50, 60, 100], 
                                     labels=['<30', '30-40', '40-50', '50-60', '60+'])
    le_age_group = LabelEncoder()
    df_processed['age_group_encoded'] = le_age_group.fit_transform(df_processed['age_group'].astype(str))
    
    print(f"✅ Final processed dataset shape: {df_processed.shape}")
    
    return df_processed

# ============================================================================
# 4. FEATURE SELECTION AND PREPARATION
# ============================================================================

def prepare_features(df_processed):
    """Select and prepare features for modeling"""
    print("\n" + "="*50)
    print("4. FEATURE SELECTION AND PREPARATION")
    print("="*50)
    
    # Select features for modeling
    feature_columns = [
        'age_years', 'gender_encoded', 'height_m', 'weight', 'bmi',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 
        'smoke', 'alco', 'active', 'country_encoded', 'occupation_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['disease']
    
    print(f"📊 Selected features ({len(feature_columns)}):")
    for i, feature in enumerate(feature_columns, 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n🎯 Target variable: disease")
    print(f"   • Shape of X: {X.shape}")
    print(f"   • Shape of y: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✂️ Data Split:")
    print(f"   • Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   • Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"⚖️ Features standardized using StandardScaler")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns

# ============================================================================
# 5. MODEL DEVELOPMENT AND TRAINING
# ============================================================================

def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train multiple classification models"""
    print("\n" + "="*50)
    print("5. MODEL DEVELOPMENT AND TRAINING")
    print("="*50)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    # Store results
    results = {}
    
    print(f"🤖 Training {len(models)} different models...")
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 65)
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        print(f"{name:<25} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    return results

# ============================================================================
# 6. MODEL EVALUATION AND COMPARISON
# ============================================================================

def evaluate_models(results, y_test):
    """Evaluate and compare model performance"""
    print("\n" + "="*50)
    print("6. MODEL EVALUATION AND COMPARISON")
    print("="*50)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1-Score': [results[model]['f1_score'] for model in results.keys()],
        'AUC': [results[model]['auc'] if results[model]['auc'] is not None else 0 for model in results.keys()]
    })
    
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    print("\n📊 Model Performance Comparison (sorted by F1-Score):")
    print(comparison_df.round(4))
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_results = results[best_model_name]
    
    print(f"\n🏆 Best Performing Model: {best_model_name}")
    print(f"   • Accuracy: {best_model_results['accuracy']:.4f}")
    print(f"   • Precision: {best_model_results['precision']:.4f}")
    print(f"   • Recall: {best_model_results['recall']:.4f}")
    print(f"   • F1-Score: {best_model_results['f1_score']:.4f}")
    if best_model_results['auc'] is not None:
        print(f"   • AUC-ROC: {best_model_results['auc']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Performance metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[i//2, i%2].bar(x, comparison_df[metric], width=0.6, alpha=0.8)
        axes[i//2, i%2].set_title(f'{metric} Comparison')
        axes[i//2, i%2].set_xlabel('Models')
        axes[i//2, i%2].set_ylabel(metric)
        axes[i//2, i%2].set_xticks(x)
        axes[i//2, i%2].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[i//2, i%2].grid(True, alpha=0.3)
        
        # Highlight best performer
        best_idx = comparison_df[metric].idxmax()
        axes[i//2, i%2].bar(best_idx, comparison_df.loc[best_idx, metric], 
                           width=0.6, alpha=0.8, color='red', label='Best')
    
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix for best model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_model_results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve for models with probability predictions
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        if result['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_model_name, best_model_results, comparison_df

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(best_model_results, best_model_name, feature_columns):
    """Analyze feature importance"""
    print("\n" + "="*50)
    print("7. FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    model = best_model_results['model']
    
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
        importance_type = "Coefficient Magnitude"
    else:
        print(f"❌ Feature importance not available for {best_model_name}")
        return
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 {importance_type} Analysis ({best_model_name}):")
    print(feature_importance_df.round(4))
    
    # Visualization
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
    plt.xlabel(importance_type)
    plt.title(f'Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    
    # Color bars by importance level
    colors = plt.cm.viridis(feature_importance_df['Importance'] / feature_importance_df['Importance'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df

# ============================================================================
# 8. INSIGHTS AND RECOMMENDATIONS
# ============================================================================

def generate_insights(df_processed, feature_importance_df, comparison_df, best_model_name):
    """Generate insights and recommendations"""
    print("\n" + "="*80)
    print("8. INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n🔍 KEY FINDINGS:")
    print(f"{'='*50}")
    
    # Dataset insights
    total_samples = len(df_processed)
    disease_rate = df_processed['disease'].mean() * 100
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total samples analyzed: {total_samples:,}")
    print(f"   • Heart disease prevalence: {disease_rate:.1f}%")
    
    # Model performance insights
    print(f"\n🤖 Model Performance:")
    print(f"   • Best performing model: {best_model_name}")
    print(f"   • Achieved F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    print(f"   • Model accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
    
    # Feature importance insights
    print(f"\n🎯 Most Important Risk Factors:")
    top_features = feature_importance_df.head(5)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"   {i}. {row['Feature']}: {row['Importance']:.4f}")
    
    # Statistical insights
    print(f"\n📈 Risk Factor Analysis:")
    
    # Age analysis
    avg_age_disease = df_processed[df_processed['disease']==1]['age_years'].mean()
    avg_age_no_disease = df_processed[df_processed['disease']==0]['age_years'].mean()
    print(f"   • Average age with disease: {avg_age_disease:.1f} years")
    print(f"   • Average age without disease: {avg_age_no_disease:.1f} years")
    
    # BMI analysis
    avg_bmi_disease = df_processed[df_processed['disease']==1]['bmi'].mean()
    avg_bmi_no_disease = df_processed[df_processed['disease']==0]['bmi'].mean()
    print(f"   • Average BMI with disease: {avg_bmi_disease:.1f}")
    print(f"   • Average BMI without disease: {avg_bmi_no_disease:.1f}")
    
    # Blood pressure analysis
    avg_systolic_disease = df_processed[df_processed['disease']==1]['ap_hi'].mean()
    avg_systolic_no_disease = df_processed[df_processed['disease']==0]['ap_hi'].mean()
    print(f"   • Average systolic BP with disease: {avg_systolic_disease:.1f} mmHg")
    print(f"   • Average systolic BP without disease: {avg_systolic_no_disease:.1f} mmHg")
    
    # Lifestyle factor analysis
    smoking_disease_rate = df_processed[df_processed['smoke']==1]['disease'].mean() * 100
    drinking_disease_rate = df_processed[df_processed['alco']==1]['disease'].mean() * 100
    active_disease_rate = df_processed[df_processed['active']==1]['disease'].mean() * 100
    
    print(f"   • Disease rate among smokers: {smoking_disease_rate:.1f}%")
    print(f"   • Disease rate among drinkers: {drinking_disease_rate:.1f}%")
    print(f"   • Disease rate among physically active: {active_disease_rate:.1f}%")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"{'='*50}")
    print(f"   1. 🏥 Clinical Implementation:")
    print(f"      • Deploy {best_model_name} for preliminary screening")
    print(f"      • Focus on top risk factors: {', '.join(top_features['Feature'].head(3).tolist())}")
    print(f"      • Achieve {comparison_df.iloc[0]['Accuracy']*100:.1f}% accuracy in risk assessment")
    
    print(f"\n   2. 🎯 Prevention Strategies:")
    if 'ap_hi' in top_features['Feature'].head(3).tolist():
        print(f"      • Prioritize blood pressure monitoring and management")
    if 'bmi' in top_features['Feature'].head(3).tolist():
        print(f"      • Implement weight management programs")
    if 'age_years' in top_features['Feature'].head(3).tolist():
        print(f"      • Increase screening frequency for older adults")
    
    print(f"\n   3. 🔬 Model Improvement:")
    print(f"      • Collect more data on underrepresented groups")
    print(f"      • Include additional biomarkers (e.g., lipid profiles)")
    print(f"      • Implement ensemble methods for better performance")
    print(f"      • Regular model retraining with new data")
    
    print(f"\n   4. 📊 Monitoring and Validation:")
    print(f"      • Establish continuous monitoring system")
    print(f"      • Validate model performance across different populations")
    print(f"      • Set up alerts for high-risk individuals")
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE - Heart Disease Prediction Model Ready for Deployment")
    print(f"{'='*80}")

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main function to execute the complete pipeline"""
    
    # Step 1: Load and explore data
    df = load_and_explore_data('datafile.csv')
    if df is None:
        return
    
    # Step 2: Perform EDA
    df = perform_eda(df)
    
    # Step 3: Preprocess data
    df_processed = preprocess_data(df)
    
    # Step 4: Prepare features
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns = prepare_features(df_processed)
    
    # Step 5: Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 6: Evaluate models
    best_model_name, best_model_results, comparison_df = evaluate_models(results, y_test)
    
    # Step 7: Analyze feature importance
    feature_importance_df = analyze_feature_importance(best_model_results, best_model_name, feature_columns)
    
    # Step 8: Generate insights
    generate_insights(df_processed, feature_importance_df, comparison_df, best_model_name)
    
    return {
        'processed_data': df_processed,
        'best_model': best_model_results['model'],
        'scaler': scaler,
        'feature_columns': feature_columns,
        'results': results,
        'feature_importance': feature_importance_df
    }

# ============================================================================
# RUN THE COMPLETE PIPELINE
# ============================================================================

if __name__ == "__main__":
    # Execute the complete machine learning pipeline
    pipeline_results = main()
    
    print(f"\n🎉 Pipeline execution completed successfully!")
    print(f"📝 All components are ready for deployment and further analysis.")
