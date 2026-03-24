"""
Advanced Heart Disease Data Preprocessing with SMOTE
Handles imbalanced data, feature engineering, and comprehensive preprocessing
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns

class HeartDiseasePreprocessor:
    """Comprehensive preprocessor for heart disease data"""
    
    def __init__(self, use_smote=True, smote_strategy='auto', scaling_method='standard'):
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy
        self.scaling_method = scaling_method
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.smote = None
        self.feature_names = None
        
    def load_data(self, data_path):
        """Load and validate data"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def exploratory_data_analysis(self, df, save_dir):
        """Perform EDA and save visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Basic statistics
        print("\n=== DATA OVERVIEW ===")
        print(df.info())
        print("\n=== MISSING VALUES ===")
        print(df.isnull().sum())
        print("\n=== TARGET DISTRIBUTION ===")
        print(df['target'].value_counts(normalize=True))
        
        # Save EDA plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        df['target'].value_counts().plot(kind='bar', ax=axes[0,0], title='Target Distribution')
        axes[0,0].set_xlabel('Heart Disease (0=No, 1=Yes)')
        
        # Age distribution by target
        df.boxplot(column='age', by='target', ax=axes[0,1])
        axes[0,1].set_title('Age Distribution by Target')
        
        # Correlation heatmap
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,0], fmt='.2f')
        axes[1,0].set_title('Feature Correlation Matrix')
        
        # Feature importance (preliminary)
        X_temp = df.drop('target', axis=1)
        y_temp = df['target']
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X_temp, y_temp)
        
        feature_importance = pd.DataFrame({
            'feature': X_temp.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.plot(x='feature', y='importance', kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Preliminary Feature Importance')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'eda_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance
    
    def handle_missing_values(self, df):
        """Advanced missing value imputation"""
        print("\n=== HANDLING MISSING VALUES ===")
        
        # Use KNN imputer for better performance
        self.imputer = KNNImputer(n_neighbors=5)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
        
        return X_imputed, y
    
    def feature_engineering(self, X):
        """Create new features from existing ones"""
        print("\n=== FEATURE ENGINEERING ===")
        
        X_engineered = X.copy()
        
        # Age groups
        X_engineered['age_group'] = pd.cut(X['age'], 
                                         bins=[0, 40, 50, 60, 100], 
                                         labels=[0, 1, 2, 3])
        X_engineered['age_group'] = X_engineered['age_group'].astype(int)
        
        # Cholesterol risk levels
        X_engineered['chol_risk'] = pd.cut(X['chol'], 
                                         bins=[0, 200, 240, 1000], 
                                         labels=[0, 1, 2])
        X_engineered['chol_risk'] = X_engineered['chol_risk'].astype(int)
        
        # Blood pressure categories
        X_engineered['bp_category'] = pd.cut(X['trestbps'], 
                                           bins=[0, 120, 140, 1000], 
                                           labels=[0, 1, 2])
        X_engineered['bp_category'] = X_engineered['bp_category'].astype(int)
        
        # Heart rate zones
        X_engineered['hr_zone'] = pd.cut(X['thalach'], 
                                       bins=[0, 100, 150, 1000], 
                                       labels=[0, 1, 2])
        X_engineered['hr_zone'] = X_engineered['hr_zone'].astype(int)
        
        print("Performing advanced feature engineering...")
        
        # 1. Interaction features
        X_engineered['age_chol_interaction'] = X['age'] * X['chol']
        X_engineered['thalach_age_ratio'] = X['thalach'] / (X['age'] + 1)
        X_engineered['oldpeak_slope_interaction'] = X['oldpeak'] * X['slope']
        X_engineered['cp_thalach_interaction'] = X['cp'] * X['thalach']
        X_engineered['restecg_thalach_interaction'] = X['restecg'] * X['thalach']
        
        # 2. Ratio features
        X_engineered['chol_age_ratio'] = X['chol'] / (X['age'] + 1)
        X_engineered['trestbps_age_ratio'] = X['trestbps'] / (X['age'] + 1)
        X_engineered['oldpeak_thalach_ratio'] = X['oldpeak'] / (X['thalach'] + 1)
        
        # 3. Polynomial features for key variables
        poly_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for feature in poly_features:
            if feature in X.columns:
                X_engineered[f'{feature}_squared'] = X[feature] ** 2
                X_engineered[f'{feature}_cubed'] = X[feature] ** 3
        
        # 4. Advanced binned features with domain knowledge
        X_engineered['age_group'] = pd.cut(X['age'], bins=[0, 35, 45, 55, 65, 100], labels=[0, 1, 2, 3, 4])
        X_engineered['chol_level'] = pd.cut(X['chol'], bins=[0, 200, 240, 280, 1000], labels=[0, 1, 2, 3])
        X_engineered['trestbps_level'] = pd.cut(X['trestbps'], bins=[0, 120, 140, 160, 300], labels=[0, 1, 2, 3])
        X_engineered['thalach_level'] = pd.cut(X['thalach'], bins=[0, 120, 150, 180, 250], labels=[0, 1, 2, 3])
        
        # 5. Risk score features (based on medical knowledge)
        # Cardiovascular risk factors
        X_engineered['high_chol_risk'] = (X['chol'] > 240).astype(int)
        X_engineered['high_bp_risk'] = (X['trestbps'] > 140).astype(int)
        X_engineered['low_thalach_risk'] = (X['thalach'] < 120).astype(int)
        X_engineered['diabetes_risk'] = (X['fbs'] == 1).astype(int)
        
        # Composite risk score
        X_engineered['cardiovascular_risk_score'] = (
            X_engineered['high_chol_risk'] + 
            X_engineered['high_bp_risk'] + 
            X_engineered['low_thalach_risk'] + 
            X_engineered['diabetes_risk'] + 
            (X['age'] > 55).astype(int) +
            (X['sex'] == 1).astype(int)  # Male
        )
        
        # 6. Statistical features
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        X_engineered['numeric_mean'] = X[numeric_cols].mean(axis=1)
        X_engineered['numeric_std'] = X[numeric_cols].std(axis=1)
        X_engineered['numeric_range'] = X[numeric_cols].max(axis=1) - X[numeric_cols].min(axis=1)
        
        # 7. Logarithmic transformations for skewed features
        skewed_features = ['chol', 'trestbps', 'oldpeak']
        for feature in skewed_features:
            if feature in X.columns:
                X_engineered[f'{feature}_log'] = np.log1p(X[feature])
        
        # Convert categorical features to numeric
        categorical_cols = X_engineered.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            X_engineered[col] = X_engineered[col].astype(int)
        
        print(f"Features after engineering: {X_engineered.shape[1]}")
        
        return X_engineered
    
    def scale_features(self, X_train, X_test):
        """Scale features using specified method"""
        print(f"\n=== FEATURE SCALING ({self.scaling_method.upper()}) ===")
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Scaling method must be 'standard' or 'robust'")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def apply_advanced_smote(self, X_train, y_train):
        """Apply advanced SMOTE variants for optimal imbalanced data handling"""
        print("\n=== APPLYING ADVANCED SMOTE VARIANTS ===")
        
        print(f"Original distribution: {Counter(y_train)}")
        
        # Test multiple SMOTE variants and select the best one
        smote_variants = {
            'SMOTE': SMOTE(random_state=42, k_neighbors=5),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=5),
            'SVMSMOTE': SVMSMOTE(random_state=42, k_neighbors=5),
            'ADASYN': ADASYN(random_state=42, n_neighbors=5),
            'SMOTETomek': SMOTETomek(random_state=42),
            'SMOTEENN': SMOTEENN(random_state=42)
        }
        
        best_variant = None
        best_score = 0
        best_X, best_y = None, None
        
        # Quick evaluation using cross-validation
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        for variant_name, variant in smote_variants.items():
            try:
                print(f"\nTesting {variant_name}...")
                X_resampled, y_resampled = variant.fit_resample(X_train, y_train)
                
                # Quick evaluation with logistic regression
                lr = LogisticRegression(random_state=42, max_iter=1000)
                scores = cross_val_score(lr, X_resampled, y_resampled, cv=3, scoring='f1')
                avg_score = scores.mean()
                
                print(f"{variant_name} - F1 Score: {avg_score:.4f}, Distribution: {Counter(y_resampled)}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_variant = variant_name
                    best_X, best_y = X_resampled, y_resampled
                    
            except Exception as e:
                print(f"{variant_name} failed: {e}")
                continue
        
        if best_variant:
            print(f"\n🏆 Best SMOTE variant: {best_variant} with F1 Score: {best_score:.4f}")
            print(f"Final distribution: {Counter(best_y)}")
            return best_X, best_y
        else:
            # Fallback to basic SMOTE
            print("\nFalling back to basic SMOTE...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"SMOTE distribution: {Counter(y_resampled)}")
            return X_resampled, y_resampled
    
    def select_features(self, X_train, y_train, X_test, n_features=15):
        """Feature selection using multiple methods"""
        print(f"\n=== FEATURE SELECTION (Top {n_features}) ===")
        
        # Combine SelectKBest and RFE
        selector_kbest = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = selector_kbest.fit_transform(X_train, y_train)
        X_test_selected = selector_kbest.transform(X_test)
        
        # Get selected feature names
        if hasattr(X_train, 'columns'):
            selected_features = X_train.columns[selector_kbest.get_support()]
            print(f"Selected features: {list(selected_features)}")
        
        self.feature_selector = selector_kbest
        
        return X_train_selected, X_test_selected
    
    def save_preprocessor(self, save_dir):
        """Save all preprocessing components"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save all components
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.joblib'))
        if self.imputer:
            joblib.dump(self.imputer, os.path.join(save_dir, 'imputer.joblib'))
        if self.feature_selector:
            joblib.dump(self.feature_selector, os.path.join(save_dir, 'feature_selector.joblib'))
        if self.smote:
            joblib.dump(self.smote, os.path.join(save_dir, 'smote.joblib'))
        
        print(f"Preprocessor components saved to {save_dir}")

def preprocess_data(client_id, use_smote=True, generate_data=False):
    """
    Main preprocessing function with advanced techniques
    """
    print(f"Starting advanced preprocessing for {client_id}...")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'clients', client_id, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Generate data if needed
    if generate_data:
        print("Generating heart disease dataset...")
        from generate_heart_disease_data import generate_heart_disease_dataset, save_dataset
        df = generate_heart_disease_dataset(n_samples=1000)
        save_dataset(df, base_dir)
    
    # Load data - first check raw directory, then data directory
    data_path = os.path.join(data_dir, 'raw', 'heart_disease.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(data_dir, 'heart_disease_data.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find data file. Checked: {data_path} and {os.path.join(data_dir, 'heart_disease_data.csv')}")
    
    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor(
        use_smote=use_smote,
        smote_strategy='auto',
        scaling_method='standard'
    )
    
    # Load and analyze data
    df = preprocessor.load_data(data_path)
    feature_importance = preprocessor.exploratory_data_analysis(df, processed_dir)
    
    # Handle missing values
    X, y = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    X_engineered = preprocessor.feature_engineering(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Apply advanced SMOTE if requested
    if use_smote:
        X_train_balanced, y_train_balanced = preprocessor.apply_advanced_smote(X_train_scaled, y_train)

    # Feature selection
    X_train_final, X_test_final = preprocessor.select_features(
        X_train_balanced, y_train_balanced, X_test_scaled, n_features=15
    )

    # Save processed data
    np.savez(os.path.join(processed_dir, 'train.npz'), 
             X=X_train_final, y=y_train)
    np.savez(os.path.join(processed_dir, 'test.npz'), 
             X=X_test_final, y=y_test)
    
    # Save preprocessor
    preprocessor.save_preprocessor(processed_dir)
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(processed_dir, 'feature_importance.csv'), index=False)
    
    print(f"\n=== PREPROCESSING COMPLETE ===")
    print(f"Final training shape: {X_train_final.shape}")
    print(f"Final test shape: {X_test_final.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return X_train_final, X_test_final, y_train, y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced preprocessing for heart disease data.')
    parser.add_argument('--client_id', type=str, required=True, 
                       help='Client ID (e.g., client_1)')
    parser.add_argument('--use_smote', action='store_true', default=True,
                       help='Use SMOTE for handling imbalanced data')
    parser.add_argument('--generate_data', action='store_true', default=False,
                       help='Generate synthetic data first')
    
    args = parser.parse_args()
    preprocess_data(args.client_id, args.use_smote, args.generate_data)
