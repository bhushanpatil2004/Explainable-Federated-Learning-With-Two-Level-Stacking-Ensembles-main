import os
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    StackingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    StratifiedKFold, cross_validate, train_test_split
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PolynomialFeatures,
    PowerTransformer, QuantileTransformer
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import shap
from tqdm import tqdm
import json

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class TwoLevelStackingEnsemble:
    """Two-level stacking ensemble that combines multiple base models with a meta-model"""
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def predict_proba(self, X):
        X_l2 = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                if pred.shape[1] == 2:  # Binary classification
                    pred = pred[:, 1:]
            else:
                pred = model.predict(X).reshape(-1, 1)
            X_l2.append(pred)
        X_l2 = np.hstack(X_l2)
        return self.meta_model.predict_proba(X_l2)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:  # Binary classification
            return (proba[:, 1] > 0.5).astype(int)
        return np.argmax(proba, axis=1)

class BlendingEnsemble:
    """Blending ensemble that averages predictions from multiple models"""
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self
        
    def predict_proba(self, X):
        predictions = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(predictions, axis=0)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

class HeartDiseaseModelTrainer:
    """Advanced model trainer with hyperparameter optimization"""
    
    def __init__(self, optimization_method='optuna', cv_folds=5):
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.best_models = {}
        self.best_params = {}
        self.model_scores = {}
        
    def select_features_with_shap(self, X, y, model, n_features=20):
        """Select top features using SHAP values"""
        print("\n" + "="*70)
        print("SELECTING TOP FEATURES USING SHAP VALUES")
        print("="*70)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification, get SHAP values for positive class
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[1])  # Take absolute SHAP values
        else:
            shap_values = np.abs(shap_values)
            
        # Get mean absolute SHAP values for each feature
        shap_means = np.mean(shap_values, axis=0)
        
        # Get indices of top n features
        top_feature_indices = np.argsort(shap_means)[-n_features:]
        
        print(f"Selected top {n_features} features based on SHAP importance")
        return top_feature_indices
    
    def load_data(self, processed_dir, use_shap_selection=True, n_features=20):
        """Load and enhance preprocessed data with advanced feature engineering"""
        train_data = np.load(os.path.join(processed_dir, 'train.npz'))
        test_data = np.load(os.path.join(processed_dir, 'test.npz'))
        
        X_train, y_train = train_data['X'], train_data['y']
        X_test, y_test = test_data['X'], test_data['y']
        
        # Store original shapes
        original_train_shape = X_train.shape
        original_test_shape = X_test.shape
        
        # Convert to DataFrame for easier manipulation
        feature_columns = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_columns)
        X_test = pd.DataFrame(X_test, columns=feature_columns)
        
        # Feature Engineering
        def add_interaction_terms(X):
            # Add polynomial features
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(X)
            poly_columns = [f"interaction_{i}" for i in range(poly_features.shape[1])]
            return pd.concat([X, pd.DataFrame(poly_features, columns=poly_columns)], axis=1)
        
        # Apply feature engineering
        X_train = add_interaction_terms(X_train)
        X_test = add_interaction_terms(X_test)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y_train)[0]-1))
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        print(f"\n{'='*70}")
        print("DATA PREPROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Original training data shape: {original_train_shape}")
        print(f"Training data shape after feature engineering: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(y_train)}")
        
        # Apply SHAP-based feature selection if enabled
        if use_shap_selection and X_train.shape[1] > n_features:
            print("\nTraining initial model for SHAP-based feature selection...")
            
            # Train a quick model for SHAP
            model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            model.fit(X_train, y_train)
            
            # Select top features
            top_feature_indices = self.select_features_with_shap(
                X_train.values, y_train, model, n_features
            )
            
            # Apply feature selection
            X_train = X_train.iloc[:, top_feature_indices]
            X_test = X_test.iloc[:, top_feature_indices]
            
            print(f"\nSelected {X_train.shape[1]} most important features using SHAP")
            
        print(f"\nFinal training data shape: {X_train.shape}")
        print(f"Final test data shape: {X_test.shape}")
        print(f"{'='*70}\n")
        
        return X_train.values, X_test.values, y_train, y_test
    
    def optimize_lightgbm(self, X_train, y_train):
        """Optimize LightGBM using Optuna with advanced parameters"""
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 2000, step=20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'verbosity': -1,
                'random_state': 42,
                'n_jobs': -1
            }
            
            if params['boosting_type'] == 'goss':
                params['feature_fraction'] = 1.0
                params['bagging_fraction'] = 1.0
            
            model = lgb.LGBMClassifier(**params)
            
            # Use stratified k-fold cross-validation with multiple metrics
            scoring = {
                'accuracy': 'accuracy',
                'f1': 'f1',
                'roc_auc': 'roc_auc',
                'precision': 'precision',
                'recall': 'recall'
            }
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_validate(
                model, X_train, y_train, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )
            
            # Return the mean test accuracy (primary metric)
            return np.mean(scores['test_accuracy'])
        
        # Optimize with early stopping
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour timeout
        
        # Get best parameters and create final model
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1
        })
        
        return lgb.LGBMClassifier(**best_params), best_params
    
    def optimize_xgboost(self, X_train, y_train):
        """Optimize XGBoost using Optuna with advanced parameters"""
        def objective(trial):
            params = {
                'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0, step=0.05),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0, step=0.05),
                'gamma': trial.suggest_float('gamma', 0, 1, step=0.05),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, step=0.1),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0, step=0.1),
                'random_state': 42,
                'n_jobs': -1,
                'use_label_encoder': False,
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
            }
            
            if params['booster'] == 'dart':
                params.update({
                    'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
                    'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
                    'rate_drop': trial.suggest_float('rate_drop', 0.0, 1.0, step=0.1),
                    'skip_drop': trial.suggest_float('skip_drop', 0.0, 1.0, step=0.1)
                })
            
            model = xgb.XGBClassifier(**params)
            
            # Use stratified k-fold cross-validation with multiple metrics
            scoring = {
                'accuracy': 'accuracy',
                'f1': 'f1',
                'roc_auc': 'roc_auc',
                'precision': 'precision',
                'recall': 'recall',
                'average_precision': 'average_precision'
            }
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_validate(
                model, X_train, y_train, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )
            
            # Return the mean test accuracy (primary metric)
            return np.mean(scores['test_accuracy'])
        
        # Optimize with early stopping
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour timeout
        
        # Get best parameters and create final model
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'aucpr',
            'tree_method': 'hist'
        })
        
        return xgb.XGBClassifier(**best_params), best_params
    
    def optimize_catboost(self, X_train, y_train):
        """Optimize CatBoost using Optuna with simplified parameters"""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-5, 10, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255, step=32),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
                'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', 
                                                                  ['Newton', 'Gradient']),
                'grow_policy': trial.suggest_categorical('grow_policy', 
                                                       ['SymmetricTree', 'Depthwise']),
                'random_state': 42,
                'verbose': False,
                'thread_count': -1,
                'task_type': 'CPU',
                'boosting_type': 'Plain',  # Simplified to avoid complexity
                'bootstrap_type': 'Bernoulli',  # Use Bernoulli as default
                'subsample': trial.suggest_float('subsample', 0.1, 1.0)
            }
            
            # Only include max_leaves for Lossguide grow policy (removed for simplicity)
            if params['grow_policy'] == 'Lossguide':
                params['max_leaves'] = trial.suggest_int('max_leaves', 16, 64, step=4)
            
            model = cb.CatBoostClassifier(**params)
            
            # Use stratified k-fold cross-validation with multiple metrics
            scoring = {
                'accuracy': 'accuracy',
                'f1': 'f1',
                'roc_auc': 'roc_auc',
                'precision': 'precision',
                'recall': 'recall',
                'average_precision': 'average_precision'
            }
            
            cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y_train))), 
                               shuffle=True, 
                               random_state=42)
            
            try:
                scores = cross_validate(
                    model, X_train, y_train, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs=1,  # CatBoost doesn't support n_jobs=-1 with cross_validate
                    return_train_score=False
                )
                
                # Return the mean test accuracy (primary metric)
                return np.mean(scores['test_accuracy'])
                
            except Exception as e:
                # Return a very low score if an error occurs during training
                print(f"Error during CatBoost training: {str(e)}")
                return 0.0
        
        # Optimize with early stopping
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour timeout
        
        # Get best parameters and create final model
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'verbose': False,
            'thread_count': -1,
            'task_type': 'CPU'
        })
        
        return cb.CatBoostClassifier(**best_params), best_params
    
    def optimize_random_forest(self, X_train, y_train):
        """Optimize Random Forest using GridSearch with enhanced error handling"""
        try:
            print("\nStarting Random Forest optimization...")
            print(f"Input data shape: {X_train.shape}, Target shape: {y_train.shape}")
            print("Sample data:", X_train[:2])
            
            # Simplified parameter grid for initial testing
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True]
            }
            
            print("\nFitting Random Forest model...")
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, 
                param_grid, 
                cv=min(3, self.cv_folds),  # Use smaller CV for faster testing
                scoring='accuracy', 
                n_jobs=1,  # Reduce parallel jobs to avoid memory issues
                verbose=3
            )
            
            grid_search.fit(X_train, y_train)
            print("\nRandom Forest optimization completed successfully!")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, grid_search.best_params_
            
        except Exception as e:
            print(f"\nError in Random Forest optimization: {str(e)}")
            print("Using default Random Forest parameters...")
            # Return a simple model with default parameters
            rf = RandomForestClassifier(random_state=42, n_estimators=100)
            rf.fit(X_train, y_train)
            return rf, {'n_estimators': 100, 'max_depth': None}
    
    def optimize_neural_network(self, X_train, y_train):
        """Optimize Neural Network using Optuna"""
        def objective(trial):
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                    [(100,), (200,), (100, 50), (200, 100), (300, 150, 75)]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'max_iter': 1000,
                'random_state': 42
            }
            
            model = MLPClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=30)
        
        best_params = study.best_params
        best_params.update({'max_iter': 1000, 'random_state': 42})
        
        return MLPClassifier(**best_params), best_params
    
    def optimize_svm(self, X_train, y_train):
        """Optimize SVM using GridSearch"""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1, 10],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'class_weight': ['balanced', None]
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=self.cv_folds, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def optimize_gradient_boosting(self, X_train, y_train):
        """Optimize Gradient Boosting using Optuna"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            
            model = GradientBoostingClassifier(**params)
            
            # Use cross-validation for more robust evaluation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, n_jobs=-1)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_model = GradientBoostingClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        return best_model, best_params
    
    def train_all_models(self, X_train, y_train):
        """Train and optimize all models including advanced ensembles"""
        print("="*70)
        print("TRAINING BASE MODELS")
        print("="*70)
        
        # Train each model type
        print("\nOptimizing Random Forest...")
        rf_model, rf_params = self.optimize_random_forest(X_train, y_train)
        self.best_models['RandomForest'] = rf_model
        self.best_params['RandomForest'] = rf_params
        
        print("\nOptimizing XGBoost...")
        xgb_model, xgb_params = self.optimize_xgboost(X_train, y_train)
        self.best_models['XGBoost'] = xgb_model
        self.best_params['XGBoost'] = xgb_params
        
        print("\nOptimizing LightGBM...")
        lgb_model, lgb_params = self.optimize_lightgbm(X_train, y_train)
        self.best_models['LightGBM'] = lgb_model
        self.best_params['LightGBM'] = lgb_params
        
        print("\nOptimizing CatBoost...")
        cb_model, cb_params = self.optimize_catboost(X_train, y_train)
        self.best_models['CatBoost'] = cb_model
        self.best_params['CatBoost'] = cb_params
        
        print("\nOptimizing Gradient Boosting...")
        gb_model, gb_params = self.optimize_gradient_boosting(X_train, y_train)
        self.best_models['GradientBoosting'] = gb_model
        self.best_params['GradientBoosting'] = gb_params
        
        # Train all models
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        for name, model in self.best_models.items():
            if hasattr(model, 'fit'):
                print(f"\nTraining {name}...")
                model.fit(X_train, y_train)
                
                # Cross-validation score
                skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
                self.model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Get base models for ensemble
        base_models = {
            'RandomForest': self.best_models.get('RandomForest'),
            'XGBoost': self.best_models.get('XGBoost'),
            'LightGBM': self.best_models.get('LightGBM'),
            'CatBoost': self.best_models.get('CatBoost'),
            'GradientBoosting': self.best_models.get('GradientBoosting')
        }
        
        # Remove any None values in case some models weren't trained
        base_models = {k: v for k, v in base_models.items() if v is not None}
        
        if not base_models:
            raise ValueError("No base models available for ensemble")
            
        print(f"\nUsing {len(base_models)} base models for ensemble: {', '.join(base_models.keys())}")
        
        return base_models
    
    def create_advanced_ensemble(self, X_train, y_train):
        """Create an advanced two-level stacking ensemble from the trained base models"""
        print("\n" + "="*70)
        print("CREATING ADVANCED TWO-LEVEL STACKING ENSEMBLE")
        print("="*70)
        
        # Get base models
        base_models = {
            'RandomForest': self.best_models.get('RandomForest'),
            'XGBoost': self.best_models.get('XGBoost'),
            'LightGBM': self.best_models.get('LightGBM'),
            'CatBoost': self.best_models.get('CatBoost'),
            'GradientBoosting': self.best_models.get('GradientBoosting')
        }
        
        # Remove any None values in case some models weren't trained
        base_models = {k: v for k, v in base_models.items() if v is not None}
        
        if not base_models:
            raise ValueError("No base models available for ensemble")
        
        # Step 1: Generate first-level predictions
        def get_first_level_preds(X):
            """Get predictions from all first-level models"""
            preds = []
            for name, model in base_models.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    if pred.shape[1] == 2:  # Binary classification
                        pred = pred[:, 1:]
                else:
                    pred = model.predict(X).reshape(-1, 1)
                preds.append(pred)
            return np.hstack(preds)
        
        # Step 2: Create second-level features
        print("\nGenerating second-level features...")
        X_train_l2 = get_first_level_preds(X_train)
        
        # Step 3: Train second-level meta-model
        print("\nTraining second-level meta-model...")
        meta_model = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            n_jobs=-1, 
            class_weight='balanced'
        )
        meta_model.fit(X_train_l2, y_train)
        
        # Create and store the final ensemble
        final_ensemble = TwoLevelStackingEnsemble(base_models, meta_model)
        self.best_models['TwoLevelStacking'] = final_ensemble
        
        # Evaluate the ensemble
        print("\nEvaluating two-level stacking ensemble...")
        y_pred = final_ensemble.predict(X_train)
        y_proba = final_ensemble.predict_proba(X_train)[:, 1] if hasattr(final_ensemble, 'predict_proba') else None
        
        acc = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_train, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_train, y_proba) if y_proba is not None else 0
        
        print(f"\nTwo-Level Stacking Ensemble Performance:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if y_proba is not None:
            print(f"  AUC:       {auc:.4f}")
            
        # Store scores
        self.model_scores['TwoLevelStacking'] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return final_ensemble
        print(f"Blending CV Accuracy: {blending_score:.4f} (+/- {np.std(blending_scores) * 2:.4f})")
        
        self.best_models['blending'] = blending
        self.model_scores['blending'] = {
            'cv_mean': blending_score,
            'cv_std': np.std(blending_scores)
        }
        
        # 5. Hyperparameter optimization for best ensemble
        print("\n5. Optimizing Best Ensemble Hyperparameters...")
        
        # Get preliminary best ensemble
        preliminary_scores = {
            'soft_voting': self.model_scores['soft_voting']['cv_mean'],
            'weighted_voting': self.model_scores['weighted_voting']['cv_mean'],
            'blending': self.model_scores['blending']['cv_mean']
        }
        
        if 'stacking' in self.model_scores:
            preliminary_scores['stacking'] = self.model_scores['stacking']['cv_mean']
        
        preliminary_best = max(preliminary_scores, key=preliminary_scores.get)
        
        # Optimize ensemble hyperparameters
        if preliminary_best == 'soft_voting':
            optimized_ensemble = self.optimize_voting_ensemble(X_train, y_train, voting_estimators, 'soft')
        elif preliminary_best == 'weighted_voting':
            optimized_ensemble = self.optimize_voting_ensemble(X_train, y_train, voting_estimators, 'soft', use_weights=True)
        elif preliminary_best == 'stacking' and best_stacking:
            optimized_ensemble = self.optimize_stacking_ensemble(X_train, y_train, base_estimators)
        else:
            optimized_ensemble = self.best_models[preliminary_best]
        
        # Final ensemble scores
        ensemble_scores = {
            'soft_voting': self.model_scores['soft_voting']['cv_mean'],
            'weighted_voting': self.model_scores['weighted_voting']['cv_mean'],
            'blending': self.model_scores['blending']['cv_mean']
        }
        
        if 'stacking' in self.model_scores:
            ensemble_scores['stacking'] = self.model_scores['stacking']['cv_mean']
        
        if 'optimized_ensemble' in self.model_scores:
            ensemble_scores['optimized_ensemble'] = self.model_scores['optimized_ensemble']['cv_mean']
        
        best_ensemble_name = max(ensemble_scores, key=ensemble_scores.get)
        best_ensemble = self.best_models.get(best_ensemble_name, optimized_ensemble)
        
        print(f"\n🏆 Best Ensemble: {best_ensemble_name.upper()} with CV Accuracy: {ensemble_scores[best_ensemble_name]:.4f}")
        
        return best_ensemble
    
    def optimize_voting_ensemble(self, X_train, y_train, estimators, voting_type='soft', use_weights=False):
        """Optimize voting ensemble hyperparameters"""
        print(f"\nOptimizing {voting_type} voting ensemble...")
        
        if use_weights:
            # Optimize weights using Optuna
            def objective(trial):
                weights = [trial.suggest_float(f'weight_{i}', 0.1, 2.0) for i in range(len(estimators))]
                ensemble = VotingClassifier(estimators=estimators, voting=voting_type, weights=weights)
                
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(ensemble, X_train, y_train, cv=skf, scoring='accuracy')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            
            best_weights = [study.best_params[f'weight_{i}'] for i in range(len(estimators))]
            optimized_ensemble = VotingClassifier(estimators=estimators, voting=voting_type, weights=best_weights)
            
            print(f"Optimized weights: {best_weights}")
        else:
            optimized_ensemble = VotingClassifier(estimators=estimators, voting=voting_type)
        
        optimized_ensemble.fit(X_train, y_train)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(optimized_ensemble, X_train, y_train, cv=skf, scoring='accuracy')
        
        print(f"Optimized ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.best_models['optimized_ensemble'] = optimized_ensemble
        self.model_scores['optimized_ensemble'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return optimized_ensemble
    
    def optimize_stacking_ensemble(self, X_train, y_train, base_estimators):
        """Optimize stacking ensemble hyperparameters"""
        print("\nOptimizing stacking ensemble...")
        
        def objective(trial):
            # Optimize meta-learner
            meta_type = trial.suggest_categorical('meta_type', ['lr', 'rf', 'xgb'])
            cv_folds = trial.suggest_int('cv_folds', 3, 5)
            
            if meta_type == 'lr':
                C = trial.suggest_float('C', 0.01, 10.0, log=True)
                meta_learner = LogisticRegression(C=C, random_state=42, max_iter=1000)
            elif meta_type == 'rf':
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                meta_learner = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            else:  # xgb
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 3, 8)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                meta_learner = xgb.XGBClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    learning_rate=learning_rate,
                    random_state=42, 
                    eval_metric='logloss'
                )
            
            stacking = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=cv_folds,
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(stacking, X_train, y_train, cv=skf, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        # Build optimized stacking ensemble
        best_params = study.best_params
        meta_type = best_params['meta_type']
        cv_folds = best_params['cv_folds']
        
        if meta_type == 'lr':
            meta_learner = LogisticRegression(C=best_params['C'], random_state=42, max_iter=1000)
        elif meta_type == 'rf':
            meta_learner = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                random_state=42
            )
        else:  # xgb
            meta_learner = xgb.XGBClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                random_state=42,
                eval_metric='logloss'
            )
        
        optimized_stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=cv_folds,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        optimized_stacking.fit(X_train, y_train)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(optimized_stacking, X_train, y_train, cv=skf, scoring='accuracy')
        
        print(f"Optimized stacking CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Best meta-learner: {meta_type}, CV folds: {cv_folds}")
        
        self.best_models['optimized_stacking'] = optimized_stacking
        self.model_scores['optimized_stacking'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return optimized_stacking
    
    def evaluate_models(self, X_test, y_test, save_dir):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        results = {}
        
        for name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            if auc:
                print(f"  AUC:       {auc:.4f}")
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'))
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
        print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        return results, best_model_name
    
    def save_models(self, save_dir, best_model_name):
        """Save all models and best model separately"""
        models_dir = os.path.join(save_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save all models
        for name, model in self.best_models.items():
            joblib.dump(model, os.path.join(models_dir, f'{name}_model.joblib'))
        
        # Save best model as main model
        best_model = self.best_models[best_model_name]
        joblib.dump(best_model, os.path.join(save_dir, 'best_model.joblib'))
        
        # Save parameters
        with open(os.path.join(save_dir, 'best_params.txt'), 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Parameters: {self.best_params.get(best_model_name, 'N/A')}\n")
        
        print(f"Models saved to {models_dir}")
        print(f"Best model ({best_model_name}) saved as best_model.joblib")

def train_model(client_id, target_accuracy=0.95):
    """
    Main training function targeting 95% accuracy
    """
    print(f"Starting advanced model training for {client_id}...")
    print(f"Target accuracy: {target_accuracy*100}%")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    processed_dir = os.path.join(base_dir, 'clients', client_id, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'clients', client_id, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if processed data exists
    if not os.path.exists(os.path.join(processed_dir, 'train.npz')):
        print("Processed data not found. Running preprocessing...")
        from advanced_preprocess import preprocess_data
        preprocess_data(client_id, use_smote=True, generate_data=True)
    
    # Initialize trainer
    trainer = HeartDiseaseModelTrainer(optimization_method='optuna', cv_folds=5)
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data(processed_dir)
    
    # Train all models
    trainer.train_all_models(X_train, y_train)
    
    # Create advanced ensemble
    ensemble = trainer.create_advanced_ensemble(X_train, y_train)
    
    # Evaluate models
    results, best_model_name = trainer.evaluate_models(X_test, y_test, model_dir)
    
    # Save models
    trainer.save_models(model_dir, best_model_name)
    
    # Check if target accuracy achieved
    best_accuracy = results[best_model_name]['accuracy']
    
    # Display comprehensive accuracy summary
    print(f"\n{'='*80}")
    print(f"🎯 FINAL MODEL ACCURACY SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Dataset Size: {len(y_train) + len(y_test)} total samples")
    print(f"   - Training: {len(y_train)} samples")
    print(f"   - Testing: {len(y_test)} samples")
    print(f"📈 Target Accuracy: {target_accuracy*100:.1f}%")
    print(f"🏆 Best Model: {best_model_name.upper()}")
    print(f"✅ Achieved Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    if best_accuracy >= target_accuracy:
        print(f"🎉 SUCCESS: TARGET ACCURACY ACHIEVED!")
        print(f"🔥 Model exceeds {target_accuracy*100}% accuracy requirement")
    else:
        print(f"⚠️  PARTIAL SUCCESS: Target not fully reached")
        print(f"📊 Achieved {best_accuracy*100:.2f}% vs target {target_accuracy*100}%")
    
    # Show all model performances
    print(f"\n📋 ALL MODEL PERFORMANCES:")
    print("-" * 50)
    for model_name, metrics in results.items():
        status = "✅" if metrics['accuracy'] >= target_accuracy else "❌"
        print(f"{status} {model_name.upper():<15}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print(f"\n💾 Model saved to: {model_dir}")
    print(f"📄 Results saved to: {model_dir}/model_comparison.csv")
    print(f"{'='*80}")
    
    return trainer, results, best_model_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train advanced heart disease prediction model.')
    parser.add_argument('--client_id', type=str, required=True, 
                       help='Client ID (e.g., client_1)')
    parser.add_argument('--target_accuracy', type=float, default=0.95,
                       help='Target accuracy (default: 0.95)')
    
    args = parser.parse_args()
    train_model(args.client_id, args.target_accuracy)
