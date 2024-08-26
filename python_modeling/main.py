import time

import os
import joblib
import json

import numpy as np
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    log_loss
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

from enum import Enum


class ModelType(Enum):
    RANDOM_FOREST = 'RandomForest'
    LOGISTIC_REGRESSION = 'LogisticRegression'
    GRADIENT_BOOSTING = 'GradientBoosting'
    SVM = 'SVM'
    NAIVE_BAYES = 'NaiveBayes'
    KNN = 'KNN'


feature_columns = [
    # 'arrive_pos_loctype',
    'requires_power',
    'nominal_length',
    'freight_kind',
    'category',
    'time_in_month',
    'time_in_hour',
    'time_in_weekday',
    'time_in_business_day',
    'time_in_week_of_year',
    'time_in_week_of_month',
]


class DataPreprocessor:
    def __init__(self, dataset, bins):
        self.dataset = dataset
        self.bins = bins
        self.right = False

    def create_labels(self):
        if self.right:
            labels = [f"{self.bins[i]:02} - {self.bins[i + 1]:02}" for i in range(len(self.bins) - 1)]
        else:
            labels = [f"{self.bins[i]:02} - {self.bins[i + 1] - 1:02}" for i in range(len(self.bins) - 1)]
        return labels

    def preprocess(self):
        labels = self.create_labels()
        self.dataset.loc[:, 'DAYS_IN_PORT_CATEGORY'] = pd.cut(
            self.dataset['DAYS_IN_PORT'],
            bins=self.bins,
            labels=labels,
            right=self.right
        )
        data = self.dataset.dropna(subset=['DAYS_IN_PORT_CATEGORY'])

        X = data[feature_columns]
        y = data['DAYS_IN_PORT_CATEGORY']
        return X, y


class ModelTrainer:
    def __init__(
            self,
            model_type: ModelType,
            dataset_name,
            split_test_size,
            save,
            bins,
            scaled_data=False,
            param_grid=None,
            stratified_split_testing=False,
    ):
        self.model_type = model_type
        self.param_grid = param_grid
        self.dataset_name = dataset_name
        self.scaled_data = scaled_data
        self.best_model = None
        self.split_test_size = split_test_size
        self.model, self.default_param_grid = self.get_model_and_params()
        self.preprocessor = None
        self.bins = bins
        self.save = save
        self.stratified_split_testing = stratified_split_testing
        self.categorical_cols = ['nominal_length', 'freight_kind', 'category']
        self.numerical_cols = [
            'time_in_month',
            'time_in_hour',
            'time_in_weekday',
            'requires_power',
            'time_in_business_day',
            'time_in_week_of_year',
            'time_in_week_of_month',
        ]


    def get_model_and_params(self):
        model_classes = {
            ModelType.RANDOM_FOREST: (
                RandomForestClassifier(random_state=42, n_jobs=-1, verbose=3),
                {
                    'model__n_estimators': [100, 200, 500],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['auto', 'sqrt', 'log2'],
                    'model__bootstrap': [True, False]
                }
            ),
            ModelType.LOGISTIC_REGRESSION: (
                LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1, verbose=3),
                {
                    'model__C': [0.01]
                }
            ),
            ModelType.GRADIENT_BOOSTING: (
                GradientBoostingClassifier(random_state=42, verbose=3),
                {
                    'model__n_estimators': [100, 200, 500],
                    'model__learning_rate': [0.01, 0.1, 0.05],
                    'model__max_depth': [3, 5, 10],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__subsample': [0.8, 1.0],
                    'model__max_features': ['auto', 'sqrt', 'log2']
                }
            ),
            ModelType.SVM: (
                SVC(random_state=42, verbose=3),
                {
                    'model__C': [0.1, 1, 10, 100],
                    'model__kernel': ['linear', 'rbf', 'poly'],
                    'model__gamma': ['scale', 'auto'],
                    'model__degree': [3, 4, 5],
                    'model__class_weight': [None, 'balanced']
                }
            ),
            ModelType.NAIVE_BAYES: (
                GaussianNB(),
                {
                    'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            ),
            ModelType.KNN: (
                KNeighborsClassifier(),
                {
                    'model__n_neighbors': [3, 5, 7, 10],
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan', 'minkowski']
                }
            )
        }
        model, default_param_grid = model_classes[self.model_type]
        return model, default_param_grid

    def train_model(self, train_df):
        param_grid = self.param_grid if self.param_grid is not None else self.default_param_grid
        data_preprocessor = DataPreprocessor(dataset=train_df, bins=self.bins)
        X, y = data_preprocessor.preprocess()

        self.preprocessor = self.preprocess_features()

        if self.stratified_split_testing:
            X_train, X_test, y_train, y_test = self.split_data_stratified(X, y)
        else:
            X_train, X_test, y_train, y_test = self.split_data(X, y)

        self.plot_classes(pd.DataFrame({'DAYS_IN_PORT_CATEGORY': y_train}))

        pipeline = Pipeline(steps=[('model', self.model)])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        elapsed_time_seconds = time.time() - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60

        best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        cv_scores = cross_val_score(self.best_model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test) if hasattr(self.best_model.named_steps['model'],
                                                              'predict_proba') else None
        test_accuracy = self.best_model.score(X_test, y_test)

        # Metrics and Results
        metrics = self.evaluate_model(y_test, y_pred, y_proba, elapsed_time_minutes)
        metrics['cv_scores'] = cv_scores.tolist()
        metrics['test_accuracy'] = test_accuracy

        if self.save:
            models_dir = "trained_models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            self.save_results(best_params, metrics)

            filename = f"{self.get_config_filename()}.joblib"
            model_filepath = os.path.join(models_dir, filename)
            joblib.dump(self.best_model, model_filepath)


        return self.best_model, metrics

    def get_config_filename(self):
        not_tuned = self.param_grid == {}
        filename_conf = f'b_{"_".join(map(str, self.bins))}_sst_{self.stratified_split_testing}_scaled_{self.scaled_data}'
        return f"{''.join(self.model_type.value.split())}_{filename_conf}_{'not_tuned' if not_tuned else 'tuned'}"

    def save_results(self, best_params, metrics, output_dir="model_results"):
        """Save the best parameters and metrics to a JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{self.get_config_filename()}.json"
        filepath = os.path.join(output_dir, filename)

        not_tuned = self.param_grid == {}

        results = {
            "model_type": self.model_type.value,
            "bins": self.bins,
            "scaled": self.scaled_data,
            "tuned": not not_tuned,
            "stratified_split_testing": self.stratified_split_testing,
            "best_params": best_params,
            "metrics": metrics,
            "dataset_name": self.dataset_name,
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)

    def preprocess_features(self):
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])

        return preprocessor

    def split_data_stratified(self, X, y):
        X['strat_group'] = X['time_in_month'].astype(str) + '_' + X['time_in_week_of_month'].astype(str)
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        X_train = pd.DataFrame()
        y_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_test = pd.DataFrame()

        # Perform the split using StratifiedShuffleSplit
        for train_idx, test_idx in strat_split.split(X, X['strat_group']):
            X_train = X.iloc[train_idx].drop(columns=['strat_group'])
            X_test = X.iloc[test_idx].drop(columns=['strat_group'])
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

        # print("DATA SPLIT FOR TRAINING :::::::::::::::::::::::::::")
        # # Check the distribution of months and weeks in the train and test sets
        # print("Train set month and week distribution:")
        # print(X_train[['time_in_month', 'time_in_week_of_month']].value_counts(normalize=True))
        #
        # print("\nTest set month and week distribution:")
        # print(X_test[['time_in_month', 'time_in_week_of_month']].value_counts(normalize=True))
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.fit_transform(X_test)
        if self.scaled_data:
            X_train, y_train = self.scale_data(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def split_data(self, X, y):
        X_processed = self.preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=self.split_test_size, random_state=42)
        if self.scaled_data:
            X_train, y_train = self.scale_data(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled

    def plot_classes(self, dataset, output_dir="plots"):
        """Plot the distribution of classes and save the plot to disk."""
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.countplot(
            data=dataset,
            x='DAYS_IN_PORT_CATEGORY',
            hue='DAYS_IN_PORT_CATEGORY',
            palette='light:#5A9',
            legend=False
        )
        plt.title(f'Distribution of Classes for {"_".join(map(str, self.bins))}{self.scaled_data and " Scaled" or ""}{self.stratified_split_testing and " Stratified" or ""} Data')
        plt.xlabel('DAYS_IN_PORT_CATEGORY')
        plt.ylabel('Frequency')

        # Save the plot to disk
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = f'{self.get_config_filename()}_class_distribution.png'
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path, bbox_inches='tight')

    def evaluate_model(self, y_test, y_pred, y_proba, elapsed_time_minutes):
        feature_importances = None
        if hasattr(self.best_model.named_steps['model'], 'coef_'):
            feature_importances = self.best_model.named_steps['model'].coef_[0]
        elif hasattr(self.best_model.named_steps['model'], 'feature_importances_'):
            feature_importances = self.best_model.named_steps['model'].feature_importances_

        feature_importances_dict = None
        if feature_importances is not None:
            categorical_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
                self.categorical_cols)
            print(self.numerical_cols, "HERE numerical_cols ::::::::::::.")
            print(categorical_feature_names, "HERE categorical_feature_names ::::::::::::.")
            feature_names = np.concatenate([self.numerical_cols, categorical_feature_names])
            feature_importances_dict = {feature: importance for feature, importance in
                                        zip(feature_names, feature_importances)}

            feature_importances_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importances
            })
            feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
            print("Feature Importances:")
            print(feature_importances_df)

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None
        log_loss_value = log_loss(y_test, y_proba) if y_proba is not None else None
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "matthews_corrcoef": mcc,
            "cohen_kappa_score": kappa,
            "roc_auc_score": auc_score,
            "log_loss": log_loss_value,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "feature_importances": feature_importances_dict,
            "elapsed_time_minutes": elapsed_time_minutes,
        }

        # Print Evaluation Metrics
        print("Evaluation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        if auc_score is not None:
            print(f"ROC AUC Score: {auc_score:.4f}")
        if log_loss_value is not None:
            print(f"Log Loss: {log_loss_value:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        return metrics

    def predict_and_evaluate(self, X_new, y_true):
        """Predict on new data using the best model and evaluate the predictions."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Please train the model first.")

        # Use the preprocessor fitted during training
        X_processed = self.preprocessor.transform(X_new)  # Use transform instead of fit_transform

        # Make predictions
        y_pred = self.best_model.predict(X_processed)
        y_proba = self.best_model.predict_proba(X_processed) if hasattr(self.best_model.named_steps['model'],
                                                                        'predict_proba') else None

        # Evaluate metrics
        return self.evaluate_model(
            y_true,
            y_pred,
            y_proba,
            elapsed_time_minutes=None
        )

    def save_week_test(self, month, week, metrics, output_dir="model_week_test_results"):
        """Save the best parameters and metrics to a JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename_wm = f'w_{week}_m_{month}'
        filename = f"{self.get_config_filename()}_{filename_wm}.json"
        filepath = os.path.join(output_dir, filename)

        tuned = self.param_grid != {}
        results = {
            "model_type": self.model_type.value,
            "bins": self.bins,
            "scaled": self.scaled_data,
            "tuned": tuned,
            "stratified_split_testing": self.stratified_split_testing,
            "week": week,
            "month": month,
            "metrics": metrics,
            "dataset_name": self.dataset_name,
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results week of month saved to {filepath}")

    def evaluate_testing_data(self, testing_data, month, week):
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("Evaluating Testing Data :::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print(f"Month: {month}, Week: {week}")
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        testing_data['time_in'] = pd.to_datetime(testing_data['time_in'], errors='coerce')
        filter_month_week = testing_data[
            (testing_data['time_in'].dt.month == month) & (testing_data['time_in_week_of_month'] == week)]
        test_data_preprocessor = DataPreprocessor(dataset=filter_month_week, bins=self.bins)
        X_final_test, y_final_test = test_data_preprocessor.preprocess()
        new_metrics = self.predict_and_evaluate(X_final_test, y_final_test)
        self.save_week_test(month, week, new_metrics)


class ReportGenerator:
    def __init__(self, output_dir="model_metrics", result_dir="model_results"):
        self.output_dir = output_dir
        self.result_dir = result_dir
        self.result_dir_month_week = 'model_week_test_results'

    def generate_model_comparison(self, output_file="model_comparison.csv"):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        files = [f for f in os.listdir(self.result_dir) if f.endswith('.json')]
        if not files:
            print(f"No results files found in '{self.result_dir}'")
            return

        comparison_data = []

        for file in files:
            filepath = os.path.join(self.result_dir, file)
            with open(filepath, 'r') as f:
                results = json.load(f)
                model_type = results['model_type']
                bins = results['bins']
                metrics = results['metrics']
                file_name = file
                scaled = results['scaled']
                tuned = results['tuned']
                stratified_split_testing = results['stratified_split_testing']

                # Extract relevant metrics
                cv_scores = metrics.get("cv_scores", None)
                test_accuracy = metrics.get("test_accuracy", None)
                accuracy = metrics.get("accuracy", None)
                precision = metrics.get("precision", None)
                recall = metrics.get("recall", None)
                matthews_corrcoef = metrics.get("matthews_corrcoef", None)
                cohen_kappa_score = metrics.get("cohen_kappa_score", None)
                roc_auc_score = metrics.get("roc_auc_score", None)
                log_loss = metrics.get("log_loss", None)
                elapsed_time_minutes = metrics.get("elapsed_time_minutes", None)
                f1_score = metrics.get("classification_report", "").split()[
                    -2] if "classification_report" in metrics else None

                comparison_data.append({
                    "file_name": file_name,
                    "model_type": model_type,
                    "bins": str(bins),
                    "scaled": scaled,
                    "tuned": tuned,
                    "stratified_split_testing": stratified_split_testing,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "matthews_corrcoef": matthews_corrcoef,
                    "cohen_kappa_score": cohen_kappa_score,
                    "roc_auc_score": roc_auc_score,
                    "log_loss": log_loss,
                    "cv_scores": cv_scores,
                    "test_accuracy": test_accuracy,
                    "elapsed_time_minutes": elapsed_time_minutes,
                })

        output_path = os.path.join(self.output_dir, output_file)
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_excel(output_path.replace('.csv', '.xlsx'), index=False)
        print(df_comparison)

    def extract_classification_report(self, report_str, model_type, bins, scaled, tuned, stratified_split_testing):
        """Extract the classification report into a structured DataFrame."""
        report_lines = report_str.strip().split("\n")
        classes = []
        data = []

        for line in report_lines[2:-3]:  # Skip the headers and footer
            parts = line.split()
            if parts:
                class_name = f"From {parts[0]} to {parts[2]} days"
                precision = float(parts[3])
                recall = float(parts[4])
                f1_score = float(parts[5])
                support = int(parts[6])
                classes.append(class_name)
                data.append([
                    model_type,
                    str(bins),
                    scaled,
                    tuned,
                    stratified_split_testing,
                    class_name,
                    precision,
                    recall,
                    f1_score,
                    support
                ])
        df_report = pd.DataFrame(
            data,
            columns=[
                "model_type",
                "bins",
                "scaled",
                "tuned",
                "stratified_split_testing",
                "class",
                "precision",
                "recall",
                "f1_score",
                "support"
            ],
            index=classes
        )
        return df_report.reset_index(drop=True)

    def generate_classification_reports_comparison(self, output_file="classification_report_comparison.xlsx"):
        if not os.path.exists(self.output_dir):
            print(f"No results directory found at '{self.output_dir}'")
            return

        files = [f for f in os.listdir(self.result_dir) if f.endswith('.json')]
        if not files:
            print(f"No results files found in '{self.result_dir}'")
            return

        all_reports = []

        for file in files:
            filepath = os.path.join(self.result_dir, file)
            with open(filepath, 'r') as f:
                results = json.load(f)
                model_type = results['model_type']
                bins = results['bins']
                scaled = results['scaled']
                tuned = results['tuned']
                stratified_split_testing = results['stratified_split_testing']
                classification_report_str = results['metrics'].get('classification_report', '')

                if classification_report_str:
                    df_report = self.extract_classification_report(
                        classification_report_str,
                        model_type,
                        bins,
                        scaled,
                        tuned,
                        stratified_split_testing
                    )
                    all_reports.append(df_report)  # Append each report DataFrame to the list

        if all_reports:
            # Combine all report DataFrames into a single DataFrame
            combined_reports_df = pd.concat(all_reports, ignore_index=True)
            combined_reports_df.to_excel(os.path.join(self.output_dir, output_file), index=False)
            print(combined_reports_df)
        else:
            print("No classification reports found to compare.")

    def generate_month_week_classification_reports(self, output_file="month_week_classification_reports.xlsx"):
        if not os.path.exists(self.output_dir):
            print(f"No results directory found at '{self.output_dir}'")
            return

        files = [f for f in os.listdir(self.result_dir_month_week) if f.endswith('.json')]
        if not files:
            print(f"No results files found in '{self.result_dir_month_week}'")
            return

        all_reports = []

        for file in files:
            filepath = os.path.join(self.result_dir_month_week, file)
            with open(filepath, 'r') as f:
                results = json.load(f)
                model_type = results['model_type']
                bins = results['bins']
                scaled = results['scaled']
                tuned = results['tuned']
                stratified_split_testing = results['stratified_split_testing']
                week = results['week']
                month = results['month']
                classification_report_str = results['metrics'].get('classification_report', '')

                if classification_report_str:
                    df_report = self.extract_classification_report(
                        classification_report_str,
                        model_type,
                        bins,
                        scaled,
                        tuned,
                        stratified_split_testing
                    )
                    df_report['week'] = week
                    df_report['month'] = month
                    all_reports.append(df_report)

        if all_reports:
            combined_reports_df = pd.concat(all_reports, ignore_index=True)
            combined_reports_df.to_excel(os.path.join(self.output_dir, output_file), index=False)
            print(combined_reports_df)
        else:
            print("No classification reports found to compare.")

    def generate_week_month_testing_metrics(self, output_file="month_week_testing_metrics.csv"):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        files = [f for f in os.listdir(self.result_dir_month_week) if f.endswith('.json')]
        if not files:
            print(f"No results files found in '{self.result_dir_month_week}'")
            return

        comparison_data = []

        for file in files:
            filepath = os.path.join(self.result_dir_month_week, file)
            with open(filepath, 'r') as f:
                results = json.load(f)
                model_type = results['model_type']
                bins = results['bins']
                week = results['week']
                tuned = results['tuned']
                month = results['month']
                metrics = results['metrics']
                scaled = results['scaled']
                stratified_split_testing = results['stratified_split_testing']
                file_name = file

                # Extract relevant metrics
                accuracy = metrics.get("accuracy", None)
                precision = metrics.get("precision", None)
                recall = metrics.get("recall", None)
                matthews_corrcoef = metrics.get("matthews_corrcoef", None)
                cohen_kappa_score = metrics.get("cohen_kappa_score", None)
                roc_auc_score = metrics.get("roc_auc_score", None)
                log_loss = metrics.get("log_loss", None)
                f1_score = metrics.get("classification_report", "").split()[
                    -2] if "classification_report" in metrics else None

                comparison_data.append({
                    "file_name": file_name,
                    "model_type": model_type,
                    "tuned": tuned,
                    "scaled": scaled,
                    "stratified_split_testing": stratified_split_testing,
                    "bins": str(bins),
                    "week": week,
                    "month": month,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "matthews_corrcoef": matthews_corrcoef,
                    "cohen_kappa_score": cohen_kappa_score,
                    "roc_auc_score": roc_auc_score,
                    "log_loss": log_loss
                })

        output_path = os.path.join(self.output_dir, output_file)
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_excel(output_path.replace('.csv', '.xlsx'), index=False)


def run_all_combinations(dataset_name, train_df, final_test_df):
    model_types = [
        # ModelType.RANDOM_FOREST,
        ModelType.LOGISTIC_REGRESSION,
        # ModelType.GRADIENT_BOOSTING
    ]

    bins_list = [
        [0, 3, 12, 21],
        [0, 4, 12, 21],
        [0, 5, 12, 21],
        [0, 6, 12, 21],
        [0, 3, 11, 21],
        [0, 4, 11, 21],
        [0, 5, 11, 21],
        [0, 6, 11, 21],
        [0, 3, 10, 21],
        [0, 4, 10, 21],
        [0, 5, 10, 21],
        [0, 6, 10, 21],
        [0, 3, 9, 21],
        [0, 4, 9, 21],
        [0, 5, 9, 21],
        [0, 6, 9, 21],
    ]

    stratified_options = [False, True]
    scaled_options = [False, False]

    for model_type in model_types:
        for bins in bins_list:
            for stratified in stratified_options:
                for scaled in scaled_options:
                    print(f"Running Model: {model_type.value}, Bins: {bins}, "
                          f"Stratified: {stratified}, Scaled: {scaled}")

                    # Initialize trainer
                    trainer = ModelTrainer(
                        split_test_size=0.2,
                        model_type=model_type,
                        dataset_name=dataset_name,
                        scaled_data=scaled,
                        save=True,
                        bins=bins,
                        param_grid={},
                        stratified_split_testing=stratified
                    )

                    # Train the model
                    best_model, metrics = trainer.train_model(train_df)

                    # Evaluate on different combinations of week and month
                    for month in range(1, 4):  # Adjust the range as per your dataset
                        weeks = range(1, 5) if month < 3 else range(1, 3)
                        for week in weeks:
                            trainer.evaluate_testing_data(final_test_df, month, week)


if __name__ == '__main__':
    dataset_name = 'subset_base.csv'
    file_path = f"dataset/{dataset_name}"
    data = pd.read_csv(file_path)

    train_df = data[data['FOR_TEST'] == False].copy()
    final_test_df = data[data['FOR_TEST'] == True].copy()
    run_all_combinations(dataset_name, train_df, final_test_df)

    report_generator = ReportGenerator(output_dir="model_metrics", result_dir="model_results")
    report_generator.generate_model_comparison()
    report_generator.generate_classification_reports_comparison()
    report_generator.generate_month_week_classification_reports()
    report_generator.generate_week_month_testing_metrics()