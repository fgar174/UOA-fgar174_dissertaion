# Project Name

## Overview

This project provides a framework for training, tuning, and evaluating machine learning models on a specified dataset.
The main script, `main.py`, handles all key tasks, including running base models, tuning specific models, and generating
comprehensive reports for model performance comparison and analysis.

## Requirements

The project requires Python 3.10.13. Install dependencies with:

```bash
pipenv install
```

Dependencies

	•	matplotlib 3.7.5
	•	pandas 2.0.3
	•	seaborn 0.13.2
	•	scikit-learn 1.3.2
	•	imbalanced-learn 0.12.3
	•	joblib 1.4.2
	•	openpyxl 3.1.5

### Usage

The main entry point for this project is main.py, which orchestrates model training, tuning, and report generation. Run
this file to initiate the full pipeline.

Key Functions in main.py

#### Running Base Models

```bash
run_all_combinations(dataset_name, train_df, final_test_df)
```

Trains a set of base models on the provided dataset, using train_df for training and final_test_df for testing, to
establish baseline performance.

#### Hyperparameter Tuning

```bash
run_all_month_weeks_tuned(train_df, dataset_name, ModelType.GRADIENT_BOOSTING, final_test_df, ScoringMetrics.F1_WEIGHTED)
```

#### Generating Reports

```bash
report_generator = ReportGenerator(output_dir="model_metrics", result_dir="model_results", tuned=True)
report_generator.generate_model_comparison()
report_generator.generate_classification_reports_comparison()
report_generator.generate_month_week_classification_reports()
report_generator.generate_week_month_testing_metrics()
```

Generates and saves a variety of reports, including:
	•	Model Comparison: Overview of all model performances.
	•	Classification Reports Comparison: Detailed classification metrics for each model.
	•	Month-Week Classification Reports: Performance based on monthly or weekly splits.
	•	Testing Metrics: Detailed evaluation on testing data across time splits.

## Running the Project

### Execute the Main Script

```bash
python main.py
```

Running main.py will perform model training, tuning, and report generation.

## Project Structure

	•	main.py: Main script for running all processes.
	•	model_metrics/: Directory for storing model performance metrics.
	•	model_results/: Directory for storing classification reports and results.
