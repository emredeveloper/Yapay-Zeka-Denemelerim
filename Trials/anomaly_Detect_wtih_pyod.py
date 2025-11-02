import pandas as pd
from sklearn.model_selection import train_test_split

# Import ECOD instead of Sampling
from pyod.models.ecod import ECOD
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

if __name__ == "__main__":
    # NOTE: The 'contamination' parameter is required by many PyOD models, including Sampling.
    # It represents the *assumed* proportion of outliers in the dataset and is used
    # to set the threshold for predictions. It's often treated as a hyperparameter
    # if the true value is unknown. Common starting points are 0.1 or 0.05.
    # ECOD uses it for predict() and determining labels_ after fit().
    contamination = 0.1  # Assumed percentage of outliers
    test_size = 0.3  # Proportion of data to use for testing

    # --- Load Your Data ---
    # Replace 'your_dataset.csv' with the path to your data file
    # Adjust parameters like 'sep', 'header' etc. based on your file format
    data_path = 'wildlife_data.csv' # <<<--- MAKE SURE TO REPLACE THIS
    df = pd.read_csv(data_path)

    # --- Prepare Features (X) and Labels (y) ---
    # Use actual numerical column names from wildlife_data.csv as features
    feature_columns = [
        'Population', 'Size (min cm)', 'Size (max cm)',
        'Weight (min kg)', 'Weight (max kg)',
        'Lifespan (min years)', 'Lifespan (max years)'
    ]
    # Set label_column to None as there is no label in the CSV
    label_column = None

    X = df[feature_columns]
    y = None  # Initialize y as None
    # This block will now be skipped as label_column is None
    if label_column and label_column in df.columns:
        y = df[label_column]
        # Ensure labels are 0 for inliers and 1 for outliers if using evaluation
        # You might need to convert your labels accordingly
        # y = y.apply(lambda x: 1 if x == 'anomaly' else 0)  # Example conversion

    # --- Split Data ---
    # This 'else' block will now be executed
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y  # Stratify if possible
        )
    # If you don't have labels
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=42
        )
        # Set y_train and y_test to None if they don't exist
        y_train, y_test = None, None

    # --- Train Detector ---
    # Use ECOD model
    clf_name = "ECOD"
    # Pass the assumed contamination to the model for thresholding in predict/labels_
    clf = ECOD(contamination=contamination)
    clf.fit(X_train)

    # --- Predictions ---
    # get the prediction labels and outlier scores of the training data
    # ECOD provides labels_ after fit if contamination is set
    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1) based on contamination
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # --- Evaluate and Print Results ---
    # NOTE: This section requires ground truth labels (y_train, y_test).
    # If you don't have labels, comment out or remove these lines.
    if y_train is not None and y_test is not None:
        print("\nOn Training Data:")
        evaluate_print(clf_name, y_train, y_train_scores)
        print("\nOn Test Data:")
        evaluate_print(clf_name, y_test, y_test_scores)
    else:
        print("\nGround truth labels not provided. Skipping evaluation.")
        print(f"Training predictions (first 10): {y_train_pred[:10]}")
        print(f"Test predictions (first 10): {y_test_pred[:10]}")

    # --- Show Outlier Data Points ---
    print("\n--- Outliers Identified ---")

    # Find indices of outliers in training data
    train_outlier_indices = [i for i, label in enumerate(y_train_pred) if label == 1]
    if train_outlier_indices:
        print("\nPotential outliers found in Training Data (X_train):")
        # Use .iloc for integer-based indexing on the original DataFrame slice
        print(X_train.iloc[train_outlier_indices])
    else:
        print("\nNo outliers found in Training Data.")

    # Find indices of outliers in test data
    test_outlier_indices = [i for i, label in enumerate(y_test_pred) if label == 1]
    if test_outlier_indices:
        print("\nPotential outliers found in Test Data (X_test):")
        # Use .iloc for integer-based indexing on the original DataFrame slice
        print(X_test.iloc[test_outlier_indices])
    else:
        print("\nNo outliers found in Test Data.")

    # --- Visualize Results ---
    # NOTE: Visualization also benefits from ground truth labels.
    # It will still plot the data points and predictions, but coloring might
    # only show predicted labels if ground truth is missing.
    # Ensure X_train/X_test have 2 features for this visualization function.
    # If you have more than 2 features, consider dimensionality reduction (e.g., PCA)
    # before visualizing, or use a different visualization method.
    # The code will now print the message below because you have more than 2 features.
    if X_train.shape[1] == 2:
        visualize(clf_name, X_train.values, y_train.values if y_train is not None else y_train_pred,
                  X_test.values, y_test.values if y_test is not None else y_test_pred,
                  y_train_pred, y_test_pred, show_figure=True, save_figure=False)
    else:
        print("\nVisualization requires 2 features. Skipping visualization.")
        print(f"(Data has {X_train.shape[1]} features)")