import os
import random
import numpy as np
import util
import time
import torch

from perceptron import PerceptronClassifier
from nn_scratch import ThreeLayerNet
from torchNN import ThreeLayerNN, train_model_with_val

def load_data(images_file, labels_file, width=60, height=70):
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f]
    data = []
    with open(images_file, 'r') as f:
        for _ in labels:
            datum = util.Counter()
            for r in range(height):
                row = f.readline().rstrip('\n')
                for c in range(width):
                    datum[(r, c)] = 1 if row[c] in ['#', '+'] else 0
            data.append(datum)
    return data, labels

def counters_to_matrix(data, width=60, height=70):
    N = len(data)
    D = width * height
    X = np.zeros((N, D), dtype=np.float32)
    for i, datum in enumerate(data):
        for (r, c), v in datum.items():
            X[i, r * width + c] = v
    return X

def evaluate_model(model_type, train_data, train_labels, test_data, test_labels, X_test, X_val, y_val):
    percentages = list(range(10, 101, 10))
    mean_accuracies = []
    std_accuracies = []
    mean_times = []

    for pct in percentages:
        accs = []
        times = []
        print(f"\n--- Training with {pct}% of data ---")
        predictions_for_display = None

        for _ in range(5):
            subset_size = int(len(train_data) * pct / 100)
            indices = random.sample(range(len(train_data)), subset_size)
            data_subset = [train_data[i] for i in indices]
            labels_subset = [train_labels[i] for i in indices]
            X_train = counters_to_matrix(data_subset, width=60, height=70)
            y_train = np.array(labels_subset)

            if model_type == 'perceptron':
                classifier = PerceptronClassifier(legalLabels=[0, 1], max_iterations=3)
                start = time.time()
                classifier.train(data_subset, labels_subset, [], [])
                end = time.time()
                predictions = classifier.classify(test_data)

            elif model_type == 'nn_scratch':
                classifier = ThreeLayerNet(X_train.shape[1], 128, 64, 2)
                start = time.time()
                classifier.train(X_train, y_train, X_val, y_val, epochs=10)
                end = time.time()
                predictions = classifier.predict(X_test)

            elif model_type == 'pytorch':
                model = ThreeLayerNN(X_train.shape[1], output_size=2)
                start = time.time()
                best_model = train_model_with_val(model, X_train, y_train, X_val, y_val, epochs=10)
                end = time.time()
                model.load_state_dict(best_model)
                with torch.no_grad():
                    logits = model(torch.tensor(X_test, dtype=torch.float32))
                    predictions = torch.argmax(logits, dim=1).numpy()

            else:
                raise ValueError("Unknown model type")

            accuracy = np.mean(np.array(predictions) == np.array(test_labels))
            accs.append(accuracy)
            times.append(end - start)
            predictions_for_display = predictions

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)

        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)
        mean_times.append(mean_time)

        print("Predicted vs Actual (first 20 samples):")
        print("Predicted:", predictions_for_display[:20])
        print("Actual:   ", test_labels[:20])
        print(f">>> [{pct}%]  Accuracy: {mean_acc:.4f} ± {std_acc:.4f} | Time: {mean_time:.2f}s")

    return percentages, mean_accuracies, std_accuracies, mean_times

if __name__ == "__main__":
    # Set model type ('perceptron', 'nn_scratch', or 'pytorch')
    model_type = 'nn_scratch'  

    # Load training, validation, and test data
    train_data, train_labels = load_data('data/facedata/facedatatrain', 'data/facedata/facedatatrainlabels')
    val_data, val_labels = load_data('data/facedata/facedatavalidation', 'data/facedata/facedatavalidationlabels')
    test_data, test_labels = load_data('data/facedata/facedatatest', 'data/facedata/facedatatestlabels')

    # Convert data to matrix format
    X_val = counters_to_matrix(val_data, width=60, height=70)
    X_test = counters_to_matrix(test_data, width=60, height=70)

    print(f"\n========== Running {model_type.upper()} ==========")

    _, mean_accuracies, _, _ = evaluate_model(
        model_type, train_data, train_labels,
        test_data, test_labels, X_test, X_val, np.array(val_labels)
    )

    overall_avg_accuracy = np.mean(mean_accuracies)
    print(f"\nOverall Average Accuracy across all percentages: {overall_avg_accuracy:.4f}")
