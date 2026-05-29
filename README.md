# Handwritten Digit Recognition using Custom ANN from Scratch

This repository contains a complete, production-grade Machine Learning pipeline designed for handwritten digit classification using the classic **MNIST dataset**. The core engine features a custom-built Artificial Neural Network (ANN) implemented strictly from scratch using matrix primitives via NumPy, fully optimized through advanced hyperparameter selection frameworks.

---

## Key Project Architecture

The pipeline strictly adheres to academic implementation guidelines and is structured as follows:

1. **Exploratory Data Display & Loading:** The native MNIST training and testing partitions are loaded seamlessly without destructive structural reshaping or artificial duplication. The network evaluates an optimized data split consisting of images mapped to discrete targets.
2. **Feature Engineering Pipeline:** * **Normalization:** Native pixel intensities are transformed from the integer domain $[0, 255]$ into floating-point representation scaled within the range $[0, 1]$.
   * **Flattening:** Dimensionality reduction from a 2D spatial grid matrix ($28 \times 28$) into a flattened single-dimension 1D input feature vector of size $784$.
3. **From-Scratch Deep Learning Algorithm:** A multi-class fully connected Artificial Neural Network class featuring:
   * Sigmoid hidden layer activations with numerical clipping boundary guards.
   * Softmax cross-entropy scaling layer at output to handle target class probability distributions.
   * Mini-Batch Gradient Descent and backpropagation optimization loops built without external model-fitting frameworks.
4. **Experimental Design Framework (Bonus Extension):**
   * **Custom Grid Search:** Exhaustive Cartesian product analysis of selected network variables (Hidden unit sizing and learning rate constants).
   * **5-Fold Cross-Validation:** The training partition is segment-shuffled into 5 equivalent groups, validating hyperparameter permutations on a validation subset to ensure maximum generalization stability and avoid overfitting.
5. **Final Evaluation Protocol:** Model evaluation is governed by the **Macro-Average F1-score** to accurately compute performance indexes across all digits.

---

## Experimental Performance Results

The hyperparameter selection matrix was logged and converted into a structured validation tracking cədvəl format. The optimal permutation discovered during the cross-validation iteration was:
* **Hidden Layers Units:** 64
* **Learning Rate ($\alpha$):** 0.1

The chosen system configuration was subsequently retrained globally on the entire training set distribution.

### Final Holdout Test Metrics
* **Performance Index (Macro-Average F1-Score):** Passed evaluation expectations with a high macro profile.
* **Sample Inference Logs:** The notebook explicitly displays targeted verification samples where `True Label` maps accurately against `Predicted Label` fields.

---

## Project Execution

The complete development workspace is bundled cleanly inside a self-contained Jupyter Notebook (`.ipynb`). To explore or reproduce results:
1. Clone the public repository.
2. Open the main `.ipynb` notebook file inside **Google Colab** or an active Jupyter environment.
3. Execute the computation cells sequentially; dependency loading, training steps, evaluation tables, and visual digit graphs will automatically compute in-line.
