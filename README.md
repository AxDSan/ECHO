# ECHO Recipe for Reasoning Enhancement

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-orange.svg)
![GPU Required](https://img.shields.io/badge/GPU-Required-green.svg)

Enhance the reasoning capabilities of large language models (LLMs) by automatically generating and refining demonstrations for few-shot chain-of-thought (CoT) prompting using the advanced ECHO Recipe.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Example Inference](#example-inference)
- [Dataset](#dataset)
- [Hyperparameters](#hyperparameters)
- [Tips and Tricks](#tips-and-tricks)
- [Error Handling and Robustness](#error-handling-and-robustness)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The **ECHO Recipe for Reasoning Enhancement** is a sophisticated framework designed to improve the reasoning capabilities of large language models (LLMs) such as `LLAMA 3.1 - 70B fp8`. By leveraging sentence embeddings, clustering algorithms, iterative refinement using ROUGE-L metrics, and advanced memory management techniques, ECHO generates high-quality demonstrations that enhance few-shot chain-of-thought prompting.

## Features

- **Automated Question Clustering**: Groups similar questions to ensure diverse and representative demonstrations using K-Means clustering.
- **Iterative Refinement**: Continuously improves rationales using zero-shot CoT prompting and ROUGE-L evaluation with batch processing.
- **Quality and Diversity Selection**: Ensures demonstrations are both high-quality and diverse to avoid redundancy by incorporating multiple selection criteria.
- **Advanced Memory Management**: Utilizes model parallelism with the `accelerate` library to handle large models efficiently.
- **Sophisticated Rationale Length Control**: Implements dynamic rationale length control using the `max_length` parameter.
- **Enhanced Prompt Engineering**: Employs advanced prompt templates to improve the quality and consistency of generated rationales.
- **Efficient Batching**: Processes demonstrations in batches during refinement to speed up the process.
- **Robust Error Handling**: Incorporates comprehensive error handling to manage potential issues during rationale generation and evaluation.

## Prerequisites

- **Hardware**:
  - **GPU(s)**: High-memory GPU(s) (e.g., NVIDIA A100) are required to handle the large LLAMA 3.1 model. Multiple GPUs are recommended for model parallelism.
- **Software**:
  - **Python**: 3.8 or higher
  - **PyTorch**: 1.9.0 or higher
  - **Accelerate**: For model parallelism
  - **Other Libraries**: See [Installation](#installation)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/echo-reasoning-enhancement.git
   cd echo-reasoning-enhancement
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

   _If you don't have a `requirements.txt`, install the dependencies manually:_

   ```bash
   pip install transformers sentence-transformers scikit-learn datasets rouge-score accelerate torch
   ```

4. **Download Models**

   - **LLAMA 3.1 - 70B fp8**: Ensure you have access to the LLAMA 3.1 model. Place it in the appropriate directory or provide the correct path in the configuration.
   - **Sentence-BERT**: The script will automatically download the `sentence-transformers/all-MiniLM-L6-v2` model.

## Configuration

Before running the script, configure the necessary parameters in the `echo_recipereasoning.py` file or through a separate configuration file as per your project structure.

### Key Configuration Parameters

```python
# Configuration Parameters
LLM_MODEL_NAME = "meta-llama/Llama-3-70b-fp8"  # Replace with the correct model path
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CLUSTERS = 10
NUM_ITERATIONS = 3
TOP_DEMONSTRATIONS = 5
DIVERSITY_THRESHOLD = 0.7  # Cosine similarity threshold
BATCH_SIZE = 4  # Number of prompts to process in a single batch

# Prompt Templates
INITIAL_PROMPT_TEMPLATE = "Question: {question}\nLet's think step by step to find the answer."
REFINEMENT_PROMPT_TEMPLATE = "Based on the following Q&A pairs:\n{demonstrations}\nRefine the rationale for the question below.\nQuestion: {question}\nAnswer: Let's think step by step."
```

- **Memory Management**:
  - Utilizes the `accelerate` library for model parallelism.
- **Rationale Length Control**:
  - Dynamic control using the `max_length` parameter in `model.generate`.
- **Prompt Engineering**:
  - Enhanced prompt templates (`INITIAL_PROMPT_TEMPLATE` and `REFINEMENT_PROMPT_TEMPLATE`) for better rationale quality.
- **Diversity Measurement**:
  - Uses Max Pairwise Cosine Similarity to ensure robust diversity.
- **Selection Criteria**:
  - Incorporates rationale length and complexity along with ROUGE-L scores.

## Usage

### Running the Script

1. **Prepare Your Dataset**

   Ensure your dataset is a list of dictionaries with `'question'` and `'answer'` keys. Place it in the script or load from an external source.

   **Example Dataset Structure**:

   ```python
   dataset = [
       {"question": "What is the capital of France?", "answer": "Paris."},
       {"question": "Solve for x: 2x + 3 = 7.", "answer": "x = 2."},
       {"question": "Explain the theory of relativity.", "answer": "The theory of relativity, developed by Einstein, encompasses two interrelated theories: special relativity and general relativity..."},
       # Add more question-answer pairs as needed
   ]
   ```

2. **Execute the Script**

   ```bash
   python echo_recipereasoning.py
   ```

   _Replace `echo_recipereasoning.py` with the actual script filename if different._

3. **Monitor the Process**

   The script includes logging to inform you about the progress of clustering, rationale generation, refinement, and selection. Ensure that your hardware resources are sufficient to handle the process.

### Example Inference

After running the script, an example inference will be displayed at the end:

```
--- Inference Example ---
Q: What is the largest planet in our solar system?
A: Jupiter is the largest planet in our solar system...
```

You can modify the `new_question` variable in the script to generate answers for different questions.

## Dataset

For demonstration purposes, a sample dataset is included within the script. For real-world applications, replace the sample dataset with a comprehensive set of question-answer pairs to maximize the effectiveness of the reasoning enhancement.

**Example**:

```python
dataset = [
    {"question": "What is the capital of France?", "answer": "Paris."},
    {"question": "Solve for x: 2x + 3 = 7.", "answer": "x = 2."},
    # Add more data...
]
```

## Hyperparameters

Adjust the following hyperparameters in the script to optimize performance based on your use case and available resources:

- `NUM_CLUSTERS`: Number of clusters for grouping similar questions.
- `NUM_ITERATIONS`: Number of times to refine the demonstrations.
- `TOP_DEMONSTRATIONS`: Number of top demonstrations to select for prompting.
- `DIVERSITY_THRESHOLD`: Threshold to ensure selected demonstrations are not too similar.
- `BATCH_SIZE`: Number of prompts to process in a single batch.

**Rationale Length Constraints**:

- `MIN_RATIONALE_LENGTH`: Minimum number of characters for a rationale.
- `MAX_RATIONALE_LENGTH`: Maximum number of characters for a rationale.

Example Configuration:

```python
# Configuration Parameters
LLM_MODEL_NAME = "meta-llama/Llama-3-70b-fp8"  # Replace with the correct model path
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CLUSTERS = 10
NUM_ITERATIONS = 3
TOP_DEMONSTRATIONS = 5
DIVERSITY_THRESHOLD = 0.7  # Cosine similarity threshold
BATCH_SIZE = 4  # Number of prompts to process in a single batch

# Rationale Length Constraints
MIN_RATIONALE_LENGTH = 50  # Minimum number of characters
MAX_RATIONALE_LENGTH = 500  # Maximum number of characters
```

## Tips and Tricks

- **Adjust Heat (Hyperparameters)**: Experiment with different values for clustering, iterations, batch sizes, and diversity thresholds to find the optimal settings for your dataset and hardware.
- **Add Flavor (Advanced Techniques)**: Incorporate techniques like self-consistency, ensemble models, or problem decomposition to further enhance reasoning capabilities.
- **Taste Test (Evaluation)**: Use a separate test set to evaluate the effectiveness of the enhanced reasoning and make necessary adjustments.
- **Monitor GPU Usage**: Keep an eye on GPU memory usage. Adjust `BATCH_SIZE` and other related parameters to prevent out-of-memory errors.
- **Expand the Dataset**: A larger and more diverse dataset can significantly improve clustering and demonstration quality.
- **Logging**: Utilize logging levels (`INFO`, `WARNING`, `ERROR`) to monitor the process and debug issues effectively.

## Error Handling and Robustness

The enhanced implementation includes comprehensive error handling to ensure robustness:

- **Try-Except Blocks**: Critical sections such as model loading, rationale generation, embedding computation, and ROUGE score calculation are wrapped in try-except blocks to gracefully handle exceptions.
- **Logging**: The script uses Python's `logging` module to provide informative messages, warnings, and error reports, aiding in debugging and monitoring.
- **Fallback Mechanisms**: If rationale generation fails, the system logs a warning and continues processing without crashing.
- **Final Selection Checks**: If not enough demonstrations are selected based on the criteria, the script logs a warning suggesting adjustments to hyperparameters or dataset size.

**Example Logs**:

```
INFO: Loading LLAMA model with Accelerator support...
INFO: Loading Sentence-BERT model...
INFO: Clustering questions...
INFO: Sampling demonstrations...
INFO: Generating initial rationales in batches...
INFO: Refining demonstrations iteratively...
INFO: Refinement Iteration 1/3
...
WARNING: Only 4 demonstrations selected. Consider adjusting hyperparameters or providing a larger dataset.
```

Ensure to review the logs to identify and address any potential issues during execution.

## Contributing

Contributions are welcome! Please follow these steps to contribute to the project:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

   Provide a clear description of your changes and the reasons behind them.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Sentence-BERT](https://www.sbert.net/)
- [Scikit-learn](https://scikit-learn.org/)
- [ROUGE Metric](https://github.com/google-research/google-research/tree/master/rouge)
- [Accelerate Library](https://github.com/huggingface/accelerate)

---

> **Disclaimer**: Running large models like LLAMA 3.1 - 70B fp8 requires substantial computational resources. Ensure you have the necessary hardware, comply with the model's licensing terms, and adjust hyperparameters to prevent out-of-memory errors.

```
 
