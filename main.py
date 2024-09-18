# ECHO Recipe Implementation for Reasoning Enhancement (Enhanced Version)
# ========================================================================

# Step 1: Install Required Libraries
# ---------------------------------
# !pip install transformers sentence-transformers scikit-learn datasets rouge-score accelerate

# Step 2: Import Libraries
# ------------------------
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_metric
import random
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import logging

# Step 3: Configuration and Hyperparameters
# ------------------------------------------
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

# Device Configuration with Accelerator for Model Parallelism
accelerator = Accelerator()
device = accelerator.device

# Initialize ROUGE Metric
rouge = load_metric("rouge")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 4: Load Models with Memory Management
# ------------------------------------------
# Load LLAMA 3.1 - 70B fp8 Model with Accelerator for Model Parallelism
try:
    logger.info("Loading LLAMA model with Accelerator support...")
    tokenizer = LlamaTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        # For model parallelism, you can specify tensors parallelism etc.
    )
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.eval()
except Exception as e:
    logger.error(f"Error loading LLAMA model: {e}")
    raise e

# Load Sentence Embedding Model
try:
    logger.info("Loading Sentence-BERT model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
except Exception as e:
    logger.error(f"Error loading Sentence-BERT model: {e}")
    raise e

# Initialize Clustering Algorithm
clustering = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)

# Step 5: Load Dataset
# --------------------
# For demonstration, we'll use a sample dataset. Replace this with your actual dataset.
# The dataset should be a list of dictionaries with 'question' and 'answer' keys.

# Sample Dataset
dataset = [
    {"question": "What is the capital of France?", "answer": "Paris."},
    {"question": "Solve for x: 2x + 3 = 7.", "answer": "x = 2."},
    {"question": "Explain the theory of relativity.", "answer": "The theory of relativity, developed by Einstein, encompasses two interrelated theories: special relativity and general relativity..."},
    # Add more question-answer pairs as needed
]

# Ensure enough data for clustering
assert len(dataset) >= NUM_CLUSTERS, "Dataset size must be at least equal to the number of clusters."

# Step 6: Question Clustering
# ---------------------------
logger.info("Clustering questions...")
questions = [item['question'] for item in dataset]
try:
    question_embeddings = embedding_model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
except Exception as e:
    logger.error(f"Error encoding questions: {e}")
    raise e

question_embeddings_np = question_embeddings.cpu().numpy()

# Perform K-Means Clustering
clusters = clustering.fit_predict(question_embeddings_np)

# Group questions by cluster
clustered_questions = {}
for idx, cluster_id in enumerate(clusters):
    clustered_questions.setdefault(cluster_id, []).append(idx)

# Step 7: Demonstration Sampling
# ------------------------------
logger.info("Sampling demonstrations...")

def generate_zero_shot_cot_batch(questions_batch):
    prompts = [INITIAL_PROMPT_TEMPLATE.format(question=q) for q in questions_batch]
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_length=150,  # Dynamic max_length
            temperature=0.0,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=False,
            batch_size=BATCH_SIZE,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rationales = []
        for prompt, gen_text in zip(prompts, generated_texts):
            rationale = gen_text.replace(prompt, "").strip()
            # Optionally, you can implement more sophisticated rationale extraction here
            rationales.append(rationale)
        return rationales
    except Exception as e:
        logger.error(f"Error during zero-shot CoT generation: {e}")
        return [""] * len(questions_batch)

# Select representative questions
demonstrations = []
for cluster_id, indices in clustered_questions.items():
    for idx in indices:
        question = dataset[idx]['question']
        answer = dataset[idx]['answer']
        if len(tokenizer.encode(question)) < 2048:  # Adjust token limit as needed
            demonstrations.append({
                "question": question,
                "answer": answer,
                "rationale": "",  # To be filled
                "rouge_score": 0.0,  # Placeholder
            })
            break  # Select the first suitable question in the cluster

# Batch Processing for Initial Rationale Generation
batched_questions = [demo['question'] for demo in demonstrations]
rationales = []

logger.info("Generating initial rationales in batches...")
for i in tqdm(range(0, len(batched_questions), BATCH_SIZE), desc="Initial Rationales"):
    batch = batched_questions[i:i+BATCH_SIZE]
    batch_rationales = generate_zero_shot_cot_batch(batch)
    rationales.extend(batch_rationales)

# Assign rationales to demonstrations
for demo, rationale in zip(demonstrations, rationales):
    demo['rationale'] = rationale[:500]  # Optional: Trim to desired length

# Step 8: Demonstration Unification (Iterative Refinement)
# ---------------------------------------------------------
logger.info("Refining demonstrations iteratively...")

def generate_refined_cot_batch(refinement_prompts):
    try:
        inputs = tokenizer(refinement_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_length=300,  # Adjust as needed
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            batch_size=BATCH_SIZE,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refined_rationales = [gen_text.split("Answer:")[-1].strip() for gen_text in generated_texts]
        return refined_rationales
    except Exception as e:
        logger.error(f"Error during refined CoT generation: {e}")
        return [""] * len(refinement_prompts)

for iteration in range(NUM_ITERATIONS):
    logger.info(f"Refinement Iteration {iteration + 1}/{NUM_ITERATIONS}")
    # Shuffle demonstrations to ensure random order
    random.shuffle(demonstrations)
    
    # Prepare prompts for batch processing
    refinement_prompts = []
    questions_to_refine = []
    for demo in demonstrations:
        other_demos = [d for d in demonstrations if d != demo]
        demonstrations_str = "\n".join([f"Q: {d['question']}\nA: {d['rationale']}" for d in other_demos])
        prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            demonstrations=demonstrations_str,
            question=demo['question']
        )
        refinement_prompts.append(prompt)
        questions_to_refine.append(demo['question'])
    
    # Batch generate refined rationales
    refined_rationales = []
    logger.info("Generating refined rationales in batches...")
    for i in tqdm(range(0, len(refinement_prompts), BATCH_SIZE), desc="Refining Rationales"):
        batch_prompts = refinement_prompts[i:i+BATCH_SIZE]
        batch_rationales = generate_refined_cot_batch(batch_prompts)
        refined_rationales.extend(batch_rationales)
    
    # Evaluate and update demonstratations
    for demo, new_rationale in zip(demonstrations, refined_rationales):
        if not new_rationale:
            logger.warning(f"Empty rationale generated for question: {demo['question']}")
            continue  # Skip updating if rationale generation failed
        
        # Evaluate with ROUGE-L against gold answer
        try:
            rouge_result = rouge.compute(
                predictions=[new_rationale],
                references=[demo['answer']],
                rouge_types=["rougeL"]
            )
            rouge_l = rouge_result["rougeL"].mid.fmeasure
        except Exception as e:
            logger.error(f"Error computing ROUGE score: {e}")
            rouge_l = 0.0
        
        # Update demonstration
        demo['rationale'] = new_rationale[:500]  # Optional: Trim to desired length
        demo['rouge_score'] = rouge_l

# Step 9: Select Top Demonstrations (Quality and Diversity)
# ----------------------------------------------------------
logger.info("Selecting top demonstrations based on quality and diversity...")

# Additional Selection Criteria: Rationale Length and Complexity
MIN_RATIONALE_LENGTH = 50  # Minimum number of characters
MAX_RATIONALE_LENGTH = 500  # Maximum number of characters

# Sort demonstrations by ROUGE-L score descending
sorted_demos = sorted(demonstrations, key=lambda x: x['rouge_score'], reverse=True)

selected_demonstrations = []
selected_embeddings = []

def calculate_max_pairwise_similarity(new_embedding, selected_embeddings):
    if not selected_embeddings:
        return 0
    similarities = cosine_similarity([new_embedding], selected_embeddings)
    return similarities.max()

for demo in sorted_demos:
    if len(selected_demonstrations) >= TOP_DEMONSTRATIONS:
        break
    rationale_length = len(demo['rationale'])
    if not (MIN_RATIONALE_LENGTH <= rationale_length <= MAX_RATIONALE_LENGTH):
        continue  # Skip if rationale length is not within desired range
    # Compute embedding
    try:
        embedding = embedding_model.encode(demo['question'], convert_to_tensor=True).cpu().numpy()
    except Exception as e:
        logger.error(f"Error encoding question for diversity: {e}")
        continue
    # Calculate Max Pairwise Similarity
    if selected_embeddings:
        max_similarity = calculate_max_pairwise_similarity(embedding, selected_embeddings)
    else:
        max_similarity = 0
    if max_similarity < DIVERSITY_THRESHOLD:
        selected_demonstrations.append(demo)
        selected_embeddings.append(embedding)
        logger.debug(f"Selected demonstration: {demo['question']} with similarity {max_similarity:.2f}")

# Final check if enough demonstrations are selected
if len(selected_demonstrations) < TOP_DEMONSTRATIONS:
    logger.warning(f"Only {len(selected_demonstrations)} demonstrations selected. Consider adjusting hyperparameters or providing a larger dataset.")

# Display Selected Demonstrations
logger.info("\nSelected Demonstrations:")
for idx, demo in enumerate(selected_demonstrations, 1):
    logger.info(f"\nDemonstration {idx}:")
    logger.info(f"Q: {demo['question']}")
    logger.info(f"A: {demo['rationale']}")

# Step 10: Serve and Enjoy (Inference)
# -----------------------------------
def generate_answer(new_question):
    try:
        prompt = ""
        for demo in selected_demonstrations:
            prompt += f"Q: {demo['question']}\nA: {demo['rationale']}\n\n"
        prompt += f"Q: {new_question}\nA: Let's think step by step."
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=500,  # Adjust as needed
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.replace(prompt, "").strip()
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error: Unable to generate answer."

# Example Usage
logger.info("\n--- Inference Example ---")
new_question = "What is the largest planet in our solar system?"
answer = generate_answer(new_question)
logger.info(f"Q: {new_question}\nA: {answer}")

# Tips and Tricks:
# ----------------
# - Adjust hyperparameters (NUM_CLUSTERS, NUM_ITERATIONS, TOP_DEMONSTRATIONS, DIVERSITY_THRESHOLD) as needed.
# - Expand the dataset with more diverse question-answer pairs for better clustering and demonstration quality.
# - Implement additional evaluation metrics or advanced techniques like self-consistency for further enhancement.
# - Monitor GPU memory usage and adjust batch sizes accordingly to prevent out-of-memory errors.