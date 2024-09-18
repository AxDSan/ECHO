import os
import asyncio
import logging
from typing import List, Dict
import httpx
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from tqdm.asyncio import tqdm_asyncio

# Configuration Parameters
API_URL = "https://openrouter.ai/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_MODEL_NAME = "openrouter/google/gemini-flash-1.5-exp"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CLUSTERS = 10
NUM_ITERATIONS = 3
TOP_DEMONSTRATIONS = 5
DIVERSITY_THRESHOLD = 0.7
BATCH_SIZE = 4

INITIAL_PROMPT_TEMPLATE = "Question: {question}\nLet's think step by step to find the answer."
REFINEMENT_PROMPT_TEMPLATE = "Based on the following Q&A pairs:\n{demonstrations}\nRefine the rationale for the question below.\nQuestion: {question}\nAnswer: Let's think step by step."

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTTP Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

async def send_prompt_async(prompts: List[str], max_length: int, temperature: float, top_p: float, repetition_penalty: float) -> List[str]:
    """
    Asynchronously send prompts to the hosted model and retrieve responses.
    """
    async with httpx.AsyncClient() as client:
        tasks = []
        for prompt in prompts:
            payload = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "num_return_sequences": 1,
                # Add other parameters as needed
            }
            tasks.append(client.post(API_URL, headers=headers, json=payload))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        rationales = []
        for response in responses:
            if isinstance(response, Exception):
                rationales.append("")
                logger.error(f"Request failed: {response}")
            elif response.status_code == 200:
                data = response.json()
                rationale = data.get("choices", [{}])[0].get("text", "").strip()
                rationales.append(rationale)
            else:
                rationales.append("")
                logger.error(f"Error {response.status_code}: {response.text}")
        
        return rationales

async def generate_zero_shot_cot_batch_api(questions_batch: List[str]) -> List[str]:
    prompts = [INITIAL_PROMPT_TEMPLATE.format(question=q) for q in questions_batch]
    rationales = await send_prompt_async(prompts, max_length=150, temperature=0.0, top_p=0.95, repetition_penalty=1.2)
    return rationales

async def generate_refined_cot_batch_api(refinement_prompts: List[str]) -> List[str]:
    rationales = await send_prompt_async(refinement_prompts, max_length=300, temperature=0.7, top_p=0.95, repetition_penalty=1.2)
    return rationales

def load_sentence_transformer_model() -> SentenceTransformer:
    """Loads the Sentence-BERT embedding model."""
    try:
        logger.info("Loading Sentence-BERT model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        return embedding_model
    except Exception as e:
        logger.error(f"Error loading Sentence-BERT model: {e}")
        raise e

def cluster_questions(questions: List[str], embedding_model: SentenceTransformer, num_clusters: int) -> Dict[int, List[int]]:
    """Clusters questions based on their semantic embeddings."""
    logger.info("Clustering questions...")
    question_embeddings = embedding_model.encode(questions, convert_to_tensor=False, show_progress_bar=True)
    clustering = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = clustering.fit_predict(question_embeddings)
    
    clustered_questions = {}
    for idx, cluster_id in enumerate(clusters):
        clustered_questions.setdefault(cluster_id, []).append(idx)
    
    return clustered_questions

def sample_demonstrations(dataset: List[Dict[str, str]], clustered_questions: Dict[int, List[int]]) -> List[Dict[str, str]]:
    """Selects representative questions from each cluster."""
    demonstrations = []
    logger.info("Sampling demonstrations...")
    for cluster_id, indices in clustered_questions.items():
        for idx in indices:
            question = dataset[idx]['question']
            answer = dataset[idx]['answer']
            # Assume token length check is handled at API level
            demonstrations.append({
                "question": question,
                "answer": answer,
                "rationale": "",
                "rouge_score": 0.0,
            })
            break  # Select the first suitable question in the cluster
    return demonstrations

def select_top_demonstrations(
    demonstrations: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
    top_k: int,
    diversity_threshold: float,
    min_rationale_length: int = 50,
    max_rationale_length: int = 500,
) -> List[Dict[str, str]]:
    """Selects top demonstrations based on quality and diversity."""
    logger.info("Selecting top demonstrations based on quality and diversity...")
    sorted_demos = sorted(demonstrations, key=lambda x: x['rouge_score'], reverse=True)
    selected_demonstrations = []
    selected_embeddings = []
    
    for demo in sorted_demos:
        if len(selected_demonstrations) >= top_k:
            break
        rationale_length = len(demo['rationale'])
        if not (min_rationale_length <= rationale_length <= max_rationale_length):
            continue
        try:
            embedding = embedding_model.encode(demo['question'], convert_to_tensor=False)
        except Exception as e:
            logger.error(f"Error encoding question for diversity: {e}")
            continue
        if selected_embeddings:
            similarities = cosine_similarity([embedding], selected_embeddings)
            max_similarity = similarities.max()
        else:
            max_similarity = 0
        if max_similarity < diversity_threshold:
            selected_demonstrations.append(demo)
            selected_embeddings.append(embedding)
            logger.debug(f"Selected demonstration: {demo['question']} with similarity {max_similarity:.2f}")
    
    if len(selected_demonstrations) < top_k:
        logger.warning(f"Only {len(selected_demonstrations)} demonstrations selected. Consider adjusting hyperparameters or providing a larger dataset.")
    
    logger.info("\nSelected Demonstrations:")
    for idx, demo in enumerate(selected_demonstrations, 1):
        logger.info(f"\nDemonstration {idx}:")
        logger.info(f"Q: {demo['question']}")
        logger.info(f"A: {demo['rationale']}")
    
    return selected_demonstrations

async def refine_demonstrations(
    demonstrations: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
    num_iterations: int,
) -> List[Dict[str, str]]:
    """Refines rationales by iterative regeneration using other demonstrations as context."""
    logger.info("Refining demonstrations iteratively...")
    for iteration in range(num_iterations):
        logger.info(f"Refinement Iteration {iteration + 1}/{num_iterations}")
        random.shuffle(demonstrations)
        
        refinement_prompts = []
        for demo in demonstrations:
            other_demos = [d for d in demonstrations if d != demo]
            demonstrations_str = "\n".join([f"Q: {d['question']}\nA: {d['rationale']}" for d in other_demos])
            prompt = REFINEMENT_PROMPT_TEMPLATE.format(
                demonstrations=demonstrations_str,
                question=demo['question']
            )
            refinement_prompts.append(prompt)
        
        refined_rationales = await generate_refined_cot_batch_api(refinement_prompts)
        
        for demo, new_rationale in zip(demonstrations, refined_rationales):
            if not new_rationale:
                logger.warning(f"Empty rationale generated for question: {demo['question']}")
                continue
            try:
                rouge_result = rouge.compute(
                    predictions=[new_rationale],
                    references=[demo['answer']],
                    rouge_types=["rougeL"],
                )
                rouge_l = rouge_result["rougeL"].mid.fmeasure
            except Exception as e:
                logger.error(f"Error computing ROUGE score: {e}")
                rouge_l = 0.0
            demo['rationale'] = new_rationale[:500]
            demo['rouge_score'] = rouge_l
    return demonstrations

async def generate_answer_api(new_question: str, selected_demonstrations: List[Dict[str, str]]) -> str:
    """Generates an answer to a new question using the selected demonstrations via API."""
    try:
        prompt = ""
        for demo in selected_demonstrations:
            prompt += f"Q: {demo['question']}\nA: {demo['rationale']}\n\n"
        prompt += f"Q: {new_question}\nA: Let's think step by step."
        
        payload = {
            "prompt": prompt,
            "max_length": 500,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "num_return_sequences": 1,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("choices", [{}])[0].get("text", "").strip()
            return answer
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
            return "Error: Unable to generate answer."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error: Unable to generate answer."

async def main():
    embedding_model = load_sentence_transformer_model()
    
    # Replace with actual dataset loading
    dataset = [
        {"question": "There are 15 trees originally. After some more were planted, there are now 21 trees. How many trees were planted?", "answer": "6"},
        {"question": "Jason had 20 lollipops. He gave some to Denny and now has 12 lollipops left. How many lollipops did Jason give to Denny?", "answer": "8"},
        {"question": "There are 3 cars in the parking lot. 2 more cars arrive. How many cars are in the parking lot now?", "answer": "5"},
    ]
    
    clustered_questions = cluster_questions(
        questions=[item["question"] for item in dataset],
        embedding_model=embedding_model,
        num_clusters=NUM_CLUSTERS,
    )
    
    demonstrations = sample_demonstrations(
        dataset=dataset,
        clustered_questions=clustered_questions,
        tokenizer=None,  # Not needed for clustering
    )
    
    # Generate initial rationales using API
    batched_questions = [demo['question'] for demo in demonstrations]
    logger.info("Generating initial rationales in batches...")
    initial_rationales = await generate_zero_shot_cot_batch_api(batched_questions)
    
    for demo, rationale in zip(demonstrations, initial_rationales):
        demo['rationale'] = rationale[:500]  # Trim as necessary
    
    # Refine rationales
    refined_demonstrations = await refine_demonstrations(
        demonstrations=demonstrations,
        embedding_model=embedding_model,
        num_iterations=NUM_ITERATIONS,
    )
    
    # Select top demonstrations
    selected_demonstrations = select_top_demonstrations(
        demonstrations=refined_demonstrations,
        embedding_model=embedding_model,
        top_k=TOP_DEMONSTRATIONS,
        diversity_threshold=DIVERSITY_THRESHOLD,
        min_rationale_length=50,
        max_rationale_length=500,
    )
    
    # Inference with new question
    new_question = "What is the largest planet in our solar system?"
    answer = await generate_answer_api(new_question, selected_demonstrations)
    logger.info(f"Q: {new_question}\nA: {answer}")

if __name__ == "__main__":
    asyncio.run(main())