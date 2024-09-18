import logging
import random
from typing import List, Dict, Tuple

import numpy as np
from accelerate import Accelerator
from datasets import load_metric
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer


# --- Configuration ---
LLM_MODEL_NAME = "meta-llama/Llama-3-70b-fp8"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_CLUSTERS = 10
NUM_ITERATIONS = 3
TOP_DEMONSTRATIONS = 5
DIVERSITY_THRESHOLD = 0.7
BATCH_SIZE = 4

# Prompt Templates
INITIAL_PROMPT_TEMPLATE = "Question: {question}\nLet's think step by step to find the answer."
REFINEMENT_PROMPT_TEMPLATE = "Based on the following Q&A pairs:\n{demonstrations}\nRefine the rationale for the question below.\nQuestion: {question}\nAnswer: Let's think step by step."
# --- End of Configuration ---


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End of Logging Setup ---


def load_models(accelerator: Accelerator) -> Tuple[LlamaTokenizer, LlamaForCausalLM, SentenceTransformer]:
    """Loads LLAMA and Sentence-BERT models with accelerator support."""
    device = accelerator.device
    try:
        logger.info("Loading LLAMA model with Accelerator support...")
        tokenizer = LlamaTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = LlamaForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model, tokenizer = accelerator.prepare(model, tokenizer)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading LLAMA model: {e}")
        raise e

    try:
        logger.info("Loading Sentence-BERT model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception as e:
        logger.error(f"Error loading Sentence-BERT model: {e}")
        raise e

    return tokenizer, model, embedding_model


def cluster_questions(questions: List[str], embedding_model: SentenceTransformer, num_clusters: int) -> Dict[int, List[int]]:
    """Clusters questions based on embeddings and returns a dictionary of cluster assignments."""
    logger.info("Clustering questions...")
    try:
        question_embeddings = embedding_model.encode(
            questions, convert_to_tensor=True, show_progress_bar=True
        )
    except Exception as e:
        logger.error(f"Error encoding questions: {e}")
        raise e

    question_embeddings_np = question_embeddings.cpu().numpy()
    clustering = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = clustering.fit_predict(question_embeddings_np)

    clustered_questions = {}
    for idx, cluster_id in enumerate(clusters):
        clustered_questions.setdefault(cluster_id, []).append(idx)

    return clustered_questions


def sample_demonstrations(
    dataset: List[Dict[str, str]], clustered_questions: Dict[int, List[int]], tokenizer: LlamaTokenizer
) -> List[Dict[str, str]]:
    """Samples representative questions from clusters and generates initial rationales."""
    demonstrations = []
    logger.info("Sampling demonstrations...")
    for cluster_id, indices in clustered_questions.items():
        for idx in indices:
            question = dataset[idx]['question']
            answer = dataset[idx]['answer']
            if len(tokenizer.encode(question)) < 2048:
                demonstrations.append(
                    {
                        "question": question,
                        "answer": answer,
                        "rationale": "",
                        "rouge_score": 0.0,
                    }
                )
                break

    return demonstrations


def generate_zero_shot_cot_batch(questions_batch: List[str], tokenizer: LlamaTokenizer, model: LlamaForCausalLM, device: torch.device, batch_size: int) -> List[str]:
    """Generates zero-shot chain-of-thought rationales in batches."""
    prompts = [INITIAL_PROMPT_TEMPLATE.format(question=q) for q in questions_batch]
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_length=150,
            temperature=0.0,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=False,
            batch_size=batch_size,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rationales = []
        for prompt, gen_text in zip(prompts, generated_texts):
            rationale = gen_text.replace(prompt, "").strip()
            rationales.append(rationale)
        return rationales
    except Exception as e:
        logger.error(f"Error during zero-shot CoT generation: {e}")
        return [""] * len(questions_batch)


def refine_demonstrations(
    demonstrations: List[Dict[str, str]],
    tokenizer: LlamaTokenizer,
    model: LlamaForCausalLM,
    device: torch.device,
    batch_size: int,
    num_iterations: int,
) -> List[Dict[str, str]]:
    """Iteratively refines rationales using other demonstrations as context."""
    logger.info("Refining demonstrations iteratively...")
    for iteration in range(num_iterations):
        logger.info(f"Refinement Iteration {iteration + 1}/{num_iterations}")
        random.shuffle(demonstrations)

        refinement_prompts = []
        questions_to_refine = []
        for demo in demonstrations:
            other_demos = [d for d in demonstrations if d != demo]
            demonstrations_str = "\n".join(
                [f"Q: {d['question']}\nA: {d['rationale']}" for d in other_demos]
            )
            prompt = REFINEMENT_PROMPT_TEMPLATE.format(
                demonstrations=demonstrations_str, question=demo['question']
            )
            refinement_prompts.append(prompt)
            questions_to_refine.append(demo['question'])

        refined_rationales = []
        logger.info("Generating refined rationales in batches...")
        for i in tqdm(
            range(0, len(refinement_prompts), batch_size), desc="Refining Rationales"
        ):
            batch_prompts = refinement_prompts[i : i + batch_size]
            batch_rationales = generate_refined_cot_batch(
                batch_prompts, tokenizer, model, device, batch_size
            )
            refined_rationales.extend(batch_rationales)

        for demo, new_rationale in zip(demonstrations, refined_rationales):
            if not new_rationale:
                logger.warning(
                    f"Empty rationale generated for question: {demo['question']}"
                )
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


def generate_refined_cot_batch(
    refinement_prompts: List[str], 
    tokenizer: LlamaTokenizer, 
    model: LlamaForCausalLM, 
    device: torch.device, 
    batch_size: int
) -> List[str]:
    """Generates refined rationales in batches."""
    try:
        inputs = tokenizer(
            refinement_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            batch_size=batch_size,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        refined_rationales = [
            gen_text.split("Answer:")[-1].strip() for gen_text in generated_texts
        ]
        return refined_rationales
    except Exception as e:
        logger.error(f"Error during refined CoT generation: {e}")
        return [""] * len(refinement_prompts)


def select_top_demonstrations(
    demonstrations: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
    top_k: int,
    diversity_threshold: float,
    min_rationale_length: int,
    max_rationale_length: int,
) -> List[Dict[str, str]]:
    """Selects top demonstrations based on quality, diversity, and rationale length."""
    logger.info("Selecting top demonstrations based on quality and diversity...")

    sorted_demos = sorted(demonstrations, key=lambda x: x['rouge_score'], reverse=True)
    selected_demonstrations = []
    selected_embeddings = []

    def calculate_max_pairwise_similarity(
        new_embedding: np.ndarray, selected_embeddings: List[np.ndarray]
    ) -> float:
        if not selected_embeddings:
            return 0
        similarities = cosine_similarity([new_embedding], selected_embeddings)
        return similarities.max()

    for demo in sorted_demos:
        if len(selected_demonstrations) >= top_k:
            break
        rationale_length = len(demo['rationale'])
        if not (min_rationale_length <= rationale_length <= max_rationale_length):
            continue
        try:
            embedding = embedding_model.encode(
                demo['question'], convert_to_tensor=True
            ).cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding question for diversity: {e}")
            continue

        if selected_embeddings:
            max_similarity = calculate_max_pairwise_similarity(embedding, selected_embeddings)
        else:
            max_similarity = 0
        if max_similarity < diversity_threshold:
            selected_demonstrations.append(demo)
            selected_embeddings.append(embedding)
            logger.debug(
                f"Selected demonstration: {demo['question']} with similarity {max_similarity:.2f}"
            )

    if len(selected_demonstrations) < top_k:
        logger.warning(
            f"Only {len(selected_demonstrations)} demonstrations selected. Consider adjusting hyperparameters or providing a larger dataset."
        )

    logger.info("\nSelected Demonstrations:")
    for idx, demo in enumerate(selected_demonstrations, 1):
        logger.info(f"\nDemonstration {idx}:")
        logger.info(f"Q: {demo['question']}")
        logger.info(f"A: {demo['rationale']}")

    return selected_demonstrations

def generate_answer(new_question: str, tokenizer: LlamaTokenizer, model: LlamaForCausalLM, selected_demonstrations: List[Dict[str, str]], device: torch.device) -> str:
   """Generates an answer to a new question using the selected demonstrations."""
   try:
       prompt = ""
       for demo in selected_demonstrations:
           prompt += f"Q: {demo['question']}\nA: {demo['rationale']}\n\n"
       prompt += f"Q: {new_question}\nA: Let's think step by step."

       inputs = tokenizer(prompt, return_tensors="pt").to(device)
       with torch.no_grad():
           outputs = model.generate(
               **inputs,
               max_length=500,
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


def main():
    """Main function to run the ECHO recipe."""
    accelerator = Accelerator()

    tokenizer, model, embedding_model = load_models(accelerator)
    
    # Sample Dataset
    dataset = [
        {"question": "What is the capital of France?", "answer": "Paris."},
        {"question": "Solve for x: 2x + 3 = 7.", "answer": "x = 2."},
        {"question": "Explain the theory of relativity.", "answer": "The theory of relativity, developed by Einstein, encompasses two interrelated theories: special relativity and general relativity..."},
    ]

    clustered_questions = cluster_questions(
        questions=[item["question"] for item in dataset],
        embedding_model=embedding_model,
        num_clusters=NUM_CLUSTERS,
    )

    demonstrations = sample_demonstrations(
        dataset=dataset,
        clustered_questions=clustered_questions,
        tokenizer=tokenizer,
    )

    demonstrations = [
        {**d, "rationale": r} for d, r in zip(demonstrations, generate_zero_shot_cot_batch(
            [d["question"] for d in demonstrations], 
            tokenizer=tokenizer, 
            model=model, 
            device=accelerator.device, 
            batch_size=BATCH_SIZE,
        ))
    ]


    refined_demonstrations = refine_demonstrations(
        demonstrations=demonstrations,
        tokenizer=tokenizer,
        model=model,
        device=accelerator.device,
        batch_size=BATCH_SIZE,
        num_iterations=NUM_ITERATIONS
    )


    selected_demonstrations = select_top_demonstrations(
        demonstrations=refined_demonstrations,
        embedding_model=embedding_model,
        top_k=TOP_DEMONSTRATIONS,
        diversity_threshold=DIVERSITY_THRESHOLD,
        min_rationale_length=50,
        max_rationale_length=500,
    )

    # Example inference
    new_question = "What is the largest planet in our solar system?"
    answer = generate_answer(
        new_question=new_question, 
        tokenizer=tokenizer, 
        model=model,
        selected_demonstrations=selected_demonstrations,
        device=accelerator.device
    )
    logger.info(f"Q: {new_question}\nA: {answer}")

if __name__ == "__main__":
    main()