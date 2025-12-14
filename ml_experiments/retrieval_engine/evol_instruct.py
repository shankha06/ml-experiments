import json
import random

# --- Mock ask_llm function ---
def ask_llm(prompt, assumed_role="You are a helpful AI assistant."):
    """
    Mock function to simulate calling an LLM (GPT/Gemini).
    Replace this with your actual API call.
    """
    print(f"\n[LLM Call] Role: {assumed_role}")
    print(f"[LLM Call] Prompt: {prompt[:100]}...") # Print start of prompt
    
    # Simulating responses based on prompt keywords for demonstration
    prompt_lower = prompt.lower()
    if "generate a list of" in prompt_lower or "fact-based questions" in prompt_lower:
        return """1. What are the key financial highlights mentioned in the text?
2. Which platform launch drove the revenue growth?
3. How much did the revenue increase year-over-year?
"""
    elif "breadth" in prompt_lower or "rare vocabulary" in prompt_lower:
        return "What are the key financial highlights, barring any reference to the subsidiary's performance?"
    elif "constraint" in prompt_lower:
        return "What are the key financial highlights mentioned in the text, specifically excluding the fourth quarter?"
    elif "implications" in prompt_lower or "significance" in prompt_lower:
        return "How do the reported financial highlights impact the company's long-term investment strategy?"
    elif "specific scenarios" in prompt_lower or "concrete" in prompt_lower:
        return "If a competitor releases a similar product, how might the mentioned financial highlights change?"
    elif "reasoning" in prompt_lower and "difficult" not in prompt_lower: # differentiate generic reasoning from in-depth
        return "Given the market volatility described, why are the financial highlights considered positive?"
    elif "multi-step reasoning" in prompt_lower or "reasoning difficulty" in prompt_lower:
        return "Considering the debt restructuring mentioned earlier, how do the current financial highlights explain the subsequent stock price shift?"
    else:
        return "Evolution failed or generic response."

# --- Evolution Prompts ---

BASE_QUESTION_PROMPT = """
You are an expert question generator.
Given the following text chunk and its context, generate a list of 5 simple, fact-based questions that can be answered using the information in the chunk.

Chunk Context:
{chunk_context}

Chunk Content:
{chunk_content}

Generate ONLY the list of questions, one per line, numbered 1. to 5.
"""

EVOLUTION_PROMPTS = {
    "add_constraints": """
I want you to rewrite the following question to make it more complex by adding constraints to it. 
Original Question: {question}
Rewrite the question to include specific conditions or constraints (e.g., time period, specific subset, excluding certain factors).
""",

    "deepening": """
I want you to rewrite the following question to require a deeper understanding of the topic.
Original Question: {question}
Rewrite the question to ask about the implications, significance, or underlying causes related to the original topic.
""",

    "concretizing": """
I want you to rewrite the following question by replacing general concepts with specific scenarios or examples.
Original Question: {question}
Make the question more concrete and applied to a specific situation found in or inferred from the text.
""",

    "reasoning": """
I want you to rewrite the following question to require explicit reasoning to answer.
Original Question: {question}
Rewrite the question so that answering it requires connecting multiple pieces of information or logical deduction.
""",

    "in_breadth": """
I want you to rewrite the following question to increase its breadth.
Original Question: {question}
Rewrite the question to include a negative constraint or use more sophisticated/rare vocabulary to describe the concepts.
""",

    "in_depth": """
I want you to rewrite the following question to increase its reasoning difficulty significantly.
Original Question: {question}
Rewrite the question to require multi-step reasoning, looking at second-order effects or complex relationships.
"""
}

# --- Main Logic ---

def generate_base_questions(chunk_content, chunk_context):
    prompt = BASE_QUESTION_PROMPT.format(chunk_content=chunk_content, chunk_context=chunk_context)
    response = ask_llm(prompt, assumed_role="You are an expert educational content creator.")
    # Parse the response into a list
    questions = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() and line[1] == '.'):
             questions.append(line.split('.', 1)[1].strip())
        elif line: # Fallback for non-numbered lines if any, though prompt asks for numbering
             questions.append(line)
    return questions

def evolve_question(question, strategy):
    if strategy not in EVOLUTION_PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    prompt_template = EVOLUTION_PROMPTS[strategy]
    prompt = prompt_template.format(question=question)
    evolved_question = ask_llm(prompt, assumed_role="You are an expert at creating complex exam questions.")
    return evolved_question.strip()

def process_chunk(chunk_data, strategies=None):
    """
    Generates a base question and then applies specified evolution strategies.
    Returns a list of all generated questions (base + evolved).
    """
    if strategies is None:
        # Default to all strategies if not specified
        strategies = list(EVOLUTION_PROMPTS.keys())

    results = []
    
    # 1. Generate Base Questions
    base_questions = generate_base_questions(chunk_data['chunk_content'], chunk_data['chunk_context'])
    
    for bq in base_questions:
        results.append({
            "type": "base_question",
            "question": bq,
            "chunk_id": chunk_data['chunk_id']
        })
        
        # 2. Apply Random Evolution Strategy to each base question
        # Pick one random strategy
        strategy = random.choice(strategies)
        
        evolved_q = evolve_question(bq, strategy)
        results.append({
            "type": f"evolved_{strategy}",
            "question": evolved_q,
            "chunk_id": chunk_data['chunk_id'],
            "parent_question": bq
        })
        
    return results

# --- Main Execution Block ---

if __name__ == "__main__":
    # Sample Dataset Chunk
    sample_chunk = {
        "chunk_id": "chk_101",
        "chunk_content": "Acme Corp reported a revenue of $500 million in Q3 2024, a 10% increase year-over-year. This growth was driven by the launch of their new AI platform.",
        "chunk_context": "Introduction: Acme Corp Annual Report 2024. Chapter 3: Financial Performance.\nPrevious content: The company has been investing heavily in R&D.",
        "chunk_tags": ["finance", "revenue", "AI"],
        "doc_id": "doc_2024_report"
    }

    print(f"Processing chunk: {sample_chunk['chunk_id']}")
    generated_questions = process_chunk(sample_chunk)

    print("\n--- Generated Questions ---")
    for q in generated_questions:
        print(f"[{q['type'].upper()}]: {q['question']}")

