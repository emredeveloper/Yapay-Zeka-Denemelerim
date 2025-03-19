import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio
from huggingface_hub import login
import getpass
import torch
import functools

nest_asyncio.apply()

WORKING_DIR = "./dickens"
# Define paths for local models
LOCAL_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# Modified approach to handle model completion
def create_model_completion_func(model_name, use_local=False):
    """Creates a model completion function with proper configuration."""
    print(f"Loading language model: {model_name}")
    
    # Configure model loading parameters
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
    }
    
    if use_local:
        model_kwargs["local_files_only"] = True
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Define the completion function that matches the expected interface
    def completion_func(prompt, **kwargs):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            num_beams=kwargs.get("num_beams", 1),
            early_stopping=False,
            do_sample=kwargs.get("do_sample", True),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            pad_token_id=tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    
    return completion_func


async def initialize_rag(use_local=False):
    """Initialize RAG with option to use local models"""
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Set offline mode for transformers if using local models
    if use_local:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"Using local models from: {LOCAL_MODELS_DIR}")
    
    # Configure torch for memory optimization
    if torch.cuda.is_available():
        print("CUDA available. Configuring for GPU usage.")
        # Set memory fraction to use
        torch.cuda.set_per_process_memory_fraction(0.85)
    else:
        print("No CUDA available. Using CPU.")
    
    # Create the model completion function
    model_func = create_model_completion_func(llm_model_name, use_local)
    
    # Load embedding models
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_name,
        local_files_only=use_local
    )
    
    embedding_model = AutoModel.from_pretrained(
        embedding_model_name,
        local_files_only=use_local
    )
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=model_func,  # Use our function that has the correct signature
        llm_model_name=llm_model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=embedding_tokenizer,
                embed_model=embedding_model,
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def setup_huggingface_auth(max_retries=3):
    """Optional authentication with Hugging Face Hub with retry mechanism."""
    print("\nHugging Face configuration:")
    print("1. Use online mode with authentication")
    print("2. Use local models only (offline mode)")
    
    choice = input("Select option (1/2): ").strip()
    
    if choice == "2":
        print("Using local models only. No authentication required.")
        # Check if models exist locally
        embedding_path = os.path.join(LOCAL_MODELS_DIR, "models--sentence-transformers--all-MiniLM-L6-v2")
        llm_path = os.path.join(LOCAL_MODELS_DIR, "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
        
        if not (os.path.exists(embedding_path) and os.path.exists(llm_path)):
            print("\nWARNING: Required models may not be available locally.")
            print("If you encounter errors, please download the models first or switch to online mode.")
        
        return {"use_local": True}
    
    # Online mode with authentication
    for attempt in range(max_retries):
        try:
            # Try to get token from environment variable first
            token = os.environ.get("HF_TOKEN")
            
            # If not in environment or we're retrying after failure
            if not token or attempt > 0:
                print("\nHugging Face authentication required.")
                print("Please enter your Hugging Face token (visit https://huggingface.co/settings/tokens to get one)")
                token = getpass.getpass("HF Token: ")
            
            # Login to Huggingface
            login(token=token)
            
            print("Successfully logged in to Huggingface Hub")
            return {"use_local": False}
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Authentication failed: {str(e)}")
                print(f"Retrying... (Attempt {attempt+1}/{max_retries})")
            else:
                print(f"Failed to authenticate after {max_retries} attempts.")
                print("Error details:", str(e))
                print("\nWould you like to try using local models instead? (y/n)")
                if input().lower().startswith('y'):
                    return {"use_local": True}
                return None


def get_sample_text():
    """Returns a short sample text for testing purposes."""
    return """
    Charles Dickens was an English writer and social critic. He created some of the world's best-known fictional 
    characters and is regarded by many as the greatest novelist of the Victorian era. His works enjoyed unprecedented 
    popularity during his lifetime, and by the 20th century, critics and scholars had recognized him as a literary genius.
    
    Among his most famous works are "A Christmas Carol", "Oliver Twist", "Great Expectations", "David Copperfield", 
    and "A Tale of Two Cities". Dickens's novels often feature themes of social inequality, poverty, and the 
    struggles of the working class in Victorian England. His characters are memorably unique, and his writing style 
    combines humor, pathos, and a keen observation of human nature.
    
    Dickens was born in Portsmouth, England, in 1812. His father was imprisoned for debt when Dickens was young, 
    and Dickens was forced to leave school and work in a factory. This experience deeply affected him and influenced 
    his later writings about social injustice and the plight of the poor.
    """

def main():
    # Call the authentication function before initializing RAG
    auth_config = setup_huggingface_auth()
    if auth_config is None:
        print("\nCannot proceed without authentication or local models.")
        return
    
    # Ask user if they want to use the full book or a sample text
    print("\nText selection:")
    print("1. Use the full book from book.txt")
    print("2. Use a shorter sample text for testing")
    
    text_choice = input("Select option (1/2): ").strip()
    
    try:
        print("Initializing RAG system...")
        rag = asyncio.run(initialize_rag(use_local=auth_config["use_local"]))

        print("Loading document...")
        if text_choice == "2":
            # Use the shorter sample text
            content = get_sample_text()
            print("Using shorter sample text for testing")
        else:
            # Use the full book
            try:
                with open("./book.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                print(f"Document loaded: {len(content)} characters")
            except FileNotFoundError:
                print("book.txt not found. Using sample text instead.")
                content = get_sample_text()
        
        rag.insert(content)
        print("Document processing complete")

        # Perform naive search
        print(
            rag.query(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            rag.query(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            rag.query(
                "What are the top themes in this story?", param=QueryParam(mode="global")
            )
        )

        # Perform hybrid search
        print(
            rag.query(
                "What are the top themes in this story?", param=QueryParam(mode="hybrid")
            )
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("- If using local mode, check if models are downloaded")
        print("- If using online mode, check if your HF_TOKEN is valid")
        print("- Verify that you have internet connection if using online mode")
        print("- The model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' might require special access")
        print("- If experiencing memory issues, try a smaller model")
        print("- Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 in your environment")
        print("- If seeing sliding window warnings, you can try a different model architecture")
        print("- Check if the expected function signatures match with LightRAG's requirements")
        print("- Try using the original hf_model_complete function if custom function fails")


if __name__ == "__main__":
    main()