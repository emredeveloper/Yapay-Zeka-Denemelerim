import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any
import random
from tqdm import tqdm
import os
import json
# Add imports for HuggingFace login
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Try to login to Hugging Face if token is available
def try_huggingface_login():
    """Try to login to Hugging Face using token from environment variable or user input"""
    token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if token:
        try:
            login(token=token)
            print("Successfully logged in to Hugging Face using environment token")
            return True
        except Exception as e:
            print(f"Error logging in with environment token: {e}")
    
    # Ask user for token if needed
    user_input = input("Do you want to provide a Hugging Face token? (y/n): ")
    if user_input.lower() == 'y':
        token = input("Enter your Hugging Face token: ")
        try:
            login(token=token)
            print("Successfully logged in to Hugging Face")
            return True
        except Exception as e:
            print(f"Error logging in: {e}")
    
    print("Proceeding without Hugging Face authentication")
    return False

class DocumentCorpus:
    """Simple document corpus for RAG system"""
    
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else []
        self.embeddings = None
        self.embedding_model = None
    
    def add_document(self, doc_text, doc_id=None):
        """Add a document to the corpus"""
        if doc_id is None:
            doc_id = len(self.documents)
        
        self.documents.append({
            'id': doc_id,
            'text': doc_text
        })
        # Reset embeddings since corpus changed
        self.embeddings = None
    
    def load_documents(self, file_path):
        """Load documents from a text file (one document per line)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.add_document(line.strip(), i)
    
    def generate_synthetic_docs(self, num_docs=100, topics=None):
        """Generate synthetic documents for testing with more structure"""
        if topics is None:
            topics = ["sports", "technology", "health", "finance", "education"]
        
        # Create more realistic documents with distinct topics
        for i in range(num_docs):
            topic = topics[i % len(topics)]  # Ensure even distribution of topics
            
            # Create topical content with more keywords related to the topic
            topic_related_words = []
            if topic == "sports":
                topic_related_words = ["team", "player", "game", "ball", "score", "win", "tournament"]
            elif topic == "technology":
                topic_related_words = ["computer", "software", "algorithm", "data", "innovation", "device", "digital"]
            elif topic == "health":
                topic_related_words = ["medical", "doctor", "patient", "treatment", "wellness", "hospital", "therapy"]
            elif topic == "finance":
                topic_related_words = ["money", "investment", "market", "stock", "bank", "trading", "economy"]
            elif topic == "education":
                topic_related_words = ["school", "student", "teacher", "learn", "curriculum", "class", "knowledge"]
            
            # Select some topic-related words
            selected_topic_words = random.sample(topic_related_words, k=min(3, len(topic_related_words)))
            
            # Add some generic words
            generic_words = [f"word_{random.randint(1, 100)}" for _ in range(random.randint(5, 10))]
            
            # Combine all words with the topic as the main subject
            all_words = [topic] + selected_topic_words + generic_words
            random.shuffle(all_words)
            
            # Create document with the topic appearing twice for emphasis
            doc_text = f"{topic} {' '.join(all_words)}"
            
            self.add_document(doc_text, i)
    
    def compute_embeddings(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Compute embeddings for all documents"""
        print("Computing document embeddings...")
        
        try:
            # Try to load the specified model
            self.embedding_model = SentenceTransformer(model_name)
        except (OSError, HfHubHTTPError) as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to TF-IDF vectorization...")
            
            # Fall back to TF-IDF if model loading fails
            vectorizer = TfidfVectorizer()
            texts = [doc['text'] for doc in self.documents]
            self.embeddings = vectorizer.fit_transform(texts).toarray()
            
            # Create a simple wrapper to maintain the same interface
            class TfidfWrapper:
                def __init__(self, vectorizer):
                    self.vectorizer = vectorizer
                
                def encode(self, texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    return self.vectorizer.transform(texts).toarray()
            
            self.embedding_model = TfidfWrapper(vectorizer)
            print(f"Computed TF-IDF embeddings with shape: {self.embeddings.shape}")
            return self.embeddings
        
        # If model loaded successfully, compute the embeddings
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts)
        
        print(f"Computed embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def get_documents(self):
        return self.documents


class RAGSystem:
    """Basic RAG system with configurable parameters"""
    
    def __init__(self, corpus, llm_function=None):
        self.corpus = corpus
        # Default LLM function just concatenates the context and query
        self.llm_function = llm_function if llm_function else self._default_llm
        
        # RAG parameters that can be tuned by RL
        self.params = {
            'top_k': 3,              # Number of documents to retrieve
            'similarity_threshold': 0.5,  # Minimum similarity score to include a document
            'context_window': 1000,  # Maximum context size (in chars)
            'reranking_weight': 0.7,  # Weight between initial and second stage ranking
            'exact_match_boost': 1.5,  # Boost factor for exact keyword matches
            'semantic_weight': 0.8,   # Weight of semantic similarity vs. keyword matching
            'diversity_factor': 0.2   # Factor to promote diversity in results
        }
    
    def _default_llm(self, context, query):
        """Improved mock LLM function that provides more detailed responses"""
        query_terms = set(query.lower().split())
        
        # Extract topics from query
        topics = [term for term in query_terms if term in ["sports", "technology", "health", "finance", "education"]]
        topic = topics[0] if topics else "general"
        
        # Extract relevant information from context
        doc_texts = []
        for line in context.split('\n'):
            if line.startswith('Doc '):
                doc_texts.append(line)
        
        if not doc_texts:
            return f"I don't have enough information to answer about {query}."
        
        # Generate a more informative response based on topic and retrieved documents
        response = f"Based on the information about {topic}, I found {len(doc_texts)} relevant documents.\n\n"
        
        # Add a summary of the retrieved information
        response += "The key information shows that:\n"
        for i, doc in enumerate(doc_texts[:3]):  # Summarize first 3 docs
            doc_content = doc.split(':', 1)[1].strip() if ':' in doc else doc
            response += f"- {doc_content[:50]}...\n"
        
        response += f"\nTo answer your query about '{query}', these documents suggest that "
        
        # Add a topic-specific conclusion
        if topic == "sports":
            response += "this relates to sporting activities or team events."
        elif topic == "technology":
            response += "this involves technological innovations or digital systems."
        elif topic == "health":
            response += "this concerns health-related information or medical practices."
        elif topic == "finance":
            response += "this pertains to financial matters or economic considerations."
        elif topic == "education":
            response += "this relates to educational content or learning materials."
        else:
            response += "the information is relevant to your search criteria."
        
        return response
    
    def retrieve(self, query, params=None):
        """Enhanced retrieval with better ranking and diversity"""
        if params is None:
            params = self.params
        
        if self.corpus.embeddings is None:
            self.corpus.compute_embeddings()
        
        # Handle potential errors with query encoding
        try:
            # Encode the query
            query_embedding = self.corpus.embedding_model.encode([query])[0]
            
            # Calculate semantic similarities
            semantic_similarities = cosine_similarity([query_embedding], self.corpus.embeddings)[0]
            
            # Calculate keyword matching scores
            query_terms = set(query.lower().split())
            keyword_scores = []
            
            for doc in self.corpus.documents:
                doc_terms = set(doc['text'].lower().split())
                # Count matching terms
                matches = len(query_terms.intersection(doc_terms))
                # Exact match bonus
                exact_match_score = matches * params.get('exact_match_boost', 1.0) / max(1, len(query_terms))
                keyword_scores.append(exact_match_score)
            
            # Combine semantic and keyword scores
            semantic_weight = params.get('semantic_weight', 0.8)
            combined_scores = (semantic_weight * semantic_similarities + 
                              (1 - semantic_weight) * np.array(keyword_scores))
            
            # Get top documents above threshold
            sorted_indices = np.argsort(combined_scores)[::-1]
            candidate_docs = []
            
            for idx in sorted_indices:
                if combined_scores[idx] >= params['similarity_threshold'] and len(candidate_docs) < params['top_k'] * 2:
                    candidate_docs.append({
                        'id': self.corpus.documents[idx]['id'],
                        'text': self.corpus.documents[idx]['text'],
                        'semantic_similarity': float(semantic_similarities[idx]),
                        'keyword_score': float(keyword_scores[idx]),
                        'combined_score': float(combined_scores[idx])
                    })
            
            # Apply diversity factor - prioritize docs with different content
            retrieved_docs = []
            if candidate_docs:
                retrieved_docs.append(candidate_docs[0])  # Add the top document
                
                # Add remaining documents with diversity consideration
                diversity_factor = params.get('diversity_factor', 0.2)
                
                for doc in candidate_docs[1:]:
                    if len(retrieved_docs) >= params['top_k']:
                        break
                        
                    # Calculate similarity to already selected documents
                    diversity_penalty = 0
                    for selected_doc in retrieved_docs:
                        # Simple text overlap for diversity calculation
                        doc_terms = set(doc['text'].lower().split())
                        selected_terms = set(selected_doc['text'].lower().split())
                        overlap = len(doc_terms.intersection(selected_terms)) / max(1, len(doc_terms.union(selected_terms)))
                        diversity_penalty += overlap
                    
                    # Average diversity penalty
                    diversity_penalty = diversity_penalty / len(retrieved_docs) if retrieved_docs else 0
                    
                    # Apply diversity-adjusted score
                    doc['diversity_adjusted_score'] = doc['combined_score'] * (1 - diversity_factor * diversity_penalty)
                    
                    # If score is still above threshold, include it
                    if doc['diversity_adjusted_score'] >= params['similarity_threshold']:
                        retrieved_docs.append(doc)
            
            # Sort final documents by score
            retrieved_docs = sorted(retrieved_docs, key=lambda x: x.get('diversity_adjusted_score', x['combined_score']), reverse=True)
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            # Return empty list on error
            return []
    
    def generate(self, query, params=None):
        """Generate a response to the query using RAG"""
        if params is None:
            params = self.params
            
        # Retrieve relevant documents
        docs = self.retrieve(query, params)
        
        # Create context from retrieved documents
        context = "\n".join([f"Doc {i+1}: {doc['text']}" for i, doc in enumerate(docs)])
        
        # Ensure context is within context window
        if len(context) > params['context_window']:
            context = context[:params['context_window']]
        
        # Generate response using LLM
        response = self.llm_function(context, query)
        
        return {
            'response': response,
            'retrieved_docs': docs,
            'params_used': params.copy()
        }


class RLAgent:
    """RL agent for optimizing RAG parameters"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.state_dim = 5  # Query features + current performance metrics
        self.action_dim = 7  # Number of RAG parameters to tune (updated for new parameters)
        
        # Define parameter ranges with new parameters
        self.param_ranges = {
            'top_k': (1, 10),
            'similarity_threshold': (0.1, 0.9),
            'context_window': (100, 2000),
            'reranking_weight': (0.0, 1.0),
            'exact_match_boost': (0.5, 3.0),
            'semantic_weight': (0.1, 0.9),
            'diversity_factor': (0.0, 0.5)
        }
        
        # Initialize Q-network with updated dimensions
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        # RL parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.min_epsilon = 0.05
        self.gamma = 0.99
        
        # Memory buffer for experience replay
        self.memory = []
        self.batch_size = 32
        self.memory_size = 10000
    
    def get_state(self, query, current_performance=None):
        """Convert query and performance to state representation"""
        if current_performance is None:
            current_performance = {'precision': 0.0, 'recall': 0.0, 'latency': 0.0}
        
        # Simple query features (length and avg word length)
        query_length = len(query)
        words = query.split()
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        
        state = [
            query_length / 100,  # Normalized query length
            avg_word_length / 10,  # Normalized avg word length
            current_performance.get('precision', 0.0),
            current_performance.get('recall', 0.0),
            current_performance.get('latency', 0.0) / 1000  # Normalized latency
        ]
        
        return torch.tensor(state, dtype=torch.float32)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: random action
            action = torch.rand(self.action_dim)
        else:
            # Exploit: use Q-network
            with torch.no_grad():
                action = self.q_network(state)
        
        # Convert action to RAG parameters
        params = self._action_to_params(action)
        return action, params
    
    def _action_to_params(self, action):
        """Convert normalized action values to RAG parameters"""
        params = {}
        param_keys = list(self.param_ranges.keys())
        
        for i, key in enumerate(param_keys):
            if i < len(action):  # Ensure we don't go out of bounds
                low, high = self.param_ranges[key]
                # Convert normalized action (0-1) to parameter range
                value = low + (high - low) * float(action[i])
                
                # Round values for discrete parameters
                if key == 'top_k':
                    value = max(1, int(round(value)))
                elif key == 'context_window':
                    value = max(100, int(round(value / 100) * 100))
                
                params[key] = value
        
        return params
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def update_q_network(self):
        """Update Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Compute Q values
        current_q = torch.sum(self.q_network(states) * actions, dim=1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = torch.max(self.q_network(next_states), dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Update network
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        return loss.item()


class QNetwork(nn.Module):
    """Q-Network for RL agent"""
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()  # Output normalized actions (0-1)
        )
    
    def forward(self, x):
        return self.network(x)


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_system, test_queries=None, ground_truth=None):
        self.rag_system = rag_system
        self.test_queries = test_queries if test_queries else []
        self.ground_truth = ground_truth if ground_truth else {}
        self.metrics_history = []
    
    def add_test_query(self, query, relevant_doc_ids=None):
        """Add a test query with ground truth relevant documents"""
        self.test_queries.append(query)
        if relevant_doc_ids is not None:
            self.ground_truth[query] = relevant_doc_ids
    
    def generate_synthetic_test_set(self, num_queries=20, topics=None):
        """Generate more realistic synthetic test queries and ground truth"""
        if topics is None:
            topics = ["sports", "technology", "health", "finance", "education"]
        
        corpus_docs = self.rag_system.corpus.documents
        
        # Create queries based on specific topics with more structure
        for topic in topics:
            # Find documents related to this topic
            topic_docs = [doc for doc in corpus_docs if topic.lower() in doc['text'].lower()]
            
            if not topic_docs:
                continue
                
            # Create several queries for each topic
            num_per_topic = max(1, num_queries // len(topics))
            
            for i in range(num_per_topic):
                # Select a random document for this topic
                doc = random.choice(topic_docs)
                
                # Create different types of queries
                if random.random() < 0.3:
                    # Type 1: Direct quote from document with topic
                    words = doc['text'].split()[:10]
                    query_words = random.sample(words, min(3, len(words)))
                    query = f"{topic} {' '.join(query_words)}"
                elif random.random() < 0.6:
                    # Type 2: Topic with related keywords
                    if topic == "sports":
                        keywords = ["team", "player", "game", "score"]
                    elif topic == "technology":
                        keywords = ["computer", "software", "data", "digital"]
                    elif topic == "health":
                        keywords = ["medical", "doctor", "patient", "treatment"]
                    elif topic == "finance":
                        keywords = ["money", "investment", "market", "bank"]
                    else:  # education
                        keywords = ["school", "student", "teacher", "learn"]
                        
                    query_keywords = random.sample(keywords, 2)
                    query = f"{topic} {' '.join(query_keywords)}"
                else:
                    # Type 3: Question format
                    query = f"What about {topic} and {random.choice(doc['text'].split())}"
                
                # Set the document as relevant
                self.add_test_query(query, [doc['id']])
                
                # Add similar documents as relevant (more precise relevance criteria)
                for other_doc in topic_docs:
                    if other_doc['id'] != doc['id']:
                        doc_words = set(doc['text'].lower().split())
                        other_words = set(other_doc['text'].lower().split())
                        # Calculate overlap
                        word_overlap = len(doc_words.intersection(other_words)) / len(doc_words.union(other_words))
                        
                        # If sufficient overlap or same topic, add as relevant
                        if word_overlap > 0.3 or topic.lower() in other_doc['text'].lower():
                            if query not in self.ground_truth:
                                self.ground_truth[query] = []
                            if doc['id'] not in self.ground_truth[query]:
                                self.ground_truth[query].append(other_doc['id'])
    
    def evaluate_query(self, query, params=None):
        """Evaluate RAG performance on a single query"""
        # Use regular Python timing instead of CUDA events for better compatibility
        import time
        
        start_time = time.time()
        result = self.rag_system.generate(query, params)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        metrics = {
            'latency': latency,
        }
        
        # If we have ground truth, calculate retrieval metrics
        if query in self.ground_truth:
            relevant_docs = set(self.ground_truth[query])
            retrieved_docs = set(doc['id'] for doc in result['retrieved_docs'])
            
            # Calculate precision, recall, F1
            true_positives = len(relevant_docs.intersection(retrieved_docs))
            precision = true_positives / max(1, len(retrieved_docs))
            recall = true_positives / max(1, len(relevant_docs))
            f1 = 2 * precision * recall / max(0.001, precision + recall)
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_retrieved': len(retrieved_docs),
                'num_relevant': len(relevant_docs)
            })
        
        return metrics, result
    
    def evaluate_test_set(self, params=None):
        """Evaluate on all test queries"""
        all_metrics = []
        
        for query in self.test_queries:
            metrics, _ = self.evaluate_query(query, params)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        agg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            agg_metrics[key] = sum(values) / len(values) if values else 0
        
        self.metrics_history.append(agg_metrics)
        return agg_metrics


def train_rl_agent(agent, evaluator, num_episodes=100):
    """Train the RL agent with an improved reward function"""
    rewards_history = []
    metrics_history = []
    
    # Initialize with default parameters
    best_reward = -float('inf')
    best_params = agent.rag_system.params.copy()
    
    print(f"Starting training for {num_episodes} episodes")
    for episode in tqdm(range(num_episodes)):
        total_reward = 0
        
        # Evaluate current parameters
        current_metrics = evaluator.evaluate_test_set()
        metrics_history.append(current_metrics)
        
        # For each test query
        for query in evaluator.test_queries:
            # Get current state
            current_state = agent.get_state(query, current_metrics)
            
            # Select action
            action_tensor, params = agent.select_action(current_state)
            
            # Apply action and get reward
            metrics, result = evaluator.evaluate_query(query, params)
            
            # Enhanced reward function with multiple components
            f1_reward = metrics.get('f1', 0) * 15  # Increased weight for F1
            latency_penalty = metrics.get('latency', 0) / 2000  # Reduced penalty for latency
            
            # Add a retrieval count component
            retrieval_count = len(result['retrieved_docs'])
            retrieval_reward = 0
            if retrieval_count == 0:
                retrieval_reward = -2  # Strong penalty for returning nothing
            elif retrieval_count <= 5:
                retrieval_reward = 1  # Reward for reasonable number of results
            else:
                retrieval_reward = 5 / retrieval_count  # Diminishing returns for too many results
                
            # Calculate total reward
            reward = f1_reward - latency_penalty + retrieval_reward
            total_reward += reward
            
            # Get next state
            next_state = agent.get_state(query, metrics)
            
            # Add to experience replay
            agent.add_experience(current_state, action_tensor, reward, next_state, False)
        
        # Update Q-network
        loss = agent.update_q_network()
        
        # Keep track of best parameters
        if total_reward > best_reward:
            best_reward = total_reward
            best_params = params.copy()
        
        rewards_history.append(total_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Avg Reward: {total_reward:.4f}, "
                  f"Epsilon: {agent.epsilon:.4f}, "
                  f"F1: {current_metrics.get('f1', 0):.4f}")
    
    # Apply best parameters
    agent.rag_system.params = best_params
    print(f"Training complete. Best parameters: {best_params}")
    
    return rewards_history, metrics_history, best_params


def visualize_training(rewards_history, metrics_history):
    """Visualize the training progress"""
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot F1 score
    plt.subplot(2, 2, 2)
    f1_scores = [m.get('f1', 0) for m in metrics_history]
    plt.plot(f1_scores)
    plt.title('F1 Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('F1 Score')
    
    # Plot precision/recall
    plt.subplot(2, 2, 3)
    precision = [m.get('precision', 0) for m in metrics_history]
    recall = [m.get('recall', 0) for m in metrics_history]
    plt.plot(precision, label='Precision')
    plt.plot(recall, label='Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot latency
    plt.subplot(2, 2, 4)
    latency = [m.get('latency', 0) for m in metrics_history]
    plt.plot(latency)
    plt.title('Latency (ms)')
    plt.xlabel('Episode')
    plt.ylabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig('rag_rl_training.png')
    plt.show()


def main():
    """Main function to run the entire system"""
    # Try to login to Hugging Face
    try_huggingface_login()
    
    # Create document corpus with more structured content
    corpus = DocumentCorpus()
    corpus.generate_synthetic_docs(num_docs=100)
    
    # Create RAG system
    rag_system = RAGSystem(corpus)
    
    # Create evaluator with more realistic test queries
    evaluator = RAGEvaluator(rag_system)
    evaluator.generate_synthetic_test_set(num_queries=20)
    
    # Create RL agent with expanded parameter space
    agent = RLAgent(rag_system)
    
    # Train RL agent with improved reward function
    rewards, metrics, best_params = train_rl_agent(agent, evaluator, num_episodes=50)
    
    # Visualize results
    visualize_training(rewards, metrics)
    
    # Evaluate final performance
    print("\nFinal RAG Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    final_metrics = evaluator.evaluate_test_set(best_params)
    print("\nFinal Performance Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save the best parameters
    with open('best_rag_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Detailed RAG system output examples
    print("\n=== RAG System Output Examples ===")
    sample_queries = evaluator.test_queries[:3]
    
    for i, query in enumerate(sample_queries):
        print(f"\n{'-'*50}")
        print(f"Example {i+1}:")
        print(f"Query: {query}")
        
        # Show ground truth information
        if query in evaluator.ground_truth:
            print(f"Ground truth relevant docs: {evaluator.ground_truth[query]}")
            relevant_texts = [corpus.documents[doc_id]['text'][:50] + "..." 
                             for doc_id in evaluator.ground_truth[query][:2]]
            print(f"Relevant content: {relevant_texts}")
        
        # Generate response with default parameters
        default_result = rag_system.generate(query)
        
        print("\nDefault Parameters Response:")
        print(f"Parameters used: {default_result['params_used']}")
        print(f"Retrieved {len(default_result['retrieved_docs'])} documents")
        for j, doc in enumerate(default_result['retrieved_docs'][:2]):
            print(f"  Doc {j+1}: \"{doc['text'][:50]}...\"")
            print(f"    Score: {doc.get('combined_score', doc.get('similarity', 0)):.4f}")
            if 'semantic_similarity' in doc:
                print(f"    Semantic: {doc['semantic_similarity']:.4f}, Keyword: {doc['keyword_score']:.4f}")
        print(f"\nResponse: {default_result['response'][:200]}...")
        
        # Generate response with optimized parameters
        optimized_result = rag_system.generate(query, best_params)
        
        print("\nOptimized Parameters Response:")
        print(f"Parameters used: {optimized_result['params_used']}")
        print(f"Retrieved {len(optimized_result['retrieved_docs'])} documents")
        for j, doc in enumerate(optimized_result['retrieved_docs'][:2]):
            print(f"  Doc {j+1}: \"{doc['text'][:50]}...\"")
            print(f"    Score: {doc.get('combined_score', doc.get('similarity', 0)):.4f}")
            if 'semantic_similarity' in doc:
                print(f"    Semantic: {doc['semantic_similarity']:.4f}, Keyword: {doc['keyword_score']:.4f}")
        print(f"\nResponse: {optimized_result['response'][:200]}...")
    
    # In-depth performance comparison
    print("\n=== Detailed Performance Comparison ===")
    default_metrics = evaluator.evaluate_test_set()
    optimized_metrics = evaluator.evaluate_test_set(best_params)
    
    print("\nMetric      | Default | Optimized | Improvement")
    print("------------|---------|-----------|------------")
    for key in ['precision', 'recall', 'f1', 'latency']:
        default_val = default_metrics.get(key, 0)
        optimized_val = optimized_metrics.get(key, 0)
        if key == 'latency':
            # For latency, lower is better
            improvement = ((default_val - optimized_val) / max(0.001, default_val)) * 100
        else:
            improvement = ((optimized_val - default_val) / max(0.001, default_val)) * 100
        print(f"{key.capitalize():<12}| {default_val:.4f}  | {optimized_val:.4f}   | {improvement:+.2f}%")


if __name__ == "__main__":
    main()
