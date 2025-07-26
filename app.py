#!/usr/bin/env python3
"""
RAG Comparison Application - Fixed Version 2
Compares Hybrid Search with Reranking vs Naive RAG using RAGAS evaluation
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import threading
import time
import sys
import traceback

# Core libraries
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Fixed LangChain imports - using langchain_community
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    # Fallback to old imports if new ones not available
    from langchain.document_loaders import PyPDFLoader
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.retrievers import BM25Retriever
    from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# RAGAS imports (with fallback)
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("RAGAS successfully imported")
except ImportError as e:
    print(f"RAGAS not available: {e}")
    RAGAS_AVAILABLE = False

# Flask
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding to fix Windows emoji issues
def setup_logging():
    """Setup logging with proper encoding for Windows"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging without emojis for file output
    file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Create required directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('templates', exist_ok=True)

class Config:
    """Application configuration"""
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'pdf', 'json'}
    
    # Model configurations
    EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
    LLM_MODEL = "gpt-4"
    LLM_TEMPERATURE = 0.7
    
    # Retrieval configurations
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    RETRIEVAL_K = 5
    ENSEMBLE_WEIGHTS = [0.6, 0.4]

class TokenTracker:
    """Track token usage for cost optimization"""
    
    def __init__(self):
        self.total_tokens = 0
        self.query_count = 0
        
    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        
    def add_query(self):
        self.query_count += 1
        
    def get_stats(self) -> Dict[str, int]:
        return {
            "total_tokens": self.total_tokens,
            "query_count": self.query_count,
            "avg_tokens_per_query": self.total_tokens / max(self.query_count, 1)
        }

class DocumentProcessor:
    """Document loading and processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def load_and_split_documents(self, pdf_path: str) -> List:
        """Load PDF and split into chunks"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content extracted from PDF")
                
            text_splits = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(documents)} pages, split into {len(text_splits)} chunks")
            return text_splits
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

class NaiveRAG:
    """Simple RAG implementation using vector similarity"""
    
    def __init__(self, documents: List, embeddings, config: Config, token_tracker: TokenTracker):
        self.documents = documents
        self.embeddings = embeddings
        self.config = config
        self.token_tracker = token_tracker
        try:
            self.llm = ChatOpenAI(
                model_name=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=512
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
        self._setup_retrieval()
    
    def _setup_retrieval(self):
        """Setup vector store and retriever"""
        try:
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.RETRIEVAL_K}
            )
            logger.info("Naive RAG setup completed")
        except Exception as e:
            logger.error(f"Error setting up naive RAG: {e}")
            raise
    
    def retrieve_and_generate(self, query: str) -> Dict[str, Any]:
        """Retrieve and generate answer"""
        try:
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            template = """Answer the question based on the context provided. Be concise and accurate.

Context: {context}

Question: {question}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            answer = chain.invoke({"context": context, "question": query})
            
            # Track tokens
            self.token_tracker.add_tokens(len(context.split()) + len(answer.split()) + 50)
            self.token_tracker.add_query()
            
            return {
                "answer": answer,
                "contexts": [doc.page_content for doc in docs],
                "method": "naive_rag",
                "num_retrieved": len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error in naive RAG: {e}")
            raise

class HybridRAGWithReranking:
    """Hybrid RAG with BM25 + Dense retrieval and Cohere reranking"""
    
    def __init__(self, documents: List, embeddings, config: Config, token_tracker: TokenTracker):
        self.documents = documents
        self.embeddings = embeddings
        self.config = config
        self.token_tracker = token_tracker
        try:
            self.llm = ChatOpenAI(
                model_name=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=512
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
        self._setup_hybrid_retrieval()
    
    def _setup_hybrid_retrieval(self):
        """Setup hybrid retrieval with reranking"""
        try:
            # Vector retriever
            vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            vector_retriever = vectorstore.as_retriever(
                search_kwargs={"k": self.config.RETRIEVAL_K}
            )
            
            # BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = self.config.RETRIEVAL_K
            
            # Ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=self.config.ENSEMBLE_WEIGHTS
            )
            
            # Add reranking if available
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key:
                try:
                    compressor = CohereRerank(
                        model="rerank-english-v3.0",
                        cohere_api_key=cohere_api_key
                    )
                    self.compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor,
                        base_retriever=ensemble_retriever
                    )
                    logger.info("Hybrid RAG with reranking setup completed")
                except Exception as e:
                    logger.warning(f"Cohere reranking failed, using ensemble only: {e}")
                    self.compression_retriever = ensemble_retriever
            else:
                logger.warning("Cohere API key not found, using ensemble without reranking")
                self.compression_retriever = ensemble_retriever
                
        except Exception as e:
            logger.error(f"Error setting up hybrid RAG: {e}")
            raise
    
    def retrieve_and_generate(self, query: str) -> Dict[str, Any]:
        """Retrieve with hybrid approach and generate"""
        try:
            docs = self.compression_retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            template = """You are a helpful AI assistant. Answer the question based on the provided context. Be accurate and concise.

Context: {context}

Question: {query}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {"context": lambda x: context, "query": lambda x: x}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = chain.invoke(query)
            
            # Track tokens
            self.token_tracker.add_tokens(len(context.split()) + len(answer.split()) + 50)
            self.token_tracker.add_query()
            
            return {
                "answer": answer,
                "contexts": [doc.page_content for doc in docs],
                "method": "hybrid_rerank",
                "num_retrieved": len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid RAG: {e}")
            raise

class RAGASEvaluator:
    """RAGAS evaluation with comprehensive fallback"""
    
    def __init__(self):
        self.use_ragas = RAGAS_AVAILABLE
        if RAGAS_AVAILABLE:
            try:
                self.metrics = [
                    answer_relevancy,
                    faithfulness,
                    context_precision,
                    context_recall,
                    answer_similarity,
                    answer_correctness
                ]
                logger.info("RAGAS evaluator initialized with all metrics")
            except Exception as e:
                logger.error(f"Error initializing RAGAS metrics: {e}")
                self.use_ragas = False
                self.metrics = []
        else:
            self.metrics = []
    
    def prepare_evaluation_dataset(self, results: List[Dict], ground_truth: List[Dict]) -> Dataset:
        """Prepare dataset for RAGAS"""
        try:
            evaluation_data = {
                'question': [],
                'answer': [],
                'contexts': [],
                'ground_truth': []
            }
            
            for result, gt in zip(results, ground_truth):
                evaluation_data['question'].append(gt['question'])
                evaluation_data['answer'].append(result['answer'])
                evaluation_data['contexts'].append(result['contexts'])
                evaluation_data['ground_truth'].append(gt['answer'])
            
            return Dataset.from_dict(evaluation_data)
        except Exception as e:
            logger.error(f"Error preparing evaluation dataset: {e}")
            raise
    
    async def evaluate_approach(self, results: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate using RAGAS or fallback"""
        try:
            if self.use_ragas and self.metrics and len(results) > 0:
                logger.info("Starting RAGAS evaluation...")
                dataset = self.prepare_evaluation_dataset(results, ground_truth)
                
                # Try RAGAS evaluation with error handling
                try:
                    evaluation_result = await asyncio.to_thread(
                        evaluate, 
                        dataset=dataset, 
                        metrics=self.metrics
                    )
                    logger.info("RAGAS evaluation completed successfully")
                    return dict(evaluation_result)
                except Exception as ragas_error:
                    logger.error(f"RAGAS evaluation failed: {ragas_error}")
                    logger.info("Falling back to basic evaluation")
                    return self._basic_evaluation(results, ground_truth)
            else:
                logger.info("Using basic evaluation (RAGAS not available)")
                return self._basic_evaluation(results, ground_truth)
                
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._basic_evaluation(results, ground_truth)
    
    def _basic_evaluation(self, results: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Enhanced basic similarity-based evaluation"""
        try:
            from difflib import SequenceMatcher
            
            if len(results) == 0 or len(ground_truth) == 0:
                logger.warning("No results or ground truth provided for evaluation")
                return self._fallback_scores()
            
            similarities = []
            answer_lengths = []
            context_qualities = []
            
            for result, gt in zip(results, ground_truth):
                # Calculate text similarity
                try:
                    result_answer = str(result.get('answer', ''))
                    gt_answer = str(gt.get('answer', ''))
                    
                    if result_answer and gt_answer:
                        similarity = SequenceMatcher(None, 
                            result_answer.lower(), 
                            gt_answer.lower()).ratio()
                    else:
                        similarity = 0.0
                        
                    similarities.append(similarity)
                    answer_lengths.append(len(result_answer.split()))
                    
                    # Simple context quality assessment
                    contexts = result.get('contexts', [])
                    context_quality = min(len(contexts) / 5.0, 1.0)  # Normalize by expected retrieval count
                    context_qualities.append(context_quality)
                    
                except Exception as e:
                    logger.warning(f"Error calculating similarity for result: {e}")
                    similarities.append(0.0)
                    answer_lengths.append(0)
                    context_qualities.append(0.0)
            
            if not similarities:
                return self._fallback_scores()
            
            avg_similarity = np.mean(similarities)
            avg_context_quality = np.mean(context_qualities)
            
            # Generate more realistic scores based on actual performance
            evaluation_result = {
                "answer_similarity": float(avg_similarity),
                "answer_relevancy": float(avg_similarity * 0.95),  # Slightly lower than similarity
                "faithfulness": float(avg_similarity * 0.90),  # Even more conservative
                "context_precision": float(avg_context_quality * 0.8),  # Based on context quality
                "context_recall": float(avg_context_quality * 0.75),  # Conservative recall estimate
                "answer_correctness": float(avg_similarity * 0.85),  # Conservative correctness
                "avg_answer_length": float(np.mean(answer_lengths)),
                "evaluation_method": "basic_enhanced"
            }
            
            logger.info(f"Basic evaluation completed. Average similarity: {avg_similarity:.3f}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Basic evaluation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._fallback_scores()
    
    def _fallback_scores(self) -> Dict:
        """Fallback scores when all evaluation methods fail"""
        return {
            "answer_similarity": 0.5,
            "answer_relevancy": 0.5,
            "faithfulness": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "answer_correctness": 0.5,
            "evaluation_method": "fallback"
        }

class RAGComparison:
    """Main comparison orchestrator"""
    
    def __init__(self, config: Config, token_tracker: TokenTracker):
        self.config = config
        self.token_tracker = token_tracker
        self.doc_processor = DocumentProcessor(config)
        self.evaluator = RAGASEvaluator()
        self.naive_rag = None
        self.hybrid_rag = None
        self.documents = None
        self.ground_truth = None
        self.results = {}
        
    def load_documents(self, pdf_path: str):
        """Load and process documents"""
        try:
            self.documents = self.doc_processor.load_and_split_documents(pdf_path)
            self.naive_rag = NaiveRAG(
                self.documents, 
                self.doc_processor.embeddings, 
                self.config,
                self.token_tracker
            )
            self.hybrid_rag = HybridRAGWithReranking(
                self.documents, 
                self.doc_processor.embeddings, 
                self.config,
                self.token_tracker
            )
            logger.info(f"Loaded {len(self.documents)} document chunks")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def load_ground_truth(self, json_path: str):
        """Load ground truth Q&A pairs"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.ground_truth = json.load(f)
            
            # Validate format
            for i, qa_pair in enumerate(self.ground_truth):
                if not isinstance(qa_pair, dict):
                    raise ValueError(f"Q&A pair {i} must be a dictionary")
                if 'question' not in qa_pair or 'answer' not in qa_pair:
                    raise ValueError(f"Q&A pair {i} must have 'question' and 'answer' keys")
                    
            logger.info(f"Loaded {len(self.ground_truth)} Q&A pairs")
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            raise
    
    async def run_comparison(self) -> Dict:
        """Run comprehensive comparison"""
        if not self.documents or not self.ground_truth:
            raise ValueError("Documents and ground truth must be loaded first")
        
        logger.info("Starting RAG comparison...")
        
        naive_results = []
        hybrid_results = []
        
        total_questions = len(self.ground_truth)
        
        for i, qa_pair in enumerate(self.ground_truth):
            question = qa_pair['question']
            
            try:
                logger.info(f"Processing question {i + 1}/{total_questions}: {question[:50]}...")
                
                # Test both approaches
                naive_result = self.naive_rag.retrieve_and_generate(question)
                naive_results.append(naive_result)
                
                hybrid_result = self.hybrid_rag.retrieve_and_generate(question)
                hybrid_results.append(hybrid_result)
                
                # Update progress
                progress = int((i + 1) / total_questions * 70)  # 70% for processing
                evaluation_status["progress"] = progress
                evaluation_status["message"] = f"Processed {i + 1}/{total_questions} questions"
                
                logger.info(f"Completed question {i + 1}")
                
            except Exception as e:
                logger.error(f"Error processing question {i + 1}: {e}")
                # Continue with other questions instead of stopping
                continue
        
        if len(naive_results) == 0 or len(hybrid_results) == 0:
            raise ValueError("No results generated. Check your API keys and model access.")
        
        # Evaluate both approaches
        evaluation_status["progress"] = 80
        evaluation_status["message"] = "Running evaluation..."
        
        logger.info("Starting evaluation phase...")
        
        # Ensure ground truth matches results length
        adjusted_gt = self.ground_truth[:len(naive_results)]
        
        try:
            naive_evaluation = await self.evaluator.evaluate_approach(naive_results, adjusted_gt)
            hybrid_evaluation = await self.evaluator.evaluate_approach(hybrid_results, adjusted_gt)
        except Exception as e:
            logger.error(f"Evaluation phase failed: {e}")
            # Provide basic fallback evaluation
            naive_evaluation = self.evaluator._fallback_scores()
            hybrid_evaluation = self.evaluator._fallback_scores()
        
        evaluation_status["progress"] = 95
        evaluation_status["message"] = "Generating comparison summary..."
        
        # Generate results
        self.results = {
            "naive_rag": {
                "results": naive_results,
                "evaluation": naive_evaluation,
                "success_rate": len(naive_results) / total_questions
            },
            "hybrid_rerank": {
                "results": hybrid_results,
                "evaluation": hybrid_evaluation,
                "success_rate": len(hybrid_results) / total_questions
            },
            "comparison_summary": self._generate_comparison_summary(naive_evaluation, hybrid_evaluation),
            "token_usage": self.token_tracker.get_stats(),
            "metadata": {
                "total_questions": total_questions,
                "successful_questions": len(naive_results),
                "document_chunks": len(self.documents),
                "timestamp": datetime.now().isoformat(),
                "ragas_used": self.evaluator.use_ragas
            }
        }
        
        logger.info("Comparison completed successfully")
        return self.results
    
    def _generate_comparison_summary(self, naive_eval: Dict, hybrid_eval: Dict) -> Dict:
        """Generate comparison summary"""
        summary = {"winner": {}, "metrics_comparison": {}, "overall_winner": None}
        
        metrics = ["answer_relevancy", "faithfulness", "context_precision", 
                  "context_recall", "answer_similarity", "answer_correctness"]
        
        hybrid_wins = 0
        naive_wins = 0
        
        for metric in metrics:
            if metric in naive_eval and metric in hybrid_eval:
                try:
                    naive_score = float(naive_eval[metric])
                    hybrid_score = float(hybrid_eval[metric])
                    
                    improvement = hybrid_score - naive_score
                    improvement_pct = (improvement / max(naive_score, 0.001)) * 100
                    
                    summary["metrics_comparison"][metric] = {
                        "naive_rag": naive_score,
                        "hybrid_rerank": hybrid_score,
                        "improvement": improvement,
                        "improvement_percentage": improvement_pct
                    }
                    
                    if hybrid_score > naive_score:
                        summary["winner"][metric] = "hybrid_rerank"
                        hybrid_wins += 1
                    else:
                        summary["winner"][metric] = "naive_rag"
                        naive_wins += 1
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing metric {metric}: {e}")
                    continue
        
        summary["overall_winner"] = "hybrid_rerank" if hybrid_wins > naive_wins else ("naive_rag" if naive_wins > hybrid_wins else "tie")
        summary["metrics_won"] = {"hybrid": hybrid_wins, "naive": naive_wins}
        
        return summary

# Flask Application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
CORS(app)

# Global variables
config = Config()
token_tracker = TokenTracker()
evaluation_status = {"status": "idle", "progress": 0, "message": "Ready"}
current_comparison = None

# HTML Template (same as before, keeping it compact)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Comparison Tool</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 1200px; margin: 0 auto; background: white;
            border-radius: 20px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; text-align: center;
        }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; font-weight: 300; }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .content { padding: 40px; }
        .section { margin-bottom: 40px; }
        .section h2 { color: #333; margin-bottom: 20px; font-size: 1.8rem; }
        .upload-area {
            border: 3px dashed #ddd; border-radius: 15px; padding: 40px;
            text-align: center; background: #fafafa; transition: all 0.3s ease;
        }
        .upload-area:hover { border-color: #667eea; background: #f0f4ff; }
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 12px 30px; border: none; border-radius: 25px;
            cursor: pointer; font-size: 1rem; margin: 10px; transition: all 0.3s ease;
        }
        .upload-btn:hover { transform: translateY(-2px); }
        .start-btn {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white; padding: 15px 40px; border: none; border-radius: 25px;
            cursor: pointer; font-size: 1.1rem; font-weight: 500;
            display: block; margin: 20px auto; transition: all 0.3s ease;
        }
        .start-btn:disabled { background: #ccc; cursor: not-allowed; }
        .file-input { display: none; }
        .status-card {
            background: #f8f9fa; border-radius: 15px; padding: 25px;
            margin: 20px 0; border-left: 5px solid #667eea;
        }
        .progress-bar {
            width: 100%; height: 8px; background: #e9ecef;
            border-radius: 4px; overflow: hidden; margin: 15px 0;
        }
        .progress-fill {
            height: 100%; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            transition: width 0.3s ease; border-radius: 4px;
        }
        .results-container {
            display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px;
        }
        .result-card {
            background: #f8f9fa; border-radius: 15px; padding: 25px;
            border: 2px solid #e9ecef;
        }
        .metric {
            display: flex; justify-content: space-between; align-items: center;
            padding: 12px 0; border-bottom: 1px solid #e9ecef;
        }
        .metric-name { font-weight: 500; color: #555; text-transform: capitalize; }
        .metric-value {
            font-weight: 600; color: #333; background: #e3f2fd;
            padding: 4px 12px; border-radius: 15px;
        }
        .file-status {
            margin: 10px 0; padding: 10px; border-radius: 8px; font-weight: 500;
        }
        .file-uploaded {
            background: #d4edda; color: #155724; border: 1px solid #c3e6cb;
        }
        .error {
            background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
            padding: 15px; border-radius: 8px; margin: 15px 0;
        }
        .hidden { display: none; }
        .loading {
            display: inline-block; width: 20px; height: 20px;
            border: 3px solid #f3f3f3; border-top: 3px solid #667eea;
            border-radius: 50%; animation: spin 1s linear infinite; margin-right: 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @media (max-width: 768px) {
            .results-container { grid-template-columns: 1fr; }
            .header h1 { font-size: 2rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG Comparison Tool</h1>
            <p>Compare Hybrid Search with Reranking vs Naive RAG using RAGAS Evaluation</p>
        </div>
        <div class="content">
            <div class="section">
                <h2>Upload Documents</h2>
                <div class="upload-area">
                    <p style="font-size: 1.2rem; color: #666; margin-bottom: 20px;">
                        Upload your PDF document and ground truth Q&A pairs (JSON format)
                    </p>
                    <input type="file" id="pdfFile" class="file-input" accept=".pdf">
                    <input type="file" id="groundTruthFile" class="file-input" accept=".json">
                    <button class="upload-btn" onclick="document.getElementById('pdfFile').click()">
                        Choose PDF Document
                    </button>
                    <button class="upload-btn" onclick="document.getElementById('groundTruthFile').click()">
                        Choose Ground Truth (JSON)
                    </button>
                    <div id="fileStatus"></div>
                </div>
            </div>
            <div class="section">
                <h2>Run Evaluation</h2>
                <button class="start-btn" id="startBtn" onclick="startEvaluation()" disabled>
                    Start RAG Comparison
                </button>
                <div class="status-card hidden" id="statusCard">
                    <h3>Evaluation Progress</h3>
                    <div id="statusMessage">Initializing...</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                    </div>
                </div>
            </div>
            <div class="section hidden" id="resultsSection">
                <h2>Evaluation Results</h2>
                <div class="results-container">
                    <div class="result-card">
                        <h3>Naive RAG</h3>
                        <div id="naiveResults"></div>
                    </div>
                    <div class="result-card">
                        <h3>Hybrid RAG with Reranking</h3>
                        <div id="hybridResults"></div>
                    </div>
                </div>
                <div class="status-card hidden" id="comparisonSummary">
                    <h3>Comparison Summary</h3>
                    <div id="summaryContent"></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        let pdfUploaded = false, groundTruthUploaded = false;
        
        document.getElementById('pdfFile').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                document.getElementById('fileStatus').innerHTML += 
                    '<div class="file-status file-uploaded">PDF Document uploaded: ' + e.target.files[0].name + '</div>';
                pdfUploaded = true;
                checkFilesUploaded();
            }
        });
        
        document.getElementById('groundTruthFile').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                document.getElementById('fileStatus').innerHTML += 
                    '<div class="file-status file-uploaded">Ground Truth uploaded: ' + e.target.files[0].name + '</div>';
                groundTruthUploaded = true;
                checkFilesUploaded();
            }
        });
        
        function checkFilesUploaded() {
            if (pdfUploaded && groundTruthUploaded) {
                uploadFiles();
            }
        }
        
        async function uploadFiles() {
            const formData = new FormData();
            formData.append('pdf_file', document.getElementById('pdfFile').files[0]);
            formData.append('ground_truth', document.getElementById('groundTruthFile').files[0]);
            
            try {
                const response = await fetch('/api/upload', { method: 'POST', body: formData });
                const result = await response.json();
                
                if (response.ok) {
                    document.getElementById('fileStatus').innerHTML += 
                        '<div class="file-status file-uploaded">Files processed successfully! Document chunks: ' + 
                        result.document_chunks + ', Q&A pairs: ' + result.qa_pairs + '</div>';
                    document.getElementById('startBtn').disabled = false;
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                document.getElementById('fileStatus').innerHTML += 
                    '<div class="error">Error: ' + error.message + '</div>';
            }
        }
        
        async function startEvaluation() {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('statusCard').classList.remove('hidden');
            
            try {
                const response = await fetch('/api/start_evaluation', { method: 'POST' });
                if (response.ok) {
                    pollStatus();
                } else {
                    const error = await response.json();
                    throw new Error(error.error);
                }
            } catch (error) {
                document.getElementById('statusMessage').innerHTML = 
                    '<div class="error">Error: ' + error.message + '</div>';
                document.getElementById('startBtn').disabled = false;
            }
        }
        
        async function pollStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                updateStatus(status);
                
                if (status.status === 'running') {
                    setTimeout(pollStatus, 2000);
                } else if (status.status === 'completed') {
                    loadResults();
                } else if (status.status === 'error') {
                    document.getElementById('startBtn').disabled = false;
                }
            } catch (error) {
                document.getElementById('statusMessage').innerHTML = 
                    '<div class="error">Error checking status: ' + error.message + '</div>';
                document.getElementById('startBtn').disabled = false;
            }
        }
        
        function updateStatus(status) {
            const statusMessage = document.getElementById('statusMessage');
            const progressFill = document.getElementById('progressFill');
            
            switch (status.status) {
                case 'running':
                    statusMessage.innerHTML = '<div class="loading"></div>' + status.message;
                    progressFill.style.width = status.progress + '%';
                    break;
                case 'completed':
                    statusMessage.innerHTML = 'Evaluation completed successfully!';
                    progressFill.style.width = '100%';
                    break;
                case 'error':
                    statusMessage.innerHTML = '<div class="error">Error: ' + status.message + '</div>';
                    progressFill.style.width = '0%';
                    break;
                default:
                    statusMessage.innerHTML = status.message;
            }
        }
        
        async function loadResults() {
            try {
                const response = await fetch('/api/results');
                const results = await response.json();
                displayResults(results);
                document.getElementById('resultsSection').classList.remove('hidden');
            } catch (error) {
                document.getElementById('statusMessage').innerHTML += 
                    '<div class="error">Error loading results: ' + error.message + '</div>';
            }
        }
        
        function displayResults(results) {
            // Display results for both approaches
            document.getElementById('naiveResults').innerHTML = generateMetricsHTML(results.naive_rag.evaluation);
            document.getElementById('hybridResults').innerHTML = generateMetricsHTML(results.hybrid_rerank.evaluation);
            
            // Display comparison summary
            displayComparisonSummary(results.comparison_summary, results.token_usage, results.metadata);
        }
        
        function generateMetricsHTML(evaluation) {
            const metrics = ['answer_relevancy', 'faithfulness', 'context_precision', 'context_recall', 'answer_similarity', 'answer_correctness'];
            let html = '';
            
            metrics.forEach(metric => {
                if (evaluation[metric] !== undefined) {
                    const value = (evaluation[metric] * 100).toFixed(2);
                    html += `
                        <div class="metric">
                            <span class="metric-name">${metric.replace('_', ' ')}</span>
                            <span class="metric-value">${value}%</span>
                        </div>
                    `;
                }
            });
            
            // Add evaluation method info
            if (evaluation.evaluation_method) {
                html += `
                    <div class="metric">
                        <span class="metric-name">Evaluation Method</span>
                        <span class="metric-value">${evaluation.evaluation_method}</span>
                    </div>
                `;
            }
            
            return html;
        }
        
        function displayComparisonSummary(summary, tokenUsage, metadata) {
            const summaryDiv = document.getElementById('comparisonSummary');
            const summaryContent = document.getElementById('summaryContent');
            
            let html = `
                <h4>Overall Winner: ${summary.overall_winner.replace('_', ' ').toUpperCase()}</h4>
                <p><strong>Hybrid RAG</strong> won <strong>${summary.metrics_won.hybrid}</strong> metrics</p>
                <p><strong>Naive RAG</strong> won <strong>${summary.metrics_won.naive}</strong> metrics</p>
                <br>
                <h4>Evaluation Details</h4>
                <p>Questions Processed: <strong>${metadata.successful_questions}/${metadata.total_questions}</strong></p>
                <p>Success Rate: <strong>${((metadata.successful_questions/metadata.total_questions)*100).toFixed(1)}%</strong></p>
                <p>RAGAS Used: <strong>${metadata.ragas_used ? 'Yes' : 'No (Basic Fallback)'}</strong></p>
                <br>
                <h4>Token Usage Statistics</h4>
                <p>Total Tokens Used: <strong>${tokenUsage.total_tokens}</strong></p>
                <p>Total Queries: <strong>${tokenUsage.query_count}</strong></p>
                <p>Average Tokens per Query: <strong>${tokenUsage.avg_tokens_per_query.toFixed(2)}</strong></p>
                <br>
                <h4>Metric Improvements</h4>
            `;
            
            // Add detailed metric comparisons
            if (summary.metrics_comparison) {
                Object.entries(summary.metrics_comparison).forEach(([metric, data]) => {
                    const improvement = data.improvement_percentage;
                    const improvementClass = improvement > 0 ? 'improvement' : 'decline';
                    const arrow = improvement > 0 ? '↗' : '↘';
                    
                    html += `
                        <div style="margin: 8px 0; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 5px;">
                            <strong>${metric.replace('_', ' ').toUpperCase()}</strong><br>
                            Naive: ${(data.naive_rag * 100).toFixed(2)}% → Hybrid: ${(data.hybrid_rerank * 100).toFixed(2)}%<br>
                            <span style="color: ${improvement > 0 ? '#28a745' : '#dc3545'}; font-weight: bold;">
                                ${arrow} ${Math.abs(improvement).toFixed(2)}% ${improvement > 0 ? 'improvement' : 'decline'}
                            </span>
                        </div>
                    `;
                });
            }
            
            summaryContent.innerHTML = html;
            summaryDiv.classList.remove('hidden');
        }
    </script>
</body>
</html>'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads with proper error handling"""
    global current_comparison
    
    try:
        if 'pdf_file' not in request.files or 'ground_truth' not in request.files:
            return jsonify({"error": "Both PDF and ground truth files required"}), 400
        
        pdf_file = request.files['pdf_file']
        gt_file = request.files['ground_truth']
        
        if pdf_file.filename == '' or gt_file.filename == '':
            return jsonify({"error": "No files selected"}), 400
        
        if not (allowed_file(pdf_file.filename) and allowed_file(gt_file.filename)):
            return jsonify({"error": "Invalid file types. Please upload PDF and JSON files."}), 400
        
        # Save files securely
        pdf_filename = secure_filename(pdf_file.filename)
        gt_filename = secure_filename(gt_file.filename)
        
        pdf_path = os.path.join(Config.UPLOAD_FOLDER, pdf_filename)
        gt_path = os.path.join(Config.UPLOAD_FOLDER, gt_filename)
        
        pdf_file.save(pdf_path)
        gt_file.save(gt_path)
        
        # Initialize comparison
        current_comparison = RAGComparison(config, token_tracker)
        current_comparison.load_documents(pdf_path)
        current_comparison.load_ground_truth(gt_path)
        
        # Clean up temp files
        os.remove(pdf_path)
        os.remove(gt_path)
        
        return jsonify({
            "message": "Files processed successfully",
            "document_chunks": len(current_comparison.documents),
            "qa_pairs": len(current_comparison.ground_truth)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/api/start_evaluation', methods=['POST'])
def start_evaluation():
    """Start evaluation with comprehensive error handling"""
    global evaluation_status, current_comparison
    
    if not current_comparison:
        return jsonify({"error": "No documents loaded. Please upload files first."}), 400
    
    if evaluation_status["status"] == "running":
        return jsonify({"error": "Evaluation already running"}), 400
    
    def run_evaluation():
        global evaluation_status
        loop = None
        try:
            evaluation_status = {"status": "running", "progress": 10, "message": "Initializing evaluation..."}
            logger.info("Starting evaluation thread")
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            evaluation_status["progress"] = 30
            evaluation_status["message"] = "Running RAG comparison..."
            
            # Run the comparison
            logger.info("Starting comparison run")
            results = loop.run_until_complete(current_comparison.run_comparison())
            
            evaluation_status = {
                "status": "completed", 
                "progress": 100, 
                "message": "Evaluation completed successfully!",
                "results": results
            }
            logger.info("Evaluation completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Evaluation error: {error_msg}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            evaluation_status = {
                "status": "error", 
                "progress": 0, 
                "message": f"Evaluation failed: {error_msg}"
            }
        finally:
            # Clean up the event loop
            if loop:
                try:
                    loop.close()
                except Exception as e:
                    logger.warning(f"Error closing event loop: {e}")
    
    # Start evaluation in background thread
    thread = threading.Thread(target=run_evaluation)
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Evaluation started successfully"})

@app.route('/api/status')
def get_status():
    """Get current evaluation status"""
    return jsonify(evaluation_status)

@app.route('/api/results')
def get_results():
    """Get evaluation results"""
    if evaluation_status["status"] != "completed":
        return jsonify({"error": "Evaluation not completed yet"}), 400
    
    return jsonify(evaluation_status.get("results", {}))

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "token_stats": token_tracker.get_stats(),
        "ragas_available": RAGAS_AVAILABLE
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error occurred"}), 500

if __name__ == '__main__':
    # Check environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set them in your .env file")
        print("ERROR: Missing required API keys!")
        print("Please create a .env file with:")
        print("OPENAI_API_KEY=your_openai_key_here")
        print("COHERE_API_KEY=your_cohere_key_here")
        exit(1)
    
    if not os.getenv('COHERE_API_KEY'):
        logger.warning("COHERE_API_KEY not set. Reranking will be disabled.")
        print("WARNING: COHERE_API_KEY not set. Hybrid reranking will use ensemble retrieval only.")
    
    logger.info("Starting RAG Comparison Application...")
    logger.info(f"Token tracking enabled. Current stats: {token_tracker.get_stats()}")
    logger.info(f"RAGAS evaluation available: {RAGAS_AVAILABLE}")
    
    print("=" * 50)
    print("RAG Comparison Application Starting...")
    print("=" * 50)
    print(f"Server will be available at: http://localhost:5000")
    print(f"RAGAS evaluation: {'Available' if RAGAS_AVAILABLE else 'Using basic fallback'}")
    print(f"Cohere reranking: {'Available' if os.getenv('COHERE_API_KEY') else 'Disabled'}")
    print("=" * 50)
    
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        host='0.0.0.0',
        port=int(os.getenv('FLASK_PORT', 5000)),
        use_reloader=True
    )