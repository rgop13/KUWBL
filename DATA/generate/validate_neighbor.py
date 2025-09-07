#!/usr/bin/env python3
"""
Ultra-optimized validator with failed task retry mechanism
Designed for 48 GPU DeepSeek-V3.1 setup with HTTP 500 error handling
"""
import os
import sys
import json
import jsonlines
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from tqdm.asyncio import tqdm as tqdm_asyncio
from tqdm import tqdm
from collections import defaultdict
import argparse
import time
import random

from openai import AsyncOpenAI
from httpx import Limits, Timeout, AsyncClient

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class RetryValidator:
    """Validator with failed task collection and retry mechanism"""
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:8000/v1",
                 model_name: str = "Qwen/Qwen3-235B-A22B-Thinking-2507",
                 initial_concurrent: int = 50,
                 max_concurrent: int = 256,
                 timeout_s: float = 300.0):
        
        self.base_url = base_url
        self.model_name = model_name
        self.current_concurrent = initial_concurrent
        self.max_concurrent = max_concurrent
        self.timeout_s = timeout_s
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.http_500_errors = 0
        self.start_time = None
        
        # Failed task tracking for retry
        self.failed_tasks = []  # List of (task_id, query, passage, error_type)
        self.retry_results = {}  # task_id -> result
        
        # Adaptive concurrency control
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        
        self._setup_client()
        
    def _setup_client(self):
        """Setup HTTP client with current concurrency settings"""
        http_client = AsyncClient(
            limits=Limits(
                max_connections=self.current_concurrent * 3,
                max_keepalive_connections=self.current_concurrent
            ),
            timeout=Timeout(self.timeout_s)
        )
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
            timeout=Timeout(self.timeout_s),
            max_retries=5,
            http_client=http_client,
        )
        self.semaphore = asyncio.Semaphore(self.current_concurrent)
        
        logger.info(f"🔧 Client configured with {self.current_concurrent} concurrent connections")
    
    def adapt_concurrency(self, success: bool):
        """Adapt concurrency based on success/failure patterns"""
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # Increase concurrency if we have many consecutive successes
            if (self.consecutive_successes >= 20 and 
                self.current_concurrent < self.max_concurrent):
                old_concurrent = self.current_concurrent
                self.current_concurrent = min(self.max_concurrent, int(self.current_concurrent * 1.2))
                if self.current_concurrent != old_concurrent:
                    logger.info(f"📈 Increasing concurrency: {old_concurrent} → {self.current_concurrent}")
                    self._setup_client()
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Decrease concurrency if we have failures
            if self.consecutive_failures >= 5:
                old_concurrent = self.current_concurrent
                self.current_concurrent = max(10, int(self.current_concurrent * 0.7))
                if self.current_concurrent != old_concurrent:
                    logger.warning(f"📉 Decreasing concurrency due to failures: {old_concurrent} → {self.current_concurrent}")
                    self._setup_client()
                    self.consecutive_failures = 0
    
    def get_system_prompt(self) -> str:
        """Get comprehensive system prompt for validation task"""
        return """You are an expert document relevance evaluator. Your task is to determine whether a given passage contains sufficient information to answer a specific question.

EVALUATION CRITERIA:
1. Read the question carefully and identify what specific information is needed to answer it
2. Examine the passage thoroughly to check if it contains the required information
3. The passage must contain DIRECT and SUFFICIENT information to answer the question
4. Partial information or tangentially related content is considered insufficient
5. The answer must be derivable from the passage alone, without external knowledge

OUTPUT REQUIREMENTS:
- You must respond with EXACTLY ONE WORD: either "answerable" or "unanswerable"
- "answerable": The passage contains sufficient direct information to answer the question
- "unanswerable": The passage lacks sufficient information to answer the question
- Do not provide explanations, reasoning, or additional text in your final response
- Your final answer must be the last word after any reasoning (after </think> if present)

EXAMPLES:
Question: "What team does Kim Ji-woon play for in 2021?"
Passage: "Kim Ji-woon joined Daejeon Korea Railroad FC in the 2021 season."
Answer: answerable

Question: "What is the capital of France?"
Passage: "Paris is known for its beautiful architecture and rich history."
Answer: unanswerable (doesn't explicitly state Paris is the capital)"""

    def get_user_prompt(self, query: str, passage: str) -> str:
        """Get specific user prompt for this validation task"""
        return f"""Please evaluate the following question-passage pair:

QUESTION: {query}

PASSAGE: {passage}

TASK: Determine if the passage contains sufficient information to directly answer the question.

RESPONSE: Provide only one word - "answerable" or "unanswerable"."""

    async def validate_single_with_tracking(self, query: str, passage: str, task_id: str) -> Tuple[str, bool]:
        """Single validation with failure tracking"""
        async with self.semaphore:
            try:
                system_prompt = self.get_system_prompt()
                user_prompt = self.get_user_prompt(query, passage)
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"chat_template_kwargs": {"thinking": True}}
                )
                
                raw_result = response.choices[0].message.content.strip()
                
                # Extract result after </think>
                if "</think>" in raw_result:
                    result = raw_result.split("</think>")[-1].strip().lower()
                else:
                    result = raw_result.strip().lower()
                
                # Quick matching
                if "unanswerable" in result:
                    self.adapt_concurrency(True)
                    self.successful_requests += 1
                    return "unanswerable", True
                elif "answerable" in result:
                    self.adapt_concurrency(True)
                    self.successful_requests += 1
                    return "answerable", True
                else:
                    self.adapt_concurrency(False)
                    self.failed_requests += 1
                    self.failed_tasks.append((task_id, query, passage, "INVALID_RESPONSE"))
                    return "unanswerable", False
                    
            except Exception as e:
                self.failed_requests += 1
                error_str = str(e)
                
                if "500" in error_str:
                    self.http_500_errors += 1
                    self.failed_tasks.append((task_id, query, passage, "HTTP_500"))
                    self.adapt_concurrency(False)
                else:
                    self.failed_tasks.append((task_id, query, passage, f"ERROR_{type(e).__name__}"))
                    self.adapt_concurrency(False)
                
                return "unanswerable", False
    
    async def validate_with_progress_and_retry(self, 
                                           validation_tasks: List[Tuple[str, str, str, str]], 
                                           output_dir: str,
                                           all_queries: Dict[str, str],
                                           global_query_metadata: Dict[str, Dict]) -> List[Tuple[str, str, str, str]]:
        """Main validation with progress tracking and failed task retry"""
        if not validation_tasks:
            return []
        
        self.start_time = time.time()
        self.total_requests = len(validation_tasks)
        
        logger.info(f"🚀 Starting MAIN PROCESS: {len(validation_tasks):,} tasks")
        logger.info(f"🔥 Initial concurrency: {self.current_concurrent}")
        
        # PHASE 1: MAIN PROCESSING
        logger.info("=" * 60)
        logger.info("🔥 PHASE 1: MAIN BATCH PROCESSING")
        logger.info("=" * 60)
        
        # Create all tasks with unique IDs for tracking
        async_tasks = []
        task_metadata = []
        
        for i, (query_id, query, doc_id, passage) in enumerate(validation_tasks):
            task_id = f"{query_id}_{doc_id}_{i}"
            task = self.validate_single_with_tracking(query, passage, task_id)
            async_tasks.append(task)
            task_metadata.append((query_id, query, doc_id, task_id))
        
        # Process with real-time progress
        main_results = []
        
        async for task in tqdm_asyncio(asyncio.as_completed(async_tasks), 
                                      total=len(async_tasks),
                                      desc="🔥 Main processing"):
            try:
                validation_result, success = await task
                main_results.append((validation_result, success))
                
                # Progress reporting and intermediate saving every 5000 tasks
                if len(main_results) % 5000 == 0:
                    elapsed = time.time() - self.start_time
                    rate = len(main_results) / elapsed
                    eta = (len(async_tasks) - len(main_results)) / rate / 60 if rate > 0 else 0
                    success_rate = self.successful_requests / (self.successful_requests + self.failed_requests) * 100 if self.failed_requests > 0 else 100
                    
                    logger.info(f"⚡ MAIN: {len(main_results):,}/{len(async_tasks):,} "
                               f"({rate:.0f}/sec, ETA: {eta:.1f}min, Success: {success_rate:.1f}%, "
                               f"Concurrency: {self.current_concurrent}, HTTP500: {self.http_500_errors})")
                    
                    # Save intermediate progress every 10,000 tasks
                    if len(main_results) % 10000 == 0:
                        # Create partial results for saving
                        partial_results = []
                        for i, (query_id, query, doc_id, task_id) in enumerate(task_metadata[:len(main_results)]):
                            if i < len(main_results):
                                validation_result, success = main_results[i]
                                partial_results.append((query_id, query, doc_id, validation_result))
                        
                        save_intermediate_results(output_dir, partial_results, f"phase1_progress_{len(main_results)}")
                        logger.info(f"💾 Intermediate progress saved at {len(main_results):,} tasks")
                
            except Exception as e:
                main_results.append(("unanswerable", False))
                self.failed_requests += 1
        
        # Combine main results with metadata
        combined_results = []
        for i, (query_id, query, doc_id, task_id) in enumerate(task_metadata):
            if i < len(main_results):
                validation_result, success = main_results[i]
                combined_results.append((query_id, query, doc_id, validation_result))
            else:
                combined_results.append((query_id, query, doc_id, "unanswerable"))
        
        # SAVE INTERMEDIATE RESULTS AFTER PHASE 1
        save_intermediate_results(output_dir, combined_results, "phase1_main")
        
        # SAVE CHECKPOINT AFTER PHASE 1
        save_checkpoint(output_dir, all_queries, global_query_metadata, combined_results, "phase1_main")
        
        # PHASE 2: RETRY FAILED TASKS
        if self.failed_tasks:
            logger.info("=" * 60)
            logger.info(f"🔄 PHASE 2: RETRYING {len(self.failed_tasks)} FAILED TASKS")
            logger.info("=" * 60)
            
            # Reset concurrency for retry (more conservative)
            self.current_concurrent = max(10, self.current_concurrent // 4)
            self._setup_client()
            logger.info(f"🔧 Retry concurrency: {self.current_concurrent}")
            
            # Retry failed tasks
            retry_tasks = []
            retry_metadata = []
            
            for task_id, query, passage, error_type in self.failed_tasks:
                retry_task = self.validate_single_with_tracking(query, passage, f"retry_{task_id}")
                retry_tasks.append(retry_task)
                retry_metadata.append((task_id, query, passage))
            
            retry_results = []
            async for task in tqdm_asyncio(asyncio.as_completed(retry_tasks), 
                                          total=len(retry_tasks),
                                          desc="🔄 Retrying failed"):
                try:
                    validation_result, success = await task
                    retry_results.append(validation_result)
                except Exception:
                    retry_results.append("unanswerable")
            
            # Update results with retry outcomes
            retry_lookup = {}
            for i, (task_id, query, passage) in enumerate(retry_metadata):
                if i < len(retry_results):
                    retry_lookup[task_id] = retry_results[i]
            
            # Update combined results with retry results
            for i, (query_id, query, doc_id, result) in enumerate(combined_results):
                task_id = f"{query_id}_{doc_id}_{i}"
                if task_id in retry_lookup:
                    combined_results[i] = (query_id, query, doc_id, retry_lookup[task_id])
            
            logger.info(f"✅ Retry completed: {len(retry_results)} tasks processed")
            
            # SAVE INTERMEDIATE RESULTS AFTER PHASE 2
            save_intermediate_results(output_dir, combined_results, "phase2_retry")
            
            # SAVE FINAL CHECKPOINT
            save_checkpoint(output_dir, all_queries, global_query_metadata, combined_results, "phase2_retry")
        else:
            logger.info("✅ No failed tasks - skipping Phase 2")
            
            # SAVE FINAL RESULTS (no retry needed)
            save_intermediate_results(output_dir, combined_results, "final_no_retry")
            save_checkpoint(output_dir, all_queries, global_query_metadata, combined_results, "final_no_retry")
        
        # Final performance report
        elapsed = time.time() - self.start_time
        final_rate = len(combined_results) / elapsed
        
        logger.info("🎉 VALIDATION WITH RETRY COMPLETE!")
        logger.info(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        logger.info(f"🚀 Average rate: {final_rate:.1f} validations/second")
        logger.info(f"✅ Success rate: {self.successful_requests/self.total_requests*100:.1f}%")
        logger.info(f"❌ HTTP 500 errors: {self.http_500_errors}")
        logger.info(f"🔄 Failed tasks retried: {len(self.failed_tasks)}")
        logger.info(f"🔧 Final concurrency: {self.current_concurrent}")
        
        return combined_results

# Import utility functions
def load_corpus(corpus_path: str) -> Dict[str, str]:
    """Load corpus efficiently"""
    logger.info(f"Loading corpus from {corpus_path}")
    corpus_dict = {}
    
    with jsonlines.open(corpus_path) as reader:
        for line in tqdm(reader, desc="Loading corpus"):
            corpus_dict[line["_id"]] = line["text"]
    
    logger.info(f"Loaded {len(corpus_dict):,} documents")
    return corpus_dict

def load_sampled_queries(data_path: str) -> List[Dict[str, Any]]:
    """Load sampled queries efficiently"""
    data = []
    with jsonlines.open(data_path) as reader:
        for line in reader:
            data.append(line)
    return data

def generate_query_id(file_name: str, row_idx: int) -> str:
    """Generate unique query ID"""
    key = file_name.replace("sampled_docs_gen_queries_", "").replace(".jsonl", "")
    return f"{key}_{row_idx:06d}"

def save_intermediate_results(output_dir: str, validation_results: List[Tuple[str, str, str, str]], phase: str):
    """Save intermediate validation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    intermediate_file = os.path.join(output_dir, f"intermediate_validation_results_{phase}.jsonl")
    logger.info(f"💾 Saving {len(validation_results):,} intermediate results to {intermediate_file}")
    
    with jsonlines.open(intermediate_file, 'w') as writer:
        for query_id, query_text, doc_id, validation_result in validation_results:
            writer.write({
                "query_id": query_id,
                "query_text": query_text,
                "doc_id": doc_id,
                "validation_result": validation_result,
                "phase": phase
            })
    
    logger.info(f"✅ Intermediate results saved: {intermediate_file}")

def load_intermediate_results(output_dir: str, phase: str) -> List[Tuple[str, str, str, str]]:
    """Load intermediate validation results if they exist"""
    intermediate_file = os.path.join(output_dir, f"intermediate_validation_results_{phase}.jsonl")
    
    if not os.path.exists(intermediate_file):
        return []
    
    logger.info(f"📂 Loading intermediate results from {intermediate_file}")
    results = []
    
    with jsonlines.open(intermediate_file) as reader:
        for line in reader:
            results.append((
                line["query_id"],
                line["query_text"], 
                line["doc_id"],
                line["validation_result"]
            ))
    
    logger.info(f"✅ Loaded {len(results):,} intermediate results")
    return results

def save_checkpoint(output_dir: str, 
                   queries: Dict[str, str], 
                   query_metadata: Dict[str, Dict],
                   validation_results: List[Tuple[str, str, str, str]],
                   phase: str):
    """Save complete checkpoint for resuming"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{phase}.json")
    logger.info(f"💾 Saving checkpoint to {checkpoint_file}")
    
    checkpoint_data = {
        "queries": queries,
        "query_metadata": query_metadata,
        "validation_results": validation_results,
        "phase": phase,
        "timestamp": time.time(),
        "total_validations": len(validation_results)
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Checkpoint saved: {checkpoint_file}")

def save_beir_format(output_dir: str, queries: Dict[str, str], corpus: Dict[str, str], qrels: Dict[str, Dict[str, int]]):
    """Save BEIR format efficiently"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter corpus to only referenced documents
    referenced_doc_ids = set()
    for doc_scores in qrels.values():
        referenced_doc_ids.update(doc_scores.keys())
    
    filtered_corpus = {doc_id: corpus[doc_id] for doc_id in referenced_doc_ids if doc_id in corpus}
    
    logger.info(f"Corpus filtered: {len(corpus):,} → {len(filtered_corpus):,} documents")
    
    # Save all files efficiently
    with jsonlines.open(os.path.join(output_dir, "queries.jsonl"), 'w') as writer:
        for query_id, query_text in queries.items():
            writer.write({"_id": query_id, "text": query_text})
    
    with jsonlines.open(os.path.join(output_dir, "corpus.jsonl"), 'w') as writer:
        for doc_id, doc_text in filtered_corpus.items():
            writer.write({"_id": doc_id, "text": doc_text})
    
    with open(os.path.join(output_dir, "qrels.tsv"), 'w') as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, doc_scores in qrels.items():
            for doc_id, score in doc_scores.items():
                f.write(f"{query_id}\t{doc_id}\t{score}\n")
    
    with jsonlines.open(os.path.join(output_dir, "qrels.jsonl"), 'w') as writer:
        for query_id, doc_scores in qrels.items():
            writer.write({"query_id": query_id, "doc_scores": doc_scores})
    
    logger.info(f"BEIR format saved to {output_dir}")

async def main():
    parser = argparse.ArgumentParser(description="Retry-enabled ultra-fast validator")
    parser.add_argument("--corpus_path", type=str, 
                       default="/data/data_team/test/test_data")
    parser.add_argument("--output_dir", type=str, default="/data/data_team/test/test_data/validated")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-235B-A22B-Thinking-2507")
    parser.add_argument("--initial_concurrent", type=int, default=50)
    parser.add_argument("--max_concurrent", type=int, default=300)
    parser.add_argument("--max_queries_per_file", type=int, default=-1)
    parser.add_argument("--timeout_s", type=float, default=300.0)   
    
    args = parser.parse_args()
    
    # Initialize retry validator
    validator = RetryValidator(
        base_url=args.base_url,
        model_name=args.model_name,
        initial_concurrent=args.initial_concurrent,
        max_concurrent=args.max_concurrent,
        timeout_s=args.timeout_s
    )
    
    # Load data
    corpus_path = os.path.join(args.corpus_path, "corpus.jsonl")
    corpus_dict = load_corpus(corpus_path)
    
    target_files = [
        "sampled_docs_gen_queries_dense_popular.jsonl",
        "sampled_docs_gen_queries_dense_mid.jsonl",
        "sampled_docs_gen_queries_dense_rare.jsonl",
        "sampled_docs_gen_queries_sparse_popular.jsonl",
        "sampled_docs_gen_queries_sparse_mid.jsonl",
        "sampled_docs_gen_queries_sparse_rare.jsonl",
    ]
    
    # Load all data into memory
    all_queries = {}
    all_validation_tasks = []
    global_query_metadata = {}
    
    logger.info("📚 Loading all query files into memory...")
    
    for file_name in target_files:
        file_path = os.path.join(args.corpus_path, file_name)
        if not os.path.exists(file_path):
            continue
            
        sampled_data = load_sampled_queries(file_path)
        if args.max_queries_per_file > 0:
            sampled_data = sampled_data[:args.max_queries_per_file]
        
        for row_idx, row in enumerate(sampled_data):
            query_id = generate_query_id(file_name, row_idx)
            query_text = row["query"]
            gold_doc_id = row["doc_id"]
            
            all_queries[query_id] = query_text
            global_query_metadata[query_id] = {
                "gold_doc_id": gold_doc_id,
                "top_k_doc_ids": row.get("top_k_doc_ids", []),
                "top_k_scores": row.get("top_k_scores", [])
            }
            
            # Add ALL validation tasks
            if "top_k_doc_ids" in row:
                for doc_id in row["top_k_doc_ids"]:
                    if doc_id in corpus_dict:
                        passage = corpus_dict[doc_id]
                        all_validation_tasks.append((query_id, query_text, doc_id, passage))
        
        logger.info(f"✅ {file_name}: {len(sampled_data)} queries")
    
    logger.info(f"🎯 LOADED: {len(all_queries):,} queries, {len(all_validation_tasks):,} validations")
    
    # MAIN PROCESSING WITH RETRY
    validation_results = await validator.validate_with_progress_and_retry(
        all_validation_tasks, args.output_dir, all_queries, global_query_metadata)
    
    # Process results
    query_answerable_docs = defaultdict(list)
    for query_id, query_text, doc_id, validation_result in validation_results:
        if validation_result == "answerable":
            query_answerable_docs[query_id].append(doc_id)
    
    # Create qrels
    all_qrels = {}
    for query_id in global_query_metadata:
        answerable_docs = query_answerable_docs[query_id]
        if answerable_docs:
            all_qrels[query_id] = {doc_id: 1 for doc_id in answerable_docs}
    
    # Save results
    save_beir_format(args.output_dir, all_queries, corpus_dict, all_qrels)
    
    # Final comprehensive statistics
    total_qrels = sum(len(doc_scores) for doc_scores in all_qrels.values())
    queries_with_relevant_docs = len(all_qrels)
    queries_without_relevant_docs = len(all_queries) - queries_with_relevant_docs
    
    logger.info("🏆 FINAL COMPREHENSIVE RESULTS:")
    logger.info(f"📊 Total queries: {len(all_queries):,}")
    logger.info(f"📊 Total validations: {len(validation_results):,}")
    logger.info(f"📊 Answerable queries: {queries_with_relevant_docs:,}")
    logger.info(f"📊 Unanswerable queries: {queries_without_relevant_docs:,}")
    logger.info(f"📊 Total relevant pairs: {total_qrels:,}")
    logger.info(f"📊 Success rate: {validator.successful_requests/validator.total_requests*100:.1f}%")
    logger.info(f"📊 HTTP 500 errors: {validator.http_500_errors}")
    logger.info(f"📊 Failed tasks retried: {len(validator.failed_tasks)}")
    logger.info(f"📊 Output saved to: {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
