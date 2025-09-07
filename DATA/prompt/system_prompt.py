import os, sys, json, jsonlines, random, argparse, logging, time
from utils import load_data
from typing import List, Dict, Any

# Configure pretty logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_system_prompt(domain: str) -> Dict[str, List[str]]:
    system_prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt", f"{domain}.json")
    lang_prompts = {}
    with open(system_prompt_path, 'r') as f:
        system_prompt = json.load(f)
    for lang, prompt in system_prompt.items():
        lang_prompts[lang] = prompt
    return lang_prompts

def random_insert_system_prompt(data: List[Dict[str, Any]], lang_system_prompts: Dict[str, List[str]], ratio: float = 0.2):
    # 1) random sample (ratio)
    # 2) random choose system prompt
    # 3) insert system prompt with format of dictionary as follows:
    #    {
    #        "from": "system",
    #        "content": system_prompt
    #    }
    # 4) return the data with system prompt inserted
    
    logger.info("🚀 Starting system prompt insertion")
    logger.info(f"   📊 Insertion ratio: {ratio:.1%}")
    
    # Create a copy of the data to avoid modifying the original
    modified_data = []
    
    # 1) Random sample based on ratio
    sample_size = int(len(data) * ratio)
    selected_indices = random.sample(range(len(data)), sample_size)
    selected_indices_set = set(selected_indices)
    
    logger.info(f"   🎯 Selected {sample_size:,} out of {len(data):,} samples for insertion")
    
    # Process all data
    for i, sample in enumerate(data):
        # Create a deep copy of the sample to avoid modifying original
        modified_sample = sample.copy()
        modified_sample["query_and_response"] = sample["query_and_response"].copy()
        
        if i in selected_indices_set:
            # 2) Random choose system prompt based on language
            lang = sample.get("lang", "en")  # Default to "en" if lang field is missing
            
            # Use "other" as fallback if the specific language is not available
            if lang not in lang_system_prompts:
                if "other" in lang_system_prompts:
                    lang = "other"
                elif "en" in lang_system_prompts:
                    lang = "en"
                else:
                    # If neither "other" nor "en" exist, use the first available language
                    lang = list(lang_system_prompts.keys())[0]
                    
            logger.debug(f"Sample {i}: Using language '{lang}' for system prompt")
            chosen_prompt = random.choice(lang_system_prompts[lang])
            
            # 3) Insert system prompt with the specified format
            system_message = {
                "from": "system",
                "content": chosen_prompt
            }
            
            # Insert system prompt at the beginning of the conversation
            modified_sample["query_and_response"] = [system_message] + modified_sample["query_and_response"]
        
        modified_data.append(modified_sample)
    
    # Log language distribution statistics
    lang_stats = {}
    system_prompt_count = 0
    for sample in modified_data:
        if sample["query_and_response"][0].get("from") == "system":
            system_prompt_count += 1
            lang = sample.get("lang", "unknown")
            lang_stats[lang] = lang_stats.get(lang, 0) + 1
    
    logger.info("✅ System prompt insertion completed")
    
    # Pretty print language distribution
    if lang_stats:
        logger.info("   📈 Language distribution of samples with system prompts:")
        for lang, count in sorted(lang_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / system_prompt_count) * 100
            logger.info(f"      {lang:>6}: {count:>4,} samples ({percentage:>5.1f}%)")
    else:
        logger.info("   📈 No system prompts were inserted")
    
    # 4) Return the data with system prompt inserted
    return modified_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, 
        default="/data_x/WBL/data/250905/clarity_tagged/safety_merged_22435_9836_11176.jsonl",
        help="Path to the data file"
    )
    parser.add_argument(
        "--ratio", type=float, 
        default=1.0,
        help="Ratio of data to insert system prompts"
    )
    parser.add_argument(
        "--output_path", type=str,
        default="/data_x/WBL/data/250905/generation_candidates",
        help="Path to save the modified data"
    )
    args = parser.parse_args()
    
    # Load data and system prompts
    logger.info("=" * 60)
    logger.info("🔧 SYSTEM PROMPT INSERTION TOOL")
    logger.info("=" * 60)
    
    data = load_data(args.data_path)
    domain = data[0]["도메인_대분류"]
    system_prompt = load_system_prompt(domain)
    
    logger.info(f"📁 Data file: {os.path.basename(args.data_path)}")
    logger.info(f"📊 Loaded {len(data):,} samples from domain: {domain}")
    logger.info(f"🌐 Available system prompts for languages: {', '.join(system_prompt.keys())}")
    
    # Log language distribution in the original data
    original_lang_stats = {}
    for sample in data:
        lang = sample.get("lang", "unknown")
        original_lang_stats[lang] = original_lang_stats.get(lang, 0) + 1
    
    logger.info("📈 Original data language distribution:")
    for lang, count in sorted(original_lang_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        logger.info(f"   {lang:>6}: {count:>4,} samples ({percentage:>5.1f}%)")
    
    # Apply random system prompt insertion
    modified_data = random_insert_system_prompt(data, system_prompt, args.ratio)
    
    # Count how many samples got system prompts
    samples_with_system = sum(1 for sample in modified_data if sample["query_and_response"][0].get("from") == "system")
    success_rate = (samples_with_system / len(modified_data)) * 100
    
    logger.info("-" * 60)
    logger.info(f"📋 SUMMARY:")
    logger.info(f"   ✨ Successfully inserted system prompts: {samples_with_system:,}/{len(modified_data):,} ({success_rate:.1f}%)")
    
    # Save if output path is provided
    if args.output_path:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
        file_name = os.path.basename(args.data_path).replace(".jsonl", f"_system_prompt_{args.ratio}.jsonl")
        output_file = os.path.join(args.output_path, file_name)
        
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(modified_data)
        
        # Calculate file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        logger.info(f"💾 Saved modified data:")
        logger.info(f"   📁 File: {file_name}")
        logger.info(f"   📂 Path: {args.output_path}")
        logger.info(f"   📏 Size: {file_size:.1f} MB")
        
    logger.info("=" * 60)
    logger.info("🎉 PROCESS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


