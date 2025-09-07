import os, sys, json, jsonlines, random, logging
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def load_data(data_path: str) -> Tuple[List[Dict[str, Any]]]:
    data = []
    with jsonlines.open(data_path, 'r') as reader:
        data = [line for line in tqdm(reader, desc="Loading data")] 
    return data
