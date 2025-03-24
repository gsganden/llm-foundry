import json
from pathlib import Path
import numpy as np

from streaming import MDSWriter
from tqdm import tqdm
from transformers import AutoTokenizer

def convert_tinystories_to_mds(
    input_path: Path,
    out_root: Path,
    tokenizer_name: str = "EleutherAI/gpt-neox-20b",
    concat_tokens: int = 2048,
    compression: str = "zstd",
) -> None:
    """Convert TinyStories dataset to MDS format.

    Args:
        input_path: Path to the input JSON file
        out_root: Output directory for MDS files
        tokenizer_name: Name of the tokenizer to use
        concat_tokens: Number of tokens to concatenate
        compression: Compression type for MDS files
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = int(1e30)  # Suppress warnings about sequence length

    # Create output directory
    out_root.mkdir(parents=True, exist_ok=True)

    # Read input file
    with input_path.open('r') as f:
        data = json.load(f)

    # Prepare MDS writer
    columns = {'tokens': 'ndarray:int32'}
    with MDSWriter(columns=columns, out=str(out_root), compression=compression, exist_ok=True) as out:
        # Process each story
        for item in tqdm(data, desc="Converting stories"):
            # Combine story fields into a single text
            text = f"Story: {item['story']}\nInstruction: {item['instruction']}\nSummary: {item['summary']}\nSource: {item['source']}"

            # Tokenize text
            tokens = tokenizer(text, truncation=True, max_length=concat_tokens)['input_ids']

            # Convert tokens to numpy array
            tokens = np.array(tokens, dtype=np.int32)

            # Write to MDS
            out.write({'tokens': tokens})

if __name__ == '__main__':
    # Get the script's directory
    script_dir = Path(__file__).parent

    # Example usage
    convert_tinystories_to_mds(
        input_path=script_dir / 'tinystories' / 'data00.json',
        out_root=script_dir / 'tinystories-mds',
        tokenizer_name="EleutherAI/gpt-neox-20b",
        concat_tokens=2048,
        compression="zstd"
    )
