import json
import os
from streaming import MDSWriter
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import numpy as np
import glob

def convert_tinystories_to_mds(
    input_path: str,
    out_root: str,
    split: str = "train",
    tokenizer_name: str = "EleutherAI/gpt-neox-20b",
    concat_tokens: int = 2048,
    compression: str = None,  # Changed default to None to match C4
    bos_text: str = "",
    eos_text: str = "<|endoftext|>",
) -> None:
    """Convert TinyStories dataset to MDS format.

    Args:
        input_path: Path to the input JSON file
        out_root: Output directory for MDS files
        split: Dataset split (train/val)
        tokenizer_name: Name of the tokenizer to use
        concat_tokens: Number of tokens to concatenate
        compression: Compression type for MDS files (None for no compression)
        bos_text: Text to insert at the beginning of each sequence
        eos_text: Text to insert at the end of each sequence
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = int(1e30)  # Suppress warnings about sequence length

    # Check if tokenizer has BOS/EOS tokens
    if bos_text + eos_text == '':
        test_tokens = tokenizer('test')
        if test_tokens['input_ids'][0] != tokenizer.bos_token_id and test_tokens['input_ids'][-1] != tokenizer.eos_token_id:
            print("Warning: Tokenizer does not have BOS/EOS tokens. Adding them...")
            tokenizer.add_special_tokens({
                'bos_token': '<|endoftext|>',
                'eos_token': '<|endoftext|>'
            })

    # Read input file
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Process each story
    for item in tqdm(data, desc=f"Converting stories for {split} split"):
        # Only use the story text
        text = item['story']

        # Add BOS/EOS text if specified
        if bos_text:
            text = bos_text + text
        if eos_text:
            text = text + eos_text

        # Tokenize text without padding
        tokens = tokenizer(
            text,
            truncation=False,  # Don't truncate individual stories
            padding=False,  # No padding during conversion
            add_special_tokens=False,  # BOS/EOS already added if needed
        )['input_ids']

        # Add tokens to buffer
        token_buffer.extend(tokens)

        # Write complete sequences of concat_tokens length
        while len(token_buffer) >= concat_tokens:
            sequence = token_buffer[:concat_tokens]
            token_buffer = token_buffer[concat_tokens:]  # Keep remainder for next sequence

            # Convert to numpy array and write
            sequence = np.array(sequence, dtype=np.int32)
            out.write({'tokens': sequence})

    # Write any remaining tokens if we want to keep partial sequences
    if token_buffer:
        # Pad the last sequence to concat_tokens length
        if len(token_buffer) < concat_tokens:
            token_buffer.extend([tokenizer.eos_token_id] * (concat_tokens - len(token_buffer)))
        sequence = np.array(token_buffer[:concat_tokens], dtype=np.int32)
        out.write({'tokens': sequence})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TinyStories dataset to MDS format')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSON file or directory containing JSON files')
    parser.add_argument('--out_root', type=str, required=True, help='Output directory for MDS files')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split (train/val)')
    parser.add_argument('--tokenizer_name', type=str, default='EleutherAI/gpt-neox-20b', help='Name of the tokenizer to use')
    parser.add_argument('--concat_tokens', type=int, default=2048, help='Number of tokens to concatenate')
    parser.add_argument('--compression', type=str, default=None, help='Compression type for MDS files (None for no compression)')
    parser.add_argument('--bos_text', type=str, default='', help='Text to insert at the beginning of each sequence')
    parser.add_argument('--eos_text', type=str, default='<|endoftext|>', help='Text to insert at the end of each sequence')

    args = parser.parse_args()

    # Initialize tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.model_max_length = int(1e30)  # Suppress warnings about sequence length

    # Create output directory once
    split_dir = os.path.join(args.out_root, args.split)
    os.makedirs(split_dir, exist_ok=True)

    # Initialize MDS writer once
    columns = {'tokens': 'ndarray:int32'}
    with MDSWriter(columns=columns, out=split_dir, compression=args.compression) as out:
        # Initialize buffer for token concatenation
        token_buffer = []

        # Check if input_path is a directory
        if os.path.isdir(args.input_path):
            # Get all JSON files in the directory
            input_files = sorted(glob.glob(os.path.join(args.input_path, 'data*.json')))
            if not input_files:
                raise ValueError(f"No JSON files found in directory: {args.input_path}")

            print(f"Found {len(input_files)} files to process")
            for input_file in tqdm(input_files, desc="Processing files"):
                # Read input file
                with open(input_file, 'r') as f:
                    data = json.load(f)

                # Process each story
                for item in tqdm(data, desc=f"Converting stories from {os.path.basename(input_file)}"):
                    # Only use the story text
                    text = item['story']

                    # Add BOS/EOS text if specified
                    if args.bos_text:
                        text = args.bos_text + text
                    if args.eos_text:
                        text = text + args.eos_text

                    # Tokenize text without padding
                    tokens = tokenizer(
                        text,
                        truncation=False,  # Don't truncate individual stories
                        padding=False,  # No padding during conversion
                        add_special_tokens=False,  # BOS/EOS already added if needed
                    )['input_ids']

                    # Add tokens to buffer
                    token_buffer.extend(tokens)

                    # Write complete sequences of concat_tokens length
                    while len(token_buffer) >= args.concat_tokens:
                        sequence = token_buffer[:args.concat_tokens]
                        token_buffer = token_buffer[args.concat_tokens:]  # Keep remainder for next sequence

                        # Convert to numpy array and write
                        sequence = np.array(sequence, dtype=np.int32)
                        out.write({'tokens': sequence})

                # Write any remaining tokens if we want to keep partial sequences
                if token_buffer:
                    # Pad the last sequence to concat_tokens length
                    if len(token_buffer) < args.concat_tokens:
                        token_buffer.extend([tokenizer.eos_token_id] * (args.concat_tokens - len(token_buffer)))
                    sequence = np.array(token_buffer[:args.concat_tokens], dtype=np.int32)
                    out.write({'tokens': sequence})
        else:
            # Process single file
            convert_tinystories_to_mds(
                input_path=args.input_path,
                out_root=args.out_root,
                split=args.split,
                tokenizer_name=args.tokenizer_name,
                concat_tokens=args.concat_tokens,
                compression=args.compression,
                bos_text=args.bos_text,
                eos_text=args.eos_text,
            )
