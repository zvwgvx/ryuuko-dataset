#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Token counting will be approximate.")
    print("Install with: pip install tiktoken\n")


# System prompt removed to avoid overfitting
# Inject system prompt separately during inference instead

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to cl100k_base encoding (used by GPT-3.5/GPT-4)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
    else:
        # Approximate token count: ~4 chars per token for English, ~2-3 for Vietnamese
        return len(text) // 3


def analyze_dataset_tokens(jsonl_file: str) -> dict:
    """Analyze token counts in dataset (no system prompt)"""
    total_tokens = 0
    total_conversations = 0
    user_tokens = 0
    assistant_tokens = 0

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            total_conversations += 1

            for msg in entry['messages']:
                tokens = count_tokens(msg['content'])
                total_tokens += tokens

                if msg['role'] == 'user':
                    user_tokens += tokens
                elif msg['role'] == 'assistant':
                    assistant_tokens += tokens

    return {
        'total_conversations': total_conversations,
        'total_tokens': total_tokens,
        'user_tokens': user_tokens,
        'assistant_tokens': assistant_tokens,
        'avg_tokens_per_conversation': total_tokens / total_conversations if total_conversations > 0 else 0
    }


def create_clean_dataset(jsonl_file: str):
    """Create a clean, human-readable version of the dataset"""
    clean_file = jsonl_file.replace('.jsonl', '_clean.json')

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)

    # Write with pretty formatting
    with open(clean_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Created clean version: {clean_file}")
    return clean_file


def csv_to_jsonl(csv_file: str, output_file: str):
    """
    Convert CSV to JSONL format for fine-tuning

    Note: System prompt is NOT included to avoid overfitting.
    Inject INSTRUCTION.MD separately during inference.

    JSONL Format (no system prompt):
    {
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "response"}
        ],
        "metadata": {
            "topic": "...",
            "tone": "..."
        }
    }
    """

    # Read CSV and convert to JSONL
    jsonl_data = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Create conversation entry
            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": row['prompt']
                    },
                    {
                        "role": "assistant",
                        "content": row['response']
                    }
                ],
                "metadata": {
                    "topic": row['topic'],
                    "tone": row['tone']
                }
            }

            jsonl_data.append(entry)

    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Converted {len(jsonl_data)} rows from {csv_file} to {output_file}")

    return len(jsonl_data)


def create_train_test_split(jsonl_file: str, test_ratio: float = 0.1):
    """Split dataset into train and test sets"""
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Shuffle data
    import random
    random.seed(42)
    random.shuffle(data)

    # Split
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Write train set
    train_file = jsonl_file.replace('.jsonl', '_train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Write test set
    test_file = jsonl_file.replace('.jsonl', '_test.jsonl')
    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Train set: {len(train_data)} rows -> {train_file}")
    print(f"✓ Test set: {len(test_data)} rows -> {test_file}")

    return train_file, test_file


def preview_jsonl(jsonl_file: str, num_samples: int = 3):
    """Preview a few samples from JSONL"""
    print(f"\n📋 Preview {num_samples} samples from {jsonl_file}:")
    print("=" * 80)

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            entry = json.loads(line)
            print(f"\nSample {i+1}:")
            print(f"  User: {entry['messages'][0]['content']}")
            print(f"  Assistant: {entry['messages'][1]['content']}")
            if 'metadata' in entry:
                print(f"  [Topic: {entry['metadata']['topic']}, Tone: {entry['metadata']['tone']}]")
            print("-" * 80)


def main():
    """Main function"""
    print("🚀 Ryuuko Dataset Converter - CSV to JSONL\n")

    # Paths
    csv_file = "dataset/dataset.csv"
    output_file = "dataset/dataset.jsonl"

    # Check if files exist
    if not Path(csv_file).exists():
        print(f"❌ Error: {csv_file} not found!")
        return

    # Convert CSV to JSONL (without system prompt to avoid overfitting)
    print("📝 Converting CSV to JSONL (no system prompt)...")
    total = csv_to_jsonl(csv_file, output_file)

    # Preview
    preview_jsonl(output_file, num_samples=3)

    # Create train/test split
    print("\n📊 Splitting dataset into train/test sets...")
    train_file, test_file = create_train_test_split(output_file, test_ratio=0.1)

    # Create clean version
    print("\n🧹 Creating clean, readable version...")
    clean_file = create_clean_dataset(output_file)
    create_clean_dataset(train_file)
    create_clean_dataset(test_file)

    # Analyze tokens
    print("\n🔢 Analyzing token counts...")
    stats = analyze_dataset_tokens(output_file)
    train_stats = analyze_dataset_tokens(train_file)
    test_stats = analyze_dataset_tokens(test_file)

    print(f"\n📊 Token Statistics (Full Dataset):")
    print(f"  Total conversations: {stats['total_conversations']}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  - User tokens: {stats['user_tokens']:,}")
    print(f"  - Assistant tokens: {stats['assistant_tokens']:,}")
    print(f"  Average tokens/conversation: {stats['avg_tokens_per_conversation']:.1f}")

    print(f"\n📊 Training Set:")
    print(f"  Conversations: {train_stats['total_conversations']}")
    print(f"  Total tokens: {train_stats['total_tokens']:,}")

    print(f"\n📊 Test Set:")
    print(f"  Conversations: {test_stats['total_conversations']}")
    print(f"  Total tokens: {test_stats['total_tokens']:,}")

    print(f"\n✅ Completed! Total {total} conversations")
    print("\nGenerated files:")
    print(f"  📄 {output_file} (full dataset - JSONL for training)")
    print(f"  📄 {clean_file} (clean readable version - JSON)")
    print(f"  📄 {train_file} (training set - JSONL)")
    print(f"  📄 {train_file.replace('.jsonl', '_clean.json')} (training set - clean)")
    print(f"  📄 {test_file} (test set - JSONL)")
    print(f"  📄 {test_file.replace('.jsonl', '_clean.json')} (test set - clean)")
    print("\n💡 You can now use these JSONL files to fine-tune your model!")
    print("   Use *_clean.json files for human review/editing")
    print("\n⚠️  Note: System prompt removed to avoid overfitting.")
    print("   Inject INSTRUCTION.MD separately during inference.")

    if not TIKTOKEN_AVAILABLE:
        print("\n⚠️  Note: Install tiktoken for accurate token counting:")
        print("   pip install tiktoken")


if __name__ == "__main__":
    main()
