#!/usr/bin/env python3
"""
Dataset Merger Script
Merges multiple CSV files into a single dataset-old and provides token analysis.
Designed for Ryuuko Itsuki chatbot dataset-old preparation.
"""

import pandas as pd
import os
import tiktoken
from datetime import datetime


def count_tokens(text, encoding_name="cl100k_base"):
    """
    Count tokens in text using OpenAI's tokenizer.
    cl100k_base is used by GPT-3.5-turbo and GPT-4.

    Args:
        text (str): Input text to count tokens
        encoding_name (str): Tokenizer encoding name

    Returns:
        int: Number of tokens
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"⚠️  Token counting error: {e}")
        return 0


def calculate_dataset_tokens(df):
    """
    Calculate total tokens in the dataset-old across all prompts and responses.

    Args:
        df (DataFrame): Dataset containing prompts and responses

    Returns:
        dict: Token statistics
    """
    total_prompt_tokens = 0
    total_response_tokens = 0
    total_conversation_tokens = 0

    print("🔢 Counting tokens...")

    for index, row in df.iterrows():
        prompt = str(row['prompt']) if pd.notna(row['prompt']) else ""
        response = str(row['response']) if pd.notna(row['response']) else ""

        prompt_tokens = count_tokens(prompt)
        response_tokens = count_tokens(response)
        conversation_tokens = prompt_tokens + response_tokens

        total_prompt_tokens += prompt_tokens
        total_response_tokens += response_tokens
        total_conversation_tokens += conversation_tokens

    return {
        'total_prompt_tokens': total_prompt_tokens,
        'total_response_tokens': total_response_tokens,
        'total_conversation_tokens': total_conversation_tokens,
        'average_prompt_tokens': total_prompt_tokens // len(df),
        'average_response_tokens': total_response_tokens // len(df),
        'average_conversation_tokens': total_conversation_tokens // len(df)
    }


def merge_datasets():
    """
    Main function to merge multiple CSV files into a single dataset-old.
    Performs validation, deduplication, and comprehensive analysis.
    """
    # File list to merge - order matters for priority in duplicate resolution
    files_to_merge = ['backstory.csv', 'identity.csv', 'base.csv']

    print("🚀 Starting dataset-old merge process...")
    print("=" * 50)

    # Validate all files exist before processing
    missing_files = []
    for file in files_to_merge:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Error: Missing files: {missing_files}")
        print("💡 Please ensure all files are in the same directory.")
        return None

    print("✅ All required files found!")

    try:
        # Read and process each CSV file
        dataframes = []
        file_stats = {}

        for file_path in files_to_merge:
            print(f"\n📖 Reading {file_path}...")

            # Read CSV with error handling for encoding issues
            df = pd.read_csv(file_path, encoding='utf-8')

            # Store file statistics
            file_stats[file_path] = {
                'rows': len(df),
                'columns': len(df.columns),
                'topics': df['topic'].nunique() if 'topic' in df.columns else 0
            }

            # Add source file information for tracking
            df['source_file'] = file_path
            dataframes.append(df)

            print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Merge all dataframes
        print(f"\n🔄 Merging {len(dataframes)} datasets...")
        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)

        # Record statistics before deduplication
        initial_row_count = len(merged_df)

        # Remove duplicate conversations (same prompt + response)
        print("🔍 Removing duplicates...")
        merged_df = merged_df.drop_duplicates(
            subset=['prompt', 'response'],
            keep='first'  # Keep the first occurrence (prioritizes earlier files)
        )

        final_row_count = len(merged_df)
        duplicates_removed = initial_row_count - final_row_count

        # Sort by topic for better organization
        if 'topic' in merged_df.columns:
            merged_df = merged_df.sort_values('topic')

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'dataset_{timestamp}.csv'

        # Save the merged dataset-old
        print(f"\n💾 Saving merged dataset-old...")
        merged_df.to_csv(output_filename, index=False, encoding='utf-8')
        merged_df.to_csv('dataset-old.csv', index=False, encoding='utf-8')  # Main file

        # Calculate token statistics
        token_stats = calculate_dataset_tokens(merged_df)

        # Generate comprehensive report
        print("\n" + "=" * 50)
        print("📊 **MERGE COMPLETE - FINAL REPORT**")
        print("=" * 50)

        # File statistics
        print(f"\n📁 **FILE STATISTICS:**")
        for file_path, stats in file_stats.items():
            print(f"   📄 {file_path}:")
            print(f"      Rows: {stats['rows']:>6}")
            print(f"      Columns: {stats['columns']:>4}")
            print(f"      Topics: {stats['topics']:>4}")

        # Merge statistics
        print(f"\n🔄 **MERGE STATISTICS:**")
        print(f"   Total rows before merge: {initial_row_count}")
        print(f"   Total rows after merge:  {final_row_count}")
        print(f"   Duplicates removed:      {duplicates_removed}")
        print(f"   Deduplication rate:      {(duplicates_removed / initial_row_count) * 100:.1f}%")

        # Token statistics
        print(f"\n🔢 **TOKEN STATISTICS (cl100k_base):**")
        print(f"   Total prompt tokens:     {token_stats['total_prompt_tokens']:>8}")
        print(f"   Total response tokens:   {token_stats['total_response_tokens']:>8}")
        print(f"   Total conversation tokens: {token_stats['total_conversation_tokens']:>8}")
        print(f"   Average prompt tokens:   {token_stats['average_prompt_tokens']:>8}")
        print(f"   Average response tokens: {token_stats['average_response_tokens']:>8}")
        print(f"   Average conversation:    {token_stats['average_conversation_tokens']:>8}")

        # Dataset composition
        print(f"\n📈 **DATASET COMPOSITION:**")
        if 'topic' in merged_df.columns:
            topic_counts = merged_df['topic'].value_counts()
            for topic, count in topic_counts.items():
                percentage = (count / final_row_count) * 100
                print(f"   {topic:<20} {count:>4} rows ({percentage:>5.1f}%)")

        # Source file distribution
        print(f"\n📂 **SOURCE DISTRIBUTION:**")
        source_counts = merged_df['source_file'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / final_row_count) * 100
            print(f"   {source:<15} {count:>4} rows ({percentage:>5.1f}%)")

        # Output files information
        print(f"\n💾 **OUTPUT FILES:**")
        print(f"   Primary file:    dataset-old.csv")
        print(f"   Backup file:     {output_filename}")
        print(f"   Total size:      {os.path.getsize('dataset.csv') / 1024:.1f} KB")

        return merged_df

    except Exception as e:
        print(f"❌ Error during merge process: {e}")
        return None


def validate_dataset(df):
    """
    Perform comprehensive validation on the merged dataset-old.

    Args:
        df (DataFrame): Merged dataset-old to validate
    """
    print("\n" + "=" * 50)
    print("🔍 **DATASET VALIDATION**")
    print("=" * 50)

    if df is None:
        print("❌ No dataset-old to validate.")
        return

    try:
        # Basic structure validation
        print(f"📝 Basic Structure:")
        print(f"   Total rows: {len(df)}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Columns: {list(df.columns)}")

        # Check for missing data
        print(f"\n📊 Data Quality Check:")
        missing_data = df.isnull().sum()
        missing_columns = []

        for column, missing_count in missing_data.items():
            if missing_count > 0:
                missing_columns.append(column)
                print(f"   ⚠️  {column}: {missing_count} missing values")

        if not missing_columns:
            print("   ✅ No missing data found")

        # Check data types
        print(f"\n🔧 Data Types:")
        for column in df.columns:
            dtype = df[column].dtype
            unique_count = df[column].nunique()
            print(f"   {column:<20} {str(dtype):<10} {unique_count:>4} unique values")

        # Validate required columns for chatbot training
        required_columns = ['prompt', 'response', 'topic']
        missing_required = [col for col in required_columns if col not in df.columns]

        if missing_required:
            print(f"\n❌ Missing required columns: {missing_required}")
        else:
            print(f"\n✅ All required columns present")

    except Exception as e:
        print(f"❌ Validation error: {e}")


def main():
    """
    Main execution function with comprehensive error handling.
    """
    try:
        # Check if tiktoken is available for token counting
        try:
            import tiktoken
            token_counting_available = True
        except ImportError:
            print("⚠️  tiktoken not installed. Token counting disabled.")
            print("💡 Install with: pip install tiktoken")
            token_counting_available = False

        # Execute merge process
        merged_dataset = merge_datasets()

        # Perform validation
        if merged_dataset is not None:
            validate_dataset(merged_dataset)

            print("\n" + "=" * 50)
            print("🎉 **MERGE PROCESS COMPLETED SUCCESSFULLY**")
            print("=" * 50)

            # Final recommendations
            print(f"\n💡 **RECOMMENDATIONS:**")
            if len(merged_dataset) > 1000:
                print("   ✅ Dataset size is good for training")
            else:
                print("   ⚠️  Consider adding more training examples")

            if 'emotional_intensity' in merged_dataset.columns:
                avg_intensity = merged_dataset['emotional_intensity'].mean()
                print(f"   📊 Average emotional intensity: {avg_intensity:.2f}")

        else:
            print("\n❌ Merge process failed.")

    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


if __name__ == "__main__":
    main()