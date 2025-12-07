#!/usr/bin/env python3
"""
KARI Multiple Video Processing Example
=====================================

This script shows how to process multiple video probability matrices
with the KARI system for temporal action segmentation enhancement.
"""

import numpy as np
from utils import read_grammar, read_mapping_dict
import bep as BEP

def process_multiple_videos(video_matrices, grammar_path, mapping_file):
    """
    Process multiple video probability matrices with KARI enhancement.

    Args:
        video_matrices: List of [T, C] numpy arrays (probability matrices)
        grammar_path: Path to PCFG grammar file
        mapping_file: Path to action mapping file

    Returns:
        List of dictionaries with processing results
    """

    # Load grammar and mapping
    actions_dict = read_mapping_dict(mapping_file)
    grammar = read_grammar(grammar_path, index=True, mapping=actions_dict)

    # Initialize BEP parser
    parser = BEP.BreadthFirstEarley(grammar, prior_flag=True, mapping=actions_dict)

    results = []

    for i, prob_matrix in enumerate(video_matrices):
        print(f"Processing video {i+1}/{len(video_matrices)} - Shape: {prob_matrix.shape}")

        # Normalize probabilities (ensure each row sums to 1.0)
        if prob_matrix.shape[1] == 19:  # 19 action classes for 50 salads
            prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

        # Apply KARI refinement
        try:
            refined_sequence, confidence = parser.parse(prob_matrix, prune=20, str_len=25)

            results.append({
                'video_id': i,
                'original_shape': prob_matrix.shape,
                'refined_sequence': refined_sequence,
                'confidence': confidence
            })

        except Exception as e:
            print(f"Error processing video {i}: {e}")
            results.append({
                'video_id': i,
                'original_shape': prob_matrix.shape,
                'error': str(e)
            })

    return results

# Example usage
if __name__ == "__main__":
    # Example: Process 3 videos with different lengths
    np.random.seed(42)  # For reproducible results

    video_matrices = [
        np.random.rand(50, 19),  # Video 1: 50 frames
        np.random.rand(75, 19),  # Video 2: 75 frames
        np.random.rand(40, 19),  # Video 3: 40 frames
    ]

    # Your custom PCFG and mapping files
    grammar_path = 'induced_grammars/50salads/split_custom/50salads.pcfg'
    mapping_file = 'source_50salads/mapping.txt'

    # Process all videos
    results = process_multiple_videos(video_matrices, grammar_path, mapping_file)

    print(f"\n✅ Successfully processed {len(results)} videos!")
    for result in results:
        if 'error' not in result:
            print(f"Video {result['video_id']}: {result['original_shape']} → Refined sequence length: {len(result['refined_sequence'])}")
        else:
            print(f"Video {result['video_id']}: Error - {result['error']}")
