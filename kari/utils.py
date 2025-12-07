"""
Function of 'get_pcfg' and 'read_grammar' is originated from https://github.com/SiyuanQi-zz/generalized-earley-parser/blob/master/src/python/parser/grammarutils.py.
Function of 'read_mapping_dict' is originated from https://github.com/gongda0e/FUTR/blob/main/utils.py.
"""

import collections
import os
import time
import itertools
import json

import numpy as np
import nltk

import functools


def get_pcfg(rules, index=False, mapping=None):
    root_rules = list()
    non_terminal_rules = list()
    grammar_rules = list()

    # Create reverse mapping if needed for numeric token conversion
    reverse_mapping = None
    if index and mapping:
        reverse_mapping = {str(v): k for k, v in mapping.items()}

    for rule in rules:
        tokens = rule.split()
        for i in range(len(tokens)):
            token = tokens[i]
            if token[0] == 'E':
                tokens[i] = tokens[i].replace('E', 'OR')
            elif token[0] == 'P':
                tokens[i] = tokens[i].replace('P', 'AND')
            elif index and mapping and token[0] == "'":
                stripped_token = token.strip("'")
                # Try to convert numeric tokens back to string names
                if reverse_mapping and stripped_token in reverse_mapping:
                    tokens[i] = "'{}'".format(reverse_mapping[stripped_token])
                # If not numeric or not in reverse mapping, try original mapping
                elif stripped_token in mapping:
                    tokens[i] = "'{}'".format(mapping[stripped_token])
            elif token[0] == "I":
                pass
        rule = ' '.join(tokens)

        if rule.startswith('S'):
            root_rules.append(rule)
        else:
            non_terminal_rules.append(rule)

    for k, v in collections.Counter(root_rules).items():
        grammar_rules.append(k + ' [{}]'.format(float(v) / len(root_rules)))
    grammar_rules.extend(non_terminal_rules)
    return grammar_rules


def read_grammar(filename, index=False, mapping=None, insert=True):
    with open(filename) as f:
        rules = [rule.strip() for rule in f.readlines()]
        if insert:
            rules.insert(0, 'GAMMA -> S [1.0]')
        grammar_rules = get_pcfg(rules, index, mapping)
        grammar = nltk.PCFG.fromstring(grammar_rules)
    return grammar


def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        if not a.strip():  # Skip empty lines
            continue
        parts = a.split()
        if len(parts) < 2:
            continue
        index = int(parts[0])
        action_name = parts[1]
        # Map action name to index (for datasets like GTEA that use action names)
        actions_dict[action_name] = index
        # Also map numeric string to index (for datasets like 50salads/breakfast that are already encoded)
        actions_dict[str(index)] = index

    return actions_dict


def compute_parser_accuracy(matrices_file, parser, sample_stride=100, prune=20, str_len=25, verbose=True):
    """
    Compute parser accuracy by processing softmax matrices and returning predicted sequences.

    Args:
        matrices_file: Path to pickle file containing softmax matrices (list of numpy arrays).
                       Backward compatible: can also be a directory of per-video npy/npz/pkl
                       files, or a JSON manifest listing per-video files. Legacy pickle
                       behavior is unchanged.
        parser: BEP parser instance
        sample_stride: Frame sampling rate for parsing (default: 100)
        prune: Number of states to keep in parser queue (default: 20)
        str_len: Maximum sequence length to parse (default: 25)
        verbose: Whether to print progress information (default: True)

    Returns:
        List of dictionaries containing results for each matrix:
        - video_id: Index of the video
        - shape: Shape of the softmax matrix
        - best_sequence: Predicted action sequence
        - confidence: Log probability of the best sequence
        - labels: Frame-level action labels
        - tokens: Token sequence
        - token_positions: Token positions in frames
    """
    import pickle
    import io
    import torch

    # Custom unpickler to handle PyTorch tensors on CPU
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)

    def _load_single_video(path):
        """Load a single video matrix from npy/npz/pkl, staying CPU-only."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            # Memory-map for lazy loading (reduces memory for large files)
            return np.load(path, mmap_mode='r')
        elif ext == '.npz':
            # .npz files are zipped archives; extract the first array
            with np.load(path) as data:
                # Return first array found (common convention: 'arr_0' or first key)
                key = list(data.keys())[0] if data.keys() else 'arr_0'
                return data[key]
        # Default to pickle load with CPU_Unpickler
        with open(path, 'rb') as f:
            return CPU_Unpickler(f).load()

    def _iter_video_paths():
        """Yield (idx, path) pairs if matrices_file is a dir or JSON manifest."""
        if os.path.isdir(matrices_file):
            files = sorted([
                os.path.join(matrices_file, f)
                for f in os.listdir(matrices_file)
                if os.path.splitext(f)[1].lower() in ('.npy', '.npz', '.pkl')
            ])
            return enumerate(files)
        if matrices_file.lower().endswith('.json') and os.path.exists(matrices_file):
            with open(matrices_file, 'r') as mf:
                manifest = json.load(mf)
            if isinstance(manifest, dict) and 'videos' in manifest:
                entries = manifest['videos']
                if not entries:
                    raise ValueError(
                        f"JSON manifest has empty 'videos' array: {matrices_file}")
                base_dir = manifest.get(
                    'base_dir', os.path.dirname(matrices_file))
                paths = [os.path.join(base_dir, e['path'] if isinstance(
                    e, dict) else e) for e in entries]
            else:
                if not manifest:
                    raise ValueError(
                        f"JSON manifest is empty: {matrices_file}")
                base_dir = os.path.dirname(matrices_file)
                paths = [os.path.join(base_dir, e) if not os.path.isabs(
                    e) else e for e in manifest]
            return enumerate(paths)
        return None

    # Streaming mode if directory or manifest is provided; otherwise fallback to legacy pickle load
    stream_iter = _iter_video_paths()
    streaming_mode = stream_iter is not None

    if streaming_mode:
        # Capture count for logging without holding matrices in memory
        stream_list = list(stream_iter)
        num_videos = len(stream_list)
        # stream_list already contains (idx, path) tuples from _iter_video_paths()
        video_paths = stream_list  # No need to re-enumerate
        if verbose:
            print(
                f'Loaded {num_videos} videos (streaming) from {matrices_file}')
    else:
        # Load matrices using CPU_Unpickler to handle CUDA tensors
        with open(matrices_file, 'rb') as f:
            data = CPU_Unpickler(f).load()

        if isinstance(data, dict):
            video_matrices = list(data.values())
        else:
            # Convert to a mutable list so we can release entries as we go
            try:
                video_matrices = list(data)
            except TypeError:
                # If data is a single array instead of a sequence of arrays, wrap it
                video_matrices = [data]

        # Convert PyTorch tensors to numpy arrays if needed
        # Process and convert on-the-fly to avoid keeping all matrices in memory
        num_videos = len(video_matrices)
        if verbose:
            print(f'Loaded {num_videos} videos from {matrices_file}')

    # Try to get memory info if available (for monitoring)
    try:
        import psutil
        process = psutil.Process()
        has_psutil = True
    except ImportError:
        has_psutil = False

    results = []

    iterator = video_paths if streaming_mode else enumerate(video_matrices)

    for i, matrix_or_path in iterator:
        # Convert matrix on-the-fly instead of storing all converted matrices
        if streaming_mode:
            # In streaming mode, matrix_or_path is a file path string
            matrix = _load_single_video(matrix_or_path)
        else:
            # In non-streaming mode, matrix_or_path is already a matrix
            matrix = matrix_or_path
        if isinstance(matrix, torch.Tensor):
            # Convert tensor to numpy: CPU -> numpy -> squeeze leading dim if needed
            probs = matrix.cpu().numpy()
            if probs.ndim > 2 and 1 in probs.shape:
                probs = np.squeeze(probs)
            # Explicitly delete tensor reference and clear CUDA cache if available
            del matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            probs = np.asarray(matrix)
            del matrix

        # If extra singleton dims remain (e.g., (1, C, T) or (C, T, 1)), squeeze them
        if probs.ndim > 2 and 1 in probs.shape:
            probs = np.squeeze(probs)
        # If still not 2D after squeezing, bail out with a clear error
        if probs.ndim != 2:
            raise ValueError(
                f"Softmax matrix for video {i} is not 2D after squeezing: shape={probs.shape}")

        # Aggressively free memory from the original list (if not streaming)
        # Note: Setting to None helps signal GC but doesn't guarantee immediate release
        if not streaming_mode and i < len(video_matrices):
            video_matrices[i] = None

        # Capture shape before processing
        probs_shape = probs.shape

        # Memory monitoring
        if verbose:
            if has_psutil:
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                print(
                    f"Processing video {i+1}/{num_videos} - shape: {probs_shape} | Memory: {mem_mb:.1f} MB")
            else:
                print(
                    f"Processing video {i+1}/{num_videos} - shape: {probs_shape}")

        try:
            # Parse with BEP (returns: best_sequence, best_log_prob, cached)
            # Ensure matrix is time-major: (num_frames, num_classes)
            mapping_size = len(getattr(parser, '_mapping', {}) or [])
            if probs.ndim != 2:
                raise ValueError(
                    f"Softmax matrix for video {i} is not 2D: shape={probs.shape}")

            # Decide whether to transpose based on shape and mapping size
            if mapping_size:
                if probs.shape[0] == mapping_size and probs.shape[1] != mapping_size:
                    transpose_needed = True  # classes x time -> transpose
                elif probs.shape[1] == mapping_size:
                    transpose_needed = False  # already time x classes
                else:
                    transpose_needed = probs.shape[0] < probs.shape[1]
            else:
                transpose_needed = probs.shape[0] < probs.shape[1]

            probs_time_major = probs.T.copy() if transpose_needed else probs.copy()

            # Free the original probs immediately since we have the copy
            del probs

            # Parse with sampled frames to reduce memory usage
            # Note: parse() calls _parse_init() which resets parser state, so caches are fresh
            best_sequence, best_log_prob, _ = parser.parse(
                probs_time_major[::sample_stride], prune=prune, str_len=str_len)

            # Compute refined frame-level labels at full resolution
            # compute_labels_segment also expects (num_frames, num_classes)
            # This uses self._best_l from parse(), so must be called before clearing caches
            labels, tokens, token_positions = parser.compute_labels_segment(
                probs_time_major)

            results.append({
                'video_id': i,
                'shape': probs_shape,
                'best_sequence': best_sequence,
                'confidence': best_log_prob,
                'labels': labels,
                'tokens': tokens,
                'token_positions': token_positions,
            })

            # Clean up to free memory
            del probs_time_major
            del labels, tokens, token_positions, best_sequence

            # Clear parser caches AFTER processing to prevent memory accumulation across videos
            # This is safe because parse() resets state via _parse_init() for each video
            cache = getattr(parser, '_cached_log_prob', None)
            if cache:
                cache.clear()
            cache = getattr(parser, '_cached_grammar_prob', None)
            if cache:
                cache.clear()
            if hasattr(parser, 'parsed_str'):
                parser.parsed_str = []

            import gc
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except MemoryError as e:
            print(f"⚠️  Memory error processing video {i+1}: {e}")
            print(
                f"   Skipping this video. Consider reducing sample_stride, prune, or str_len parameters.")
            # Clean up any partial results
            if 'probs_time_major' in locals():
                del probs_time_major
            if 'probs' in locals():
                del probs
            # Clear parser caches
            cache = getattr(parser, '_cached_log_prob', None)
            if cache:
                cache.clear()
            cache = getattr(parser, '_cached_grammar_prob', None)
            if cache:
                cache.clear()

            results.append({
                'video_id': i,
                'shape': probs_shape,
                'error': f'MemoryError: {str(e)}',
                'labels': None,
                'tokens': None,
                'token_positions': None,
                'best_sequence': None,
                'confidence': None,
            })
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  Error processing video {i+1}: {e}")
            print(f"   Skipping this video.")
            # Clean up any partial results
            if 'probs_time_major' in locals():
                del probs_time_major
            if 'probs' in locals():
                del probs
            # Clear parser caches
            cache = getattr(parser, '_cached_log_prob', None)
            if cache:
                cache.clear()
            cache = getattr(parser, '_cached_grammar_prob', None)
            if cache:
                cache.clear()

            results.append({
                'video_id': i,
                'shape': probs_shape,
                'error': str(e),
                'labels': None,
                'tokens': None,
                'token_positions': None,
                'best_sequence': None,
                'confidence': None,
            })
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final cleanup for this iteration
        if i % 5 == 0:  # More aggressive cleanup every 5 videos
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final cleanup: free the entire video_matrices list
    if not streaming_mode:
        del video_matrices
        del data
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n✅ Finished processing {len(results)} videos.")
    return results
