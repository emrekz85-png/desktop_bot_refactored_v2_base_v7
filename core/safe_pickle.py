#!/usr/bin/env python3
"""
Safe Pickle Utilities

Provides safer alternatives to pickle with validation and warnings.

SECURITY WARNING:
    Pickle can execute arbitrary code when unpickling. Only use pickle
    with data from trusted sources that you control. For untrusted data,
    use JSON or other safe serialization formats.

Best Practices:
    1. Prefer JSON for configuration and simple data structures
    2. Only use pickle for trusted, internal data
    3. Validate data integrity with hash checking
    4. Add version tracking to detect incompatible pickle files
    5. Consider alternatives: json, msgpack, protobuf
"""

import pickle
import hashlib
import logging
import warnings
from typing import Any, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class PickleSecurityWarning(UserWarning):
    """Warning for pickle security concerns."""
    pass


def safe_pickle_dump(
    obj: Any,
    file_path: str,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    compute_hash: bool = True,
) -> Optional[str]:
    """Safely dump object to pickle file with warnings and hash.

    Args:
        obj: Object to pickle
        file_path: Path to save pickle file
        protocol: Pickle protocol version
        compute_hash: If True, compute and return SHA256 hash

    Returns:
        SHA256 hash of pickled data (if compute_hash=True)

    Example:
        >>> data = {'config': [1, 2, 3]}
        >>> hash_val = safe_pickle_dump(data, 'data.pkl')
        >>> print(f"Saved with hash: {hash_val}")
    """
    warnings.warn(
        f"Saving pickle file: {file_path}. "
        "Only load this file from trusted sources!",
        PickleSecurityWarning,
        stacklevel=2
    )

    logger.warning(f"Pickling data to {file_path}")

    # Ensure parent directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Pickle data
    pickled_data = pickle.dumps(obj, protocol=protocol)

    # Compute hash if requested
    file_hash = None
    if compute_hash:
        file_hash = hashlib.sha256(pickled_data).hexdigest()
        logger.info(f"Pickle file hash: {file_hash}")

    # Write to file
    with open(file_path, 'wb') as f:
        f.write(pickled_data)

    logger.info(f"Pickled {len(pickled_data)} bytes to {file_path}")

    return file_hash


def safe_pickle_load(
    file_path: str,
    expected_hash: Optional[str] = None,
    trusted: bool = False,
) -> Any:
    """Safely load object from pickle file with validation.

    Args:
        file_path: Path to pickle file
        expected_hash: Expected SHA256 hash for validation
        trusted: If True, skip security warning (use with caution!)

    Returns:
        Unpickled object

    Raises:
        ValueError: If hash validation fails
        FileNotFoundError: If file doesn't exist

    Example:
        >>> # Save with hash
        >>> hash_val = safe_pickle_dump(data, 'data.pkl')
        >>> # Load with validation
        >>> data = safe_pickle_load('data.pkl', expected_hash=hash_val)
    """
    if not trusted:
        warnings.warn(
            f"Loading pickle file: {file_path}. "
            "SECURITY WARNING: Only load pickle files from trusted sources! "
            "Malicious pickle files can execute arbitrary code.",
            PickleSecurityWarning,
            stacklevel=2
        )

    logger.warning(f"Loading pickle file: {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    # Read file
    with open(file_path, 'rb') as f:
        pickled_data = f.read()

    # Validate hash if provided
    if expected_hash:
        actual_hash = hashlib.sha256(pickled_data).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"Pickle file hash mismatch! File may be corrupted or tampered with.\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            )

        logger.info(f"Hash validation passed: {actual_hash}")

    # Unpickle
    obj = pickle.loads(pickled_data)

    logger.info(f"Successfully loaded pickle file: {file_path}")

    return obj


def pickle_to_json_safe(obj: Any, json_path: str) -> None:
    """Convert pickle-able object to JSON (safer alternative).

    Attempts to convert object to JSON-serializable format.
    This is safer than pickle as JSON cannot execute code.

    Args:
        obj: Object to save
        json_path: Path to JSON file

    Raises:
        TypeError: If object is not JSON-serializable

    Example:
        >>> data = {'config': [1, 2, 3], 'name': 'test'}
        >>> pickle_to_json_safe(data, 'config.json')
    """
    import json

    logger.info(f"Saving as JSON (safer than pickle): {json_path}")

    # Ensure parent directory exists
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    # Try to serialize to JSON
    try:
        # Convert numpy arrays and other types to native Python
        def convert_to_native(obj):
            """Convert numpy/pandas objects to native Python types."""
            # Handle numpy arrays
            if hasattr(obj, 'tolist'):
                return obj.tolist()

            # Handle pandas DataFrames
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()

            # Handle dictionaries
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}

            # Handle lists/tuples
            if isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]

            # Return as-is for native types
            return obj

        converted = convert_to_native(obj)

        with open(json_path, 'w') as f:
            json.dump(converted, f, indent=2)

        logger.info(f"Successfully saved to JSON: {json_path}")

    except TypeError as e:
        raise TypeError(
            f"Object is not JSON-serializable: {e}\n"
            f"Consider using pickle with safety warnings instead."
        )


def test_pickle_serialization(obj: Any, verbose: bool = True) -> bool:
    """Test if object can be safely pickled and unpickled.

    Useful for testing before using multiprocessing (which uses pickle).

    Args:
        obj: Object to test
        verbose: Print detailed information

    Returns:
        True if serialization succeeded, False otherwise

    Example:
        >>> config = {'rr': 2.0, 'rsi': 70}
        >>> if test_pickle_serialization(config):
        ...     print("Config can be used with multiprocessing")
    """
    try:
        # Pickle
        pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"✓ Object can be pickled (size: {len(pickled)} bytes)")

        # Unpickle
        unpickled = pickle.loads(pickled)

        if verbose:
            print(f"✓ Object can be unpickled")

        # Basic validation
        if isinstance(obj, dict) and isinstance(unpickled, dict):
            if set(obj.keys()) != set(unpickled.keys()):
                if verbose:
                    print(f"✗ WARNING: Keys changed during serialization!")
                return False

        if verbose:
            print(f"✓ Serialization test passed")

        return True

    except Exception as e:
        if verbose:
            print(f"✗ Serialization test failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    import tempfile

    print("="*80)
    print("SAFE PICKLE UTILITIES - DEMONSTRATION")
    print("="*80)

    # Test data
    test_data = {
        'config': {
            'rr': 2.0,
            'rsi': 70,
            'symbols': ['BTCUSDT', 'ETHUSDT'],
        },
        'results': [1, 2, 3, 4, 5],
        'version': '1.0',
    }

    print("\n1. Testing pickle serialization...")
    if test_pickle_serialization(test_data):
        print("   Object is pickle-compatible ✓")

    print("\n2. Saving with hash validation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = f"{tmpdir}/test.pkl"
        json_path = f"{tmpdir}/test.json"

        # Save with hash
        file_hash = safe_pickle_dump(test_data, pickle_path)
        print(f"   Saved with hash: {file_hash[:16]}...")

        # Load with hash validation
        print("\n3. Loading with hash validation...")
        loaded_data = safe_pickle_load(pickle_path, expected_hash=file_hash)
        print(f"   Loaded successfully ✓")
        print(f"   Data matches: {loaded_data == test_data}")

        # Try to tamper with file
        print("\n4. Testing tamper detection...")
        try:
            # Load with wrong hash
            safe_pickle_load(pickle_path, expected_hash="wrong_hash")
            print("   ✗ FAILED: Tamper detection didn't work!")
        except ValueError as e:
            print(f"   ✓ Tamper detected: {str(e)[:50]}...")

        # Save as JSON (safer alternative)
        print("\n5. Saving as JSON (safer alternative)...")
        pickle_to_json_safe(test_data, json_path)
        print(f"   Saved to JSON ✓")

        import json
        with open(json_path) as f:
            json_data = json.load(f)
        print(f"   Data matches: {json_data == test_data}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("  1. Prefer JSON for configs and simple data (safer)")
    print("  2. Use pickle only for trusted, internal data")
    print("  3. Always validate with hash when possible")
    print("  4. Be aware of pickle security warnings")
    print("="*80)
