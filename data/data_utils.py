#!/usr/bin/env python3
"""
Data Util functions
"""
from pathlib import Path
from typing import List, Tuple
import re

################################################################################


def extract_sequence_number(filename: str) -> Tuple[int, str]:
    """
    Extract the sequence number from an image file.

    E.g., file_001.jpg returns 1, file_021.jpg returns 21, file_123.jpg returns 123
    Args:
        filename: (str) Image filename.

    Returns:
        (int) sequence number, or -1 if not found.
    """
    file_postfix = filename.rpartition('_')[-1]
    snum = re.findall("\d+", file_postfix)
    return int(snum[0]) if snum else -1, filename


def max_sequence(file_list: List[str]) -> Tuple[int, str]:
    """
    Get the max sequence length of all images in file_list.

    Args:
        file_list: (List[str]) List of image file names.

    Returns:
        (Tuple[int, str]): max sequence length, sequence name
    """
    res = max(file_list, key=extract_sequence_number)
    return extract_sequence_number(res)


def get_image_files_recursively(path: str, pattern: str = '*.jpg') -> List[str]:
    """
    Recursively retrieves all image files in a folder.

    Args:
        path: Root folder path to search for image files.
        pattern: Glob pattern of image files, default = '*.jpg'

    Returns:
        (List[str]): List of image filenames found in path.
    """
    return [img.name for img in Path(path).rglob(pattern)]
