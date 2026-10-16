# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Parsing utilities for LLM agent outputs."""

import re


def extract_block(response, tags=None):
    """Extract the first tagged block from an LLM response.

    Checks ``tags`` in order (first match wins), then falls back to a
    ```python ... ``` fenced block.

    Args:
        response: LLM response text.
        tags: tag names to search for, in priority order
            (default: ``['hypothesis', 'code']``).

    Returns:
        ``(block_type, content)`` where ``block_type`` is the matched tag name,
        ``'code'`` for a fenced fallback, or ``'none'`` if nothing matched.
    """
    if tags is None:
        tags = ['hypothesis', 'code']

    for tag in tags:
        m = re.search(f'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
        if m:
            return tag, m.group(1).strip()

    m = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if m:
        return 'code', m.group(1).strip()

    return 'none', ''
