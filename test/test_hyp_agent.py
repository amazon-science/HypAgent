# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Smoke tests for the HypAgent package (the agent) and the scm package (the
benchmark simulator).

The discoverability test needs no third-party dependencies. The public-API test
imports the scientific stack (NumPy, pandas) and is skipped when it is not present
in the environment; install requirements.txt to run it.
"""

import importlib.util

import pytest


def test_packages_are_discoverable():
    """Both source packages are laid out where the build expects them."""
    assert importlib.util.find_spec("hyp_agent") is not None
    assert importlib.util.find_spec("scm") is not None


def test_public_api_imports():
    """The documented public API is importable when the runtime stack is present."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")

    import hyp_agent

    assert hasattr(hyp_agent, "run")
    assert hasattr(hyp_agent, "load")

    import scm

    assert hasattr(scm, "evaluate")
    assert hasattr(scm, "Simulator")
