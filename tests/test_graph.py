"""Tests for the LangGraph workflow"""

import pytest
from src.graph import create_graph, ImageState


def test_create_graph():
    """Test that the graph is created successfully"""
    graph = create_graph()
    assert graph is not None


def test_image_state():
    """Test ImageState creation"""
    state: ImageState = {
        "image_path": "test.jpg",
        "analysis": "Test analysis",
        "confidence": 0.95,
    }

    assert state["image_path"] == "test.jpg"
    assert state["analysis"] == "Test analysis"
    assert state["confidence"] == 0.95


def test_analyze_image():
    """Test image analysis node"""
    # This is a placeholder test
    # Mock the LLM response for testing
    pass


def test_validate_analysis():
    """Test analysis validation node"""
    # This is a placeholder test
    pass
