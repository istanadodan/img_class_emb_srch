"""Tests for Domain Layer"""

import pytest
from backend.domain.models import ImageCategory, ClassificationResult


def test_image_category_enum():
    """Test ImageCategory enum"""
    assert ImageCategory.PEOPLE.value == "people"
    assert ImageCategory.NATURE.value == "nature"
    assert ImageCategory.TEXT.value == "text"
    assert ImageCategory.EVENTS.value == "events"


def test_classification_result():
    """Test ClassificationResult model"""
    result = ClassificationResult(
        image_path="/path/to/image.jpg",
        category=ImageCategory.PEOPLE,
        confidence=0.95,
        description="Portrait photo",
    )

    assert result.image_path == "/path/to/image.jpg"
    assert result.category == ImageCategory.PEOPLE
    assert result.confidence == 0.95
    assert result.description == "Portrait photo"
