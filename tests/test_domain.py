"""Tests for Domain Layer"""

from shared.classification_schema import ClassificationResult
from shared.image_schema import ImageCategory


def test_image_category_enum():
    """Test ImageCategory enum"""
    assert ImageCategory.PEOPLE.value == "people"
    assert ImageCategory.NATURE.value == "nature"
    assert ImageCategory.TEXT.value == "text"
    assert ImageCategory.EVENTS.value == "events"


def test_classification_result():
    """Test ClassificationResult model"""
    result = ClassificationResult(
        category=ImageCategory.PEOPLE,
        confidence=0.95,
        description="Portrait photo",
    )

    assert result.category == ImageCategory.PEOPLE
    assert result.confidence == 0.95
    assert result.description == "Portrait photo"
