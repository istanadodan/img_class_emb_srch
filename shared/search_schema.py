"""Search Schema - 이미지 검색 결과 구조 정의"""

from pydantic import BaseModel, Field
from typing import List

class ImageSearchResult(BaseModel):
    """검색 결과 내 개별 이미지 정보"""
    image_id: int = Field(description="이미지의 고유 데이터베이스 ID")
    similarity: float = Field(description="이미지 유사도 점수 (0.0에서 1.0 사이)")
    path: str = Field(description="이미지 파일의 전체 경로")
    justification: str = Field(description="이 이미지가 검색 결과에 포함된 개별적인 이유와 분석 내용")

class SearchResponse(BaseModel):
    """최종 이미지 검색 응답 구조"""
    results: List[ImageSearchResult] = Field(description="검색된 이미지들의 상세 목록")
