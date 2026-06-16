from backend.domain.db_models import ImageRecord, EmbeddingRecord
from os.path import join


def select_where(db: "session", **kwargs):
    return (
        db.query(ImageRecord, EmbeddingRecord).join(
            EmbeddingRecord, ImageRecord.id == EmbeddingRecord.image_id
        )
        # .filter_by(**kwargs)
        .all()
    )
