"""Literature Multiverse contract and pipeline package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from literature_multiverse.config import QuestionConfig
    from literature_multiverse.models import FindingRow, PaperRecord

__all__ = ["FindingRow", "PaperRecord", "QuestionConfig", "load_question_config"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Expose convenience imports without pre-importing executable submodules."""

    if name in {"QuestionConfig", "load_question_config"}:
        from literature_multiverse import config

        return getattr(config, name)
    if name in {"FindingRow", "PaperRecord"}:
        from literature_multiverse import models

        return getattr(models, name)
    raise AttributeError(name)
