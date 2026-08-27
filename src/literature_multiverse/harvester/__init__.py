"""Source-agnostic open literature harvesting."""

from .archive import ArchivedPayload, ArchiveIntegrityError, ImmutableArchive
from .contracts import (
    FullTextFetch,
    FullTextSource,
    HarvestDocument,
    RetrievedPayload,
    SearchBatch,
    SearchSource,
)
from .http import HarvestHttpError, PoliteHttpClient, UnsafeHarvestUrl
from .pipeline import HarvestQuery, HarvestResult, LiteratureHarvester, document_to_occurrence
from .sources import (
    ArxivFullTextSource,
    CompositeFullTextSource,
    DirectOpenAccessSource,
    EuropePmcFullTextSource,
    FrozenCorpusSource,
    FrozenFullTextSource,
    OpenAlexSearchSource,
    document_from_openalex,
)
from .validation import (
    FIXED_OPENALEX_QUERY,
    FIXED_RESULT_LIMIT,
    HarvesterValidationRunFailed,
    HarvesterValidationSummary,
    load_harvester_validation_summary,
    run_harvester_validation_cycle,
)

__all__ = [
    "FIXED_OPENALEX_QUERY",
    "FIXED_RESULT_LIMIT",
    "ArchiveIntegrityError",
    "ArchivedPayload",
    "ArxivFullTextSource",
    "CompositeFullTextSource",
    "DirectOpenAccessSource",
    "EuropePmcFullTextSource",
    "FrozenCorpusSource",
    "FrozenFullTextSource",
    "FullTextFetch",
    "FullTextSource",
    "HarvestDocument",
    "HarvestHttpError",
    "HarvestQuery",
    "HarvestResult",
    "HarvesterValidationRunFailed",
    "HarvesterValidationSummary",
    "ImmutableArchive",
    "LiteratureHarvester",
    "OpenAlexSearchSource",
    "PoliteHttpClient",
    "RetrievedPayload",
    "SearchBatch",
    "SearchSource",
    "UnsafeHarvestUrl",
    "document_from_openalex",
    "document_to_occurrence",
    "load_harvester_validation_summary",
    "run_harvester_validation_cycle",
]
