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
    HARVESTER_VALIDATION_SOURCE_PATHS,
    HarvesterValidationRunFailed,
    HarvesterValidationSummary,
    harvester_validation_source_hashes,
    load_harvester_validation_summary,
    reseal_pinned_public_harvester_validation_summary,
    run_harvester_validation_cycle,
    validate_harvester_validation_summary,
)

__all__ = [
    "FIXED_OPENALEX_QUERY",
    "FIXED_RESULT_LIMIT",
    "HARVESTER_VALIDATION_SOURCE_PATHS",
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
    "harvester_validation_source_hashes",
    "load_harvester_validation_summary",
    "reseal_pinned_public_harvester_validation_summary",
    "run_harvester_validation_cycle",
    "validate_harvester_validation_summary",
]
