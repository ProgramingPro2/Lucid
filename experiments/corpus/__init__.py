"""G-1 corpus construction (docs/CORPUS.md).

The package is machinery, not data. A real corpus requires a live harvest and a
per-document NLL table; neither is committed here.
"""

from .pipeline import (
    CARD_KEYS,
    build_card,
    clean_text,
    deduplicate,
    document_stratum,
    drop_low_nll,
    filter_by_date,
    filter_by_license,
    split_documents,
    token_stream_sha256,
)
from .sources import (
    LICENSE_ALLOWLIST,
    BioRxivClient,
    Document,
    FakeClient,
    FdaLabelClient,
    load_cutoff,
    normalise_license,
)

__all__ = [
    "CARD_KEYS",
    "LICENSE_ALLOWLIST",
    "BioRxivClient",
    "Document",
    "FakeClient",
    "FdaLabelClient",
    "build_card",
    "clean_text",
    "deduplicate",
    "document_stratum",
    "drop_low_nll",
    "filter_by_date",
    "filter_by_license",
    "load_cutoff",
    "normalise_license",
    "split_documents",
    "token_stream_sha256",
]
