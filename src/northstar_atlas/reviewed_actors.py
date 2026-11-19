"""Small source-reviewed claim vocabulary, not general semantic entailment.

Only claims citing the exact reviewed HR-001 initiation snapshot are restricted.
Unknown wording is withheld, never repaired or inferred from actor keywords.
Other snapshots still require their own semantic review. No IO or model calls.
"""
from northstar_atlas.evidence import CitationReference, ContextPackage, validate_citations
from northstar_atlas.grounding import GroundingError, match_quotation, normalize_whitespace
from northstar_atlas.ingestion.document_loader import APPROVED


# Reviewed against the complete synthetic HR-001 v1.0 §4.1. A future source or
# status change needs a new review; the rule must not confer authority on drafts.
_SNAPSHOT = ('HR-001', '1.0', APPROVED,
             'cca217a049003e96ee8d84f344334498d1b3a08225fd56c889b5fe8734f7710d', '4.1')
_APPROVAL = ('A role change must be approved by the receiving manager and recorded by HR '
             'before the transfer effective date.')
_RECORD = ('HR records employee identifier, previous and new departments and roles, effective date, '
           'receiving manager, and approval reference.')
_REVIEWED = (
    (_APPROVAL, (
        _APPROVAL,
        'The receiving manager must approve the role change and HR must record it before the transfer effective date.',
        'Before the transfer effective date, the receiving manager must approve the role change and HR must record it.',
    )),
    (_RECORD, (_RECORD,)),
)


def validate_reviewed_actor_claim(text: str, citations: tuple[CitationReference, ...],
                                  package: ContextPackage) -> None:
    """Fail closed for unreviewed wording attached to this critical source rule.

    Accept whole supplied source spans or sequences of explicitly reviewed complete
    sentences, with whitespace normalization only. Each sentence must have its full
    source obligation in the supplied context. No substring/keyword acceptance, no
    pronoun resolution, no automatic edits, and no completeness certification.
    """
    for citation in validate_citations(citations, package):
        identity = (citation.document_id, citation.version, citation.status,
                    citation.content_hash, citation.section_id)
        if identity != _SNAPSHOT:
            continue
        group = next(g for g in package.groups if g.citation == citation)
        # Missing or ambiguous initiation evidence cannot establish actor roles.
        match_quotation(_APPROVAL, group)
        if sum(normalize_whitespace(span.text).count(_APPROVAL) for span in group.spans) != 1:
            raise GroundingError('reviewed actor source is ambiguous')
        normalized = normalize_whitespace(text)
        if any(normalized == normalize_whitespace(span.text) for span in group.spans
               if _APPROVAL in normalize_whitespace(span.text)):
            continue
        allowed = []
        for source, statements in _REVIEWED:
            try:
                match_quotation(source, group)
            except GroundingError:
                continue
            allowed.extend(statements)
        # Entire text must be accounted for. An appended false assertion,
        # negation, actor substitution or ambiguous pronoun cannot be hidden
        # behind one correct sentence or a valid supporting quotation.
        remaining = normalized
        while remaining:
            sentence = next((s for s in sorted(allowed, key=len, reverse=True)
                             if remaining == s or remaining.startswith(s + ' ')), None)
            if sentence is None:
                raise GroundingError('critical actor claim requires reviewed wording')
            remaining = remaining[len(sentence):].lstrip()
