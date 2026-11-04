"""Pure token-usage normalization for native and historical evidence.

Counts are measurements, not billing estimates. Missing counts stay ``None``;
cached input and reasoning output are subsets of input and output respectively.
Unknown provider fields are ignored, so this view never rewrites raw evidence.
"""


USAGE_FIELDS = ('input_tokens', 'cached_input_tokens', 'output_tokens',
                'reasoning_output_tokens', 'total_tokens')
_ALIASES = {
    'input_tokens': ('input_tokens', 'inputTokens'),
    'cached_input_tokens': ('cached_input_tokens', 'cachedInputTokens'),
    'output_tokens': ('output_tokens', 'outputTokens'),
    'reasoning_output_tokens': ('reasoning_output_tokens', 'reasoningOutputTokens'),
    'total_tokens': ('total_tokens', 'totalTokens'),
}
_DETAILS = (
    ('cached_input_tokens', ('input_tokens_details', 'inputTokensDetails'),
     ('cached_tokens', 'cachedTokens')),
    ('reasoning_output_tokens', ('output_tokens_details', 'outputTokensDetails'),
     ('reasoning_tokens', 'reasoningTokens')),
)


class UsageNormalizationError(ValueError):
    """Recognized usage evidence has invalid or contradictory measurements."""


def usage_delta(before, after):
    """Recorded cumulative-count difference; missing/reset values stay unknown.

    This does not establish billing completeness. Callers separately record the
    provider's before/after completeness flags, including failed operations.
    """
    previous, current = normalize_usage(before), normalize_usage(after)
    return {field: (current[field] - previous[field] if previous[field] is not None
            and current[field] is not None and current[field] >= previous[field] else None)
            for field in USAGE_FIELDS}


def _count(value, field):
    if value is not None and (type(value) is not int or value < 0):
        raise UsageNormalizationError(field + ' must be a nonnegative integer or null')
    return value


def _merge(values, field):
    known = [value for value in values if value is not None]
    if known and any(value != known[0] for value in known[1:]):
        raise UsageNormalizationError('Conflicting usage aliases for ' + field)
    return known[0] if known else None


def normalize_usage(value):
    """Return five nullable snake-case counts, rejecting inconsistent evidence.

    Snake/camel flat aliases and nested cached/reasoning details may coexist
    only when their non-null measurements agree. An explicit null is unknown,
    so another measured alias may supply that count. No absent count is inferred
    from another count, a turn status, cancellation, or a provider default.

    A supplied total must contain each known parent; when both input and output
    are known their sum must equal that total. Subsets are never added again.
    The supplied dictionary and any nested details remain unchanged.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise UsageNormalizationError('Usage must be a dict or null')
    candidates = {field: [_count(value[alias], alias)
                          for alias in _ALIASES[field] if alias in value]
                  for field in USAGE_FIELDS}
    for field, containers, aliases in _DETAILS:
        for container in containers:
            nested = value.get(container)
            if nested is None:
                continue
            if not isinstance(nested, dict):
                raise UsageNormalizationError(container + ' must be a dict or null')
            candidates[field].extend(_count(nested[alias], container + '.' + alias)
                                     for alias in aliases if alias in nested)
    result = {field: _merge(candidates[field], field) for field in USAGE_FIELDS}
    for child, parent in (('cached_input_tokens', 'input_tokens'),
                          ('reasoning_output_tokens', 'output_tokens'),
                          ('input_tokens', 'total_tokens'),
                          ('output_tokens', 'total_tokens'),
                          ('cached_input_tokens', 'total_tokens'),
                          ('reasoning_output_tokens', 'total_tokens')):
        if result[child] is not None and result[parent] is not None and result[child] > result[parent]:
            raise UsageNormalizationError(child + ' exceeds ' + parent)
    if all(result[field] is not None for field in ('input_tokens', 'output_tokens', 'total_tokens')):
        if result['input_tokens'] + result['output_tokens'] != result['total_tokens']:
            raise UsageNormalizationError('total_tokens disagrees with input_tokens plus output_tokens')
    return result
