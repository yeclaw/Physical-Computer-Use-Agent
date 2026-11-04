"""Bounded credential redaction for diagnostic text, not observation evidence.

This is a best-effort log boundary. Arbitrary camera/state/program data and
unlabelled secrets still require private storage and review before publication.
"""
import re


def sensitive_key(key):
    key = re.sub(r'[^a-z0-9]', '', str(key).casefold())
    return (key in {'auth', 'authorization', 'proxyauthorization', 'credentials',
                    'credential', 'cookie', 'cookies', 'setcookie', 'privatekey',
                    'sessionid', 'account', 'accountid', 'userid', 'organizationid',
                    'email', 'accesskey', 'accesskeyid', 'secretkey'}
            or key.endswith(('token', 'secret', 'password', 'passwd', 'apikey',
                             'privatekey', 'secretkey', 'accesskey', 'accesskeyid',
                             'accountid', 'userid', 'organizationid')))


_KEY = (r'(?:[\w.-]*(?:token|secret|password|passwd|credential|credentials)|'
        r'[\w.-]*(?:api|private|secret|access)[-_]?key(?:[-_]?id)?|'
        r'authorization|proxy[-_]?authorization|cookies?|set[-_]?cookie)')
_VALUE = r'(?:\[REDACTED(?:_ID|_EMAIL)?\]|"[^"\r\n]*"|\'[^\'\r\n]*\'|[^\s,;}\])]+)'
_ASSIGNMENT = re.compile(r'(?i)(\b' + _KEY + r'["\']*\s*[:=]\s*)' + _VALUE)


def redact_text(value, limit=2048):
    """Return bounded text with common credential/header/URL forms removed."""
    if not isinstance(value, str):
        raise TypeError('Diagnostic redaction requires text')
    value = re.sub(r'(?i)(\b(?:proxy[-_]?authorization|authorization)["\']*\s*[:=]\s*)'
                   r'["\']?(?:basic|bearer)\s+[^\s,;"\')\]}]+',
                   r'\1[REDACTED]', value)
    value = re.sub(r'(?i)\b(bearer\s+)[^\s,;"\')\]}]+', r'\1[REDACTED]', value)
    value = re.sub(r'(?im)(\b(?:set-cookie|cookies?)\s*:\s*)[^\r\n]*', r'\1[REDACTED]', value)
    value = _ASSIGNMENT.sub(r'\1[REDACTED]', value)
    value = re.sub(r'\b(?:sk|sess|rk)-[A-Za-z0-9_-]+', '[REDACTED]', value)
    value = re.sub(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '[REDACTED]', value)
    value = re.sub(r'\b(?:acct|account|org|user)_[A-Za-z0-9_-]+', '[REDACTED_ID]', value)
    value = re.sub(r'(?i)([a-z][a-z0-9+.-]*://)[^/\s@]+@', r'\1[REDACTED]@', value)
    value = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[REDACTED_EMAIL]', value)
    value = re.sub(r'(?i)([?&](?:[\w.-]*(?:token|secret|password|signature|credential)|'
                   r'key|api_key|x-amz-signature|x-amz-credential)=)[^&#\s"\']+',
                   r'\1[REDACTED]', value)
    return value[:limit]
