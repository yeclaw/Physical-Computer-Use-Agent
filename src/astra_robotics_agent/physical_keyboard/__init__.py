"""Task-specific physical keyboard software; importing this package opens no hardware."""

__all__ = ['EvaluationPlan', 'Scope', 'Usage', 'Ledger', 'LedgerError',
           'IntegrityError', 'StaleParentError']


def __getattr__(name):
    # The reset-only GTK editor also runs on the deployment's system Python.
    # Do not import the newer-runtime ledger/contracts for this independent
    # stdlib service. Preserve the public exports when they are requested.
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module
    module = '.records.contracts' if name in ('EvaluationPlan', 'Scope', 'Usage') else '.records.ledger'
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
