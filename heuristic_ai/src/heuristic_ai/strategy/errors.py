"""Failures of One-Ply Expectation computation and its immutable audit boundary."""


class AuditIntegrityError(RuntimeError):
    """Corruption or conflicting evidence must never become an ordinary miss."""


class DecisionExecutionError(RuntimeError):
    """One-Ply Expectation cannot continue safely with the supplied state."""


class OnePlyFallback(Exception):
    """A completed static evaluation retained after an unfinished One-Ply Expectation search."""

    def __init__(self, decision, reason):
        super().__init__(reason)
        self.decision = decision
        self.reason = reason
