"""Errors raised when native objects cannot satisfy a shared interface."""


class PortabilityError(RuntimeError):
    """A selected engine object cannot provide the requested shared behavior."""
