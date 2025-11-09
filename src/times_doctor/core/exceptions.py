"""Exception hierarchy for times-doctor."""


class TimesDoctorError(Exception):
    """Base exception for all times-doctor errors."""

    pass


class ConfigError(TimesDoctorError):
    """Configuration or environment error."""

    pass


class LlmError(TimesDoctorError):
    """LLM API or provider error."""

    pass


class GamsNotFoundError(TimesDoctorError):
    """GAMS executable not found or inaccessible."""

    pass


class RunNotFoundError(TimesDoctorError):
    """TIMES run directory or required files not found."""

    pass


class UserAbort(TimesDoctorError):
    """User aborted the operation."""

    pass
