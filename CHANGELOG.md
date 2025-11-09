# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI with matrix testing across Python 3.9-3.12 and Linux/Windows
- Type safety with mypy and Protocol interfaces for LLM and GAMS providers
- Exception hierarchy for better error handling
- Log redaction for API keys and sensitive data
- Pre-commit hooks with ruff and mypy
- Cross-platform I/O utilities for Windows robustness
- Testing framework with mocked LLM providers and GAMS subprocess
- py.typed marker for typed package distribution

### Changed
- Consolidated dev dependencies in pyproject.toml
- Added ruff and mypy configuration
- Updated package classifiers to include OS support and typing

### Security
- API keys now redacted in all log files

## [0.6.4] - 2025-11-07

### Added
- TIMES source auto-download with version detection
- Improved GAMS re-run mechanism for datacheck

### Fixed
- GAMS idir parameters for proper TIMES file resolution

## [0.6.0] - 2025-11-06

### Added
- `review` command for LLM-powered run analysis
- `datacheck` command for CPLEX diagnostics
- `scan` command for testing solver configurations
- Multi-run progress monitoring

### Changed
- Improved log file parsing and condensing

## [0.5.0] - 2025-10-15

### Added
- Initial release
- Basic QA_CHECK.LOG parsing
- LST file extraction

[Unreleased]: https://github.com/dlg0/times-doctor/compare/v0.6.4...HEAD
[0.6.4]: https://github.com/dlg0/times-doctor/compare/v0.6.0...v0.6.4
[0.6.0]: https://github.com/dlg0/times-doctor/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/dlg0/times-doctor/releases/tag/v0.5.0
