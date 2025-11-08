# Deployment Guide

## Publishing to GitHub

1. **Create GitHub Repository**
   ```bash
   gh repo create dlg0/times-doctor --public --source=. --remote=origin
   ```

2. **Push Code**
   ```bash
   git add .
   git commit -m "Initial release of TIMES Doctor v0.2.0"
   git push -u origin main
   ```

3. **Create Release (Optional)**
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   gh release create v0.2.0 --title "v0.2.0" --notes "First public release"
   ```

## Using from GitHub

Once published, users can run directly:

```cmd
uvx --from git+https://github.com/dlg0/times-doctor times-doctor --help
```

Or install:
```cmd
uv tool install git+https://github.com/dlg0/times-doctor
```

## Publishing to PyPI (Optional)

If you want to publish to PyPI for easier installation:

1. **Get PyPI API token**
   - Go to https://pypi.org/manage/account/token/
   - Create new token with scope for this project

2. **Build package**
   ```bash
   uv build
   ```

3. **Upload to PyPI**
   ```bash
   uv publish --token <your-pypi-token>
   ```

4. **Users can then install via**
   ```bash
   uvx times-doctor
   # or
   uv tool install times-doctor
   ```

## Version Management

Update version in `pyproject.toml`:
```toml
version = "0.3.0"
```

Then tag and release:
```bash
git commit -am "Bump version to 0.3.0"
git tag v0.3.0
git push && git push --tags
```

## Testing Installation

From any Windows machine with uv installed:

```cmd
REM Test direct run
uvx --from git+https://github.com/dlg0/times-doctor times-doctor --help

REM Test with sample data (if available)
uvx --from git+https://github.com/dlg0/times-doctor times-doctor review "D:\path\to\run"
```

## For Private Repository Access

If you make the repo private, users will need authentication:

```cmd
set GH_TOKEN=<personal-access-token>
uvx --from git+https://github.com/dlg0/times-doctor times-doctor --help
```

Or use SSH:
```cmd
uvx --from git+ssh://git@github.com/dlg0/times-doctor times-doctor --help
```

## Troubleshooting

### Azure DevOps Index Conflicts

If you have CSIRO's Azure DevOps index configured globally, it may interfere. Users can bypass it:

**Temporary (per command):**
```cmd
set UV_EXTRA_INDEX_URL=
uvx --from git+https://github.com/csiro-energy/times-doctor times-doctor --help
```

**Permanent solution - create `.config/uv/uv.toml`:**
```toml
# Don't use extra index for public packages
extra-index-url = []
```

Or configure index only for specific projects that need it.
