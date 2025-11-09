# Publishing a New Release

1. **Update version** in `pyproject.toml`

2. **Commit version bump** (this will trigger pre-commit hooks)
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   ```

3. **Fix any linting/formatting issues** from pre-commit hooks
   ```bash
   # If pre-commit hooks modified files, commit the fixes
   git add -A
   git commit -m "Fix linting/formatting"
   ```

4. **Tag the release** (ensure tag points to the final clean commit)
   ```bash
   git tag vX.Y.Z
   ```

5. **Push everything**
   ```bash
   git push origin main
   git push origin vX.Y.Z
   ```

The tag push will trigger the GitHub Actions workflow to build and publish to PyPI.
