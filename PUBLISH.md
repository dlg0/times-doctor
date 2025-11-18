# Publishing a New Release

0. **Ensure the tests pass**
   ```bash
   uv run pytest
   ```

1. **Update version** in `pyproject.toml`

2. **Run pre-commit on all files** (catches issues CI would find)
   ```bash
   uv run pre-commit run --all-files
   ```

3. **Commit version bump and any fixes**
   ```bash
   git add -A
   git commit -m "Bump version to X.Y.Z"
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

The tag push will trigger the GitHub Actions workflow to build the package and create a GitHub Release with distribution files attached.

Users can install directly from GitHub releases using:
```bash
pip install https://github.com/dlg0/times-doctor/releases/download/vX.Y.Z/times_doctor-X.Y.Z-py3-none-any.whl
```

Or install from the repository:
```bash
pip install git+https://github.com/dlg0/times-doctor.git@vX.Y.Z
```
