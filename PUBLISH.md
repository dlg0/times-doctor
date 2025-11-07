# Publishing a New Release

1. **Update version** in `pyproject.toml`

2. **Commit and push**
   ```bash
   git add .
   git commit -m "Bump version to X.Y.Z"
   git push
   ```

3. **Tag the release**
   ```bash
   git tag vX.Y.Z
   git push --tags
   ```
