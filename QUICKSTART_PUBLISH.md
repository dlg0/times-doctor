# Quick Start: Publishing to GitHub

## One-Time Setup

1. **Initialize git** (if not already done)
   ```bash
   cd /Users/gre538/code/times-doctor
   git init
   git add .
   git commit -m "Initial commit: TIMES Doctor v0.2.0"
   ```

2. **Create GitHub repository**
   
   **Option A - Using GitHub CLI:**
   ```bash
   gh repo create dlg0/times-doctor --public --source=. --remote=origin
   git push -u origin main
   ```
   
   **Option B - Manually:**
   - Go to https://github.com/new
   - Repository name: `times-doctor`
   - Make it **Public**
   - Don't initialize with README (we have one)
   - Click "Create repository"
   - Then run:
   ```bash
   git remote add origin https://github.com/dlg0/times-doctor.git
   git branch -M main
   git push -u origin main
   ```

3. **Create first release** (optional but recommended)
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
   
   Or via GitHub UI:
   - Go to https://github.com/dlg0/times-doctor/releases/new
   - Tag: `v0.2.0`
   - Title: `v0.2.0 - Initial Public Release`
   - Description: `First public release of TIMES Doctor`

## Test It Works

From any machine with `uv` installed:

```bash
uvx --from git+https://github.com/dlg0/times-doctor times-doctor --help
```

You should see:
```
Usage: times-doctor [OPTIONS] COMMAND [ARGS]...

Commands:
  diagnose
  scan
```

## Updating Later

When you make changes:

```bash
git add .
git commit -m "Description of changes"
git push

# For new version:
# 1. Update version in pyproject.toml
# 2. Commit and tag
git commit -am "Bump version to 0.3.0"
git tag v0.3.0
git push && git push --tags
```

## Share With Others

Send them this command:
```bash
uvx --from git+https://github.com/dlg0/times-doctor times-doctor diagnose <run_dir> --gams <gams_path> --datacheck
```

Or point them to the README:
https://github.com/dlg0/times-doctor

Done! 🎉
