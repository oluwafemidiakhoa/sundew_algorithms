# TestPyPI Deployment Guide

## Prerequisites

1. **Create TestPyPI account**: https://test.pypi.org/account/register/
2. **Generate API token**: Go to Account Settings → API tokens → Create token
3. **Configure twine**:
   ```bash
   # Create ~/.pypirc file with:
   [distutils]
   index-servers =
       testpypi
       pypi

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = <your-testpypi-token>

   [pypi]
   repository = https://upload.pypi.org/legacy/
   username = __token__
   password = <your-pypi-token>
   ```

## Deployment Commands

### 1. Upload to TestPyPI
```bash
# Upload the built package
twine upload --repository testpypi dist/*

# Expected output:
# Uploading distributions to https://test.pypi.org/legacy/
# Uploading sundew_algorithms-0.5.0-py3-none-any.whl
# Uploading sundew_algorithms-0.5.0.tar.gz
```

### 2. Test Installation from TestPyPI
```bash
# Install from TestPyPI (in fresh environment)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sundew-algorithms

# Test the installation
python -c "from sundew import SundewAlgorithm; print('TestPyPI install works!')"
python -c "from sundew.simple_core import SimpleSundewAlgorithm; print('Simple core works!')"
```

### 3. Verify Package Page
Visit: https://test.pypi.org/project/sundew-algorithms/

Check that:
- ✅ Package description displays correctly
- ✅ README renders properly
- ✅ Dependencies are listed correctly
- ✅ Classifiers are accurate
- ✅ Download links work

## Production PyPI Deployment

Once TestPyPI validation is complete:

```bash
# Upload to production PyPI
twine upload dist/*

# Verify installation
pip install sundew-algorithms
```

## Troubleshooting

### Common Issues:

1. **"File already exists"** - Version 0.5.0 already uploaded
   - Solution: Increment version in pyproject.toml to 0.5.1
   - Rebuild: `python -m build`
   - Upload new version

2. **Authentication failed**
   - Check API token is correct
   - Ensure ~/.pypirc is properly configured
   - Try: `twine upload --repository testpypi dist/* --verbose`

3. **Dependency resolution errors**
   - Our dependencies (numpy, pandas) should install from main PyPI
   - Use `--extra-index-url https://pypi.org/simple/` flag

4. **README not rendering**
   - Check that README.md is included in package
   - Verify markdown syntax
   - Check pyproject.toml readme field

## Validation Checklist

After TestPyPI deployment:

- [ ] Package page loads correctly
- [ ] README displays properly
- [ ] Installation works: `pip install --index-url https://test.pypi.org/simple/ sundew-algorithms`
- [ ] Imports work: `from sundew import SundewAlgorithm`
- [ ] CLI works: `sundew --help`
- [ ] Working example runs: `python -c "from sundew.simple_core import SimpleSundewAlgorithm; print('OK')"`

## Next Steps

1. **TestPyPI Success** → Deploy to production PyPI
2. **Issues Found** → Fix and redeploy to TestPyPI
3. **Production Deployed** → Update documentation with `pip install sundew-algorithms`

Your package is ready for the world! 🎉
