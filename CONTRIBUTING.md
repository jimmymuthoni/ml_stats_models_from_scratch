#### Contributing to ML Stats Models from Scratch

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing to the repository.

#### Repository Information

- Remote Repository: [https://github.com/jimmymuthoni/ml_stats_models_from_scratch.git](https://github.com/jimmymuthoni/ml_stats_models_from_scratch.git)
- Project Goal: Implement Machine Learning algorithms from scratch to deeply understand their internal mechanics


#### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git

#### Setting Up the Development Environment

1. Fork the repository on GitHub

2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ml_stats_models_from_scratch.git
   cd ml_stats_models_from_scratch
   ```

3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/jimmymuthoni/ml_stats_models_from_scratch.git
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Verify installation:
   ```bash
   python -c "import numpy, pandas; print('Dependencies installed successfully!')"
   ```

#### Contribution Workflow

#### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions**:
- `feature/` - for new algorithms or features
- `fix/` - for bug fixes
- `docs/` - for documentation updates
- `refactor/` - for code refactoring
- `test/` - for adding tests

#### 2. Make Your Changes

- Write clean, readable code: Python or Julia
- Follow the existing code style
- Add comments explaining complex mathematical operations
- Ensure your code follows the project's philosophy of implementing algorithms from scratch

#### 3. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add: Implement logistic regression from scratch"
```

Commit message format:
- `Add:` - for new features/algorithms
- `Fix:` - for bug fixes
- `Update:` - for updates to existing code
- `Docs:` - for documentation changes
- `Refactor:` - for code refactoring

#### 4. Keep Your Branch Updated

Before pushing, make sure your branch is up to date with the main branch:

```bash
git fetch upstream
git rebase upstream/main
```

#### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- A clear title describing your changes
- A detailed description of what you implemented
- Any relevant mathematical background or references
- Example usage or test results

#### Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to classes and functions
- Keep functions focused on a single task
- Use 4 spaces for indentation (no tabs)

#### Code Structure

- Place new algorithms in appropriate directories:
  - `supervised_learning/` - for supervised learning algorithms
  - `clustering/` - for clustering algorithms
  - `dimensionality_reduction/` - for dimensionality reduction techniques
  - `tree_models/` - for tree-based models
  - `neighbors/` - for k-nearest neighbors and related algorithms
  - `utils/` - for utility functions and metrics

#### Algorithm Implementation Guidelines

1. Implement from scratch: Use only NumPy, pandas, and standard library
2. Mathematical clarity: Include comments explaining the mathematical operations
3. Class-based design: Implement algorithms as classes with:
   - `__init__()` - for initialization and hyperparameters
   - `fit()` - for training the model
   - `predict()` - for making predictions
   - `score()` or evaluation methods - for model evaluation

4. Example structure:
   ```python
   class YourAlgorithm:
       def __init__(self, param1=default, param2=default):
           """Initialize the algorithm with hyperparameters."""
           self.param1 = param1
           self.param2 = param2
           # Initialize model parameters
       
       def fit(self, X, y):
           """Train the model on training data."""
           # Implementation here
           pass
       
       def predict(self, X):
           """Make predictions on new data."""
           # Implementation here
           pass
   ```

#### Documentation

- Add docstrings to all classes and public methods
- Include mathematical formulas in comments when relevant
- Update README.md if adding new major features
- Document any assumptions or limitations of your implementation

#### Testing

While formal tests are not required yet, please:

- Test your implementation on sample datasets
- Verify that your algorithm produces reasonable results
- Include example usage in your Pull Request description
- Test edge cases (empty arrays, single samples, etc.)

#### What to Contribute

We welcome contributions in the following areas:

1. New Algorithms: Implement ML algorithms from scratch
2. Bug Fixes: Fix issues in existing implementations
3. Performance Improvements: Optimize existing code
4. Documentation: Improve code comments and README
5. Utility Functions: Add helpful utility functions or metrics
6. Examples: Add usage examples or notebooks

#### Review Process

- All Pull Requests will be reviewed
- Be open to feedback and suggestions
- Address review comments promptly
- Be patient - maintainers are volunteers

#### Questions?

If you have questions or need help:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the codebase for examples

#### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the project's educational goals

Thank you for contributing to this educational project! Your efforts help others understand machine learning algorithms at a deeper level.
