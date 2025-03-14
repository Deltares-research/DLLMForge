# DLLMForge
A repository for LLM tools in Deltares




Below is the updated Markdown version for your README. It now includes a section on using docstrings with type annotations for functions and specifies that we are using GitHub for hosting our repositories.

```markdown
# Software Development Guide

This guide is intended to help you when developing a new feature. It walks you through the process of creating a new branch, writing unit tests with pytest, adding docstrings with type annotations to your functions, and creating a pull request on GitHub.

---

## 1. Creating a New Branch

### Why Create a Branch?
Creating a dedicated branch for a feature allows you to isolate your changes from the main codebase. This helps keep the project organized and simplifies code reviews and integration.

### Steps to Create a Branch:
1. **Update Your Local Repository**  
   Make sure your local `main` (or `master`) branch is up to date:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create a New Branch**  
   Use a clear and consistent naming convention. For example:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   Replace `your-feature-name` with a concise description of your feature.

3. **Confirm Branch Creation**  
   Verify that you’re on the new branch:
   ```bash
   git branch
   ```

---

## 2. Writing Unit Tests with pytest

### Why Write Unit Tests?
Unit tests help ensure that individual parts of your code work as expected. They make it easier to catch bugs early and provide documentation on how functions should behave.

### Setting Up pytest:
1. **Install pytest (if not already installed):**
   ```bash
   pip install pytest
   ```

2. **Organize Your Tests**  
   - Place your tests in a dedicated directory (commonly named `tests`).
   - Test files should be named starting with `test_` (e.g., `test_my_feature.py`).

### Writing a Sample Test:
Here’s a basic example of a unit test for a function that adds two numbers:

```python
# In your feature file (e.g., my_feature.py)
def add(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: The sum of a and b.
    """
    return a + b
```

```python
# In your test file (e.g., tests/test_my_feature.py)
import pytest
from my_feature import add

class TestMyFeature:
    def test_add():
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
```

### Running Tests:
- To run all tests, execute:
  ```bash
  pytest
  ```

### Best Practices for Tests:
- Write tests that are isolated and independent.
- Aim for clear, concise test cases.
- Use fixtures for setup and teardown if needed.
- Name your tests descriptively so that failures are easy to interpret.

---

## 3. Writing Docstrings with Type Annotations

### Why Add Docstrings with Typing?
Adding docstrings to your functions helps document what your code does, making it easier for team members to understand and maintain the codebase. Including type annotations ensures clarity on the expected data types for function inputs and outputs.

### Guidelines:
- **Write a Clear Docstring:** Include a description, parameters, and return value.
- **Use Type Annotations:** Clearly specify the type for each parameter and the return type.
- **Follow Consistent Formatting:** Use a standard format (like Google style or NumPy style) across the project.


```python
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The product of a and b.
    """
    return a * b
```

---

## 4. Creating a Pull Request on GitHub

### Why Create a Pull Request?
Pull requests (PRs) allow team members to review your code changes before merging them into the main branch. They provide an opportunity for feedback, quality checks, and discussion.

### Steps to Create a Pull Request:
1. **Push Your Branch to GitHub:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request on GitHub:**
   - Navigate to your repository on [GitHub](https://github.com).
   - You will often see a prompt to create a pull request once your branch is pushed.
   - Click the "New Pull Request" button.
   - Ensure that the base branch (often `main` or `develop`) is correct.

3. **Fill in the PR Details:**
   - **Title:** Provide a concise summary of your changes.
   - **Description:** Explain the purpose of the feature, reference any related issues, and include instructions or notes for testing.
   - **Reviewers:** Add team members to review the PR.
   - **Labels/Milestones:** If your team uses labels or milestones, add them accordingly.

4. **Submit the Pull Request:**  
   Once all details are in place, submit your PR for review.

---

## Best Practices

- **Branch Naming:** Use clear, descriptive names (e.g., `feature/login-authentication` or `bugfix/correct-calculation`).
- **Commit Often:** Make small, logical commits with clear messages.
- **Test Early & Often:** Run your tests frequently during development to catch issues early.
- **Code Reviews:** Engage in constructive discussions during the code review process.
- **Documentation:** Update documentation if your feature includes changes to the API or behavior.
- **Docstrings with Typing:** Ensure every function includes a docstring with type annotations.
