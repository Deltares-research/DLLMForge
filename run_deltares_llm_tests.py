#!/usr/bin/env python3
"""
Test runner script for Deltares LLMs module.

This script provides examples of how to run the unit tests for the DeltaresOllamaLLM class.
"""

import subprocess
import sys
import os


def run_tests():
    """Run the test suite for Deltares LLMs."""

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Running Deltares LLMs Unit Tests")
    print("=" * 60)

    # Test commands to run
    test_commands = [
        # Run all Deltares LLMs tests
        ["python", "-m", "pytest", "tests/test_deltares_llms.py", "-v"],

        # Run integration tests (most will be skipped unless Ollama is running)
        ["python", "-m", "pytest", "tests/test_deltares_llms_integration.py", "-v"],

        # Run tests with coverage
        [
            "python", "-m", "pytest", "tests/test_deltares_llms.py", "tests/test_deltares_llms_integration.py",
            "--cov=dllmforge.LLMs.Deltares_LLMs", "--cov-report=term-missing"
        ],

        # Run only unit tests (exclude integration tests)
        ["python", "-m", "pytest", "tests/test_deltares_llms.py", "-v", "-m", "not integration"],
    ]

    print("\nAvailable test commands:")
    print("-" * 40)
    for i, cmd in enumerate(test_commands, 1):
        print(f"{i}. {' '.join(cmd)}")

    print(f"\n{len(test_commands) + 1}. Run all above commands")
    print(f"{len(test_commands) + 2}. Exit")

    while True:
        try:
            choice = input(f"\nSelect a command to run (1-{len(test_commands) + 2}): ").strip()

            if choice == str(len(test_commands) + 2):  # Exit
                print("Exiting...")
                return
            elif choice == str(len(test_commands) + 1):  # Run all
                for i, cmd in enumerate(test_commands, 1):
                    print(f"\n{'='*20} Running Command {i} {'='*20}")
                    run_command(cmd, project_root)
                return
            else:
                cmd_index = int(choice) - 1
                if 0 <= cmd_index < len(test_commands):
                    run_command(test_commands[cmd_index], project_root)
                    return
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(test_commands) + 2}")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return


def run_command(cmd, project_root):
    """Run a single test command."""
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 40)

    try:
        # Change to project root directory
        original_cwd = os.getcwd()
        os.chdir(project_root)

        # Run the command
        result = subprocess.run(cmd, capture_output=False, text=True)

        # Restore original directory
        os.chdir(original_cwd)

        if result.returncode == 0:
            print(f"\n✅ Command completed successfully")
        else:
            print(f"\n❌ Command failed with return code {result.returncode}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest pytest-cov")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def check_dependencies():
    """Check if required testing dependencies are installed."""
    required_packages = ['pytest', 'pytest-cov']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print(f"\nInstall with: pip install {' '.join(missing_packages)}")
        return False

    print("✅ All required testing dependencies are installed")
    return True


if __name__ == "__main__":
    print("Deltares LLMs Test Runner")
    print("=" * 30)

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Run tests
    run_tests()
