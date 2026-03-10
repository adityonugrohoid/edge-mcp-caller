#!/usr/bin/env python3
"""Set up the end-to-end test fixture: directory structure + git repo.

Creates tests/e2e/fixture/ with files, directories, and a git repo with
commits, branches, and staged changes — everything needed for all 14 tools
to execute successfully against real MCP servers.

Usage:
    python tests/e2e/setup_fixture.py          # create fixture
    python tests/e2e/setup_fixture.py --clean   # delete and recreate
"""

import argparse
import shutil
import subprocess
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixture"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command and return stdout."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd or FIXTURE_DIR,
    )
    if result.returncode != 0:
        print(f"  WARN: {' '.join(cmd)} → {result.stderr.strip()}")
    return result.stdout.strip()


def create_directory_structure() -> None:
    """Create the file/directory tree."""
    dirs = [
        "src",
        "src/components",
        "src/utils",
        "tests",
        "config",
        "docs",
        "scripts",
        "data",
        "logs",
        "db",
        "monitoring",
        "assets",
    ]
    for d in dirs:
        (FIXTURE_DIR / d).mkdir(parents=True, exist_ok=True)


def create_files() -> None:
    """Create files with content that supports read, search, and edit operations."""
    files = {
        # --- Source files ---
        "src/main.py": (
            "#!/usr/bin/env python3\n"
            "\"\"\"Main application entry point.\"\"\"\n\n"
            "import os\n"
            "from src.utils.helpers import setup_logging\n\n"
            "DEBUG = True\n"
            "PORT = 3000\n"
            "DATABASE_URL = 'sqlite:///app.db'\n\n"
            "def main():\n"
            "    setup_logging()\n"
            "    print(f'Starting server on port {PORT}')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "src/app.js": (
            "const express = require('express');\n"
            "const app = express();\n\n"
            "const RATE_LIMIT = 100;\n"
            "const BATCH_SIZE = 32;\n\n"
            "app.get('/', (req, res) => {\n"
            "  res.json({ status: 'ok' });\n"
            "});\n\n"
            "module.exports = app;\n"
        ),
        "src/components/dashboard.tsx": (
            "import React from 'react';\n\n"
            "interface DashboardProps {\n"
            "  theme: 'light' | 'dark';\n"
            "}\n\n"
            "export const Dashboard: React.FC<DashboardProps> = ({ theme }) => {\n"
            "  return <div className={`dashboard ${theme}`}>Dashboard</div>;\n"
            "};\n"
        ),
        "src/utils/helpers.py": (
            "\"\"\"Utility helpers.\"\"\"\n\n"
            "import logging\n\n"
            "LOG_LEVEL = 'info'\n\n"
            "def setup_logging():\n"
            "    logging.basicConfig(level=LOG_LEVEL.upper())\n"
        ),
        "src/utils/validators.go": (
            "package utils\n\n"
            "func ValidateEmail(email string) bool {\n"
            "    return len(email) > 0\n"
            "}\n"
        ),
        # --- Tests ---
        "tests/test_main.py": (
            "import pytest\n\n"
            "def test_main_starts():\n"
            "    assert True\n\n"
            "def test_port_default():\n"
            "    from src.main import PORT\n"
            "    assert PORT == 3000\n"
        ),
        "tests/test_utils.py": (
            "import pytest\n\n"
            "def test_setup_logging():\n"
            "    from src.utils.helpers import setup_logging\n"
            "    setup_logging()\n"
        ),
        # --- Config ---
        "config/settings.yaml": (
            "app:\n"
            "  name: edge-mcp-test\n"
            "  debug: true\n"
            "  port: 3000\n"
            "  log_level: info\n\n"
            "database:\n"
            "  driver: sqlite3\n"
            "  host: localhost\n"
            "  pool_size: 5\n\n"
            "cache:\n"
            "  backend: local\n"
            "  ttl: 300\n"
        ),
        "config/.env": (
            "DATABASE_URL=sqlite:///app.db\n"
            "SECRET_KEY=dev-secret-key\n"
            "DEBUG=true\n"
            "API_KEY=test-key-12345\n"
        ),
        "config/prometheus.yml": (
            "global:\n"
            "  scrape_interval: 15s\n\n"
            "scrape_configs:\n"
            "  - job_name: 'app'\n"
            "    static_configs:\n"
            "      - targets: ['localhost:3000']\n"
        ),
        # --- Docs ---
        "docs/README.md": (
            "# Test Project\n\n"
            "A sample project for end-to-end MCP testing.\n\n"
            "## Setup\n\n"
            "```bash\npip install -r requirements.txt\n```\n\n"
            "## Usage\n\n"
            "```bash\npython src/main.py\n```\n"
        ),
        "docs/API.md": (
            "# API Reference\n\n"
            "## Endpoints\n\n"
            "### GET /\n"
            "Returns server status.\n\n"
            "### POST /data\n"
            "Submit new data records.\n"
        ),
        # --- Scripts ---
        "scripts/deploy.sh": (
            "#!/bin/bash\n"
            "echo 'Deploying application...'\n"
            "docker build -t app .\n"
            "docker push app:latest\n"
        ),
        "scripts/cleanup.sh": (
            "#!/bin/bash\n"
            "echo 'Cleaning up temp files...'\n"
            "rm -rf /tmp/app-cache\n"
        ),
        # --- Data ---
        "data/sample.csv": (
            "id,name,value\n"
            "1,alpha,100\n"
            "2,beta,200\n"
            "3,gamma,300\n"
        ),
        "data/schema.sql": (
            "CREATE TABLE users (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    name TEXT NOT NULL,\n"
            "    email TEXT UNIQUE\n"
            ");\n\n"
            "CREATE TABLE logs (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    message TEXT,\n"
            "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
            ");\n"
        ),
        # --- Monitoring ---
        "monitoring/alerts.yaml": (
            "alerts:\n"
            "  - name: high_cpu\n"
            "    threshold: 90\n"
            "    action: notify\n"
            "  - name: disk_full\n"
            "    threshold: 95\n"
            "    action: page\n"
        ),
        # --- Root files ---
        ".env": (
            "APP_ENV=development\n"
            "PORT=3000\n"
            "LOG_LEVEL=debug\n"
        ),
        ".gitignore": (
            "*.pyc\n"
            "__pycache__/\n"
            ".venv/\n"
            "node_modules/\n"
            "*.log\n"
        ),
        "README.md": (
            "# E2E Test Fixture\n\n"
            "This directory is a test fixture for end-to-end MCP testing.\n"
        ),
        "requirements.txt": (
            "flask>=3.0\n"
            "pytest>=8.0\n"
            "pyyaml>=6.0\n"
        ),
        "Dockerfile": (
            "FROM python:3.12-slim\n"
            "WORKDIR /app\n"
            "COPY . .\n"
            "RUN pip install -r requirements.txt\n"
            "CMD [\"python\", \"src/main.py\"]\n"
        ),
    }

    for path, content in files.items():
        filepath = FIXTURE_DIR / path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)


def setup_git_repo() -> None:
    """Initialize a git repo with commits, branches, and staged changes."""
    # Init
    run(["git", "init"])
    run(["git", "config", "user.name", "Test User"])
    run(["git", "config", "user.email", "test@example.com"])

    # Initial commit — add everything
    run(["git", "add", "."])
    run(["git", "commit", "-m", "feat: initial project setup"])

    # Second commit — add docs
    (FIXTURE_DIR / "docs" / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 1.0.0\n- Initial release\n"
    )
    run(["git", "add", "docs/CHANGELOG.md"])
    run(["git", "commit", "-m", "docs: add changelog"])

    # Third commit — update config
    settings = FIXTURE_DIR / "config" / "settings.yaml"
    text = settings.read_text()
    settings.write_text(text.replace("pool_size: 5", "pool_size: 10"))
    run(["git", "add", "config/settings.yaml"])
    run(["git", "commit", "-m", "fix: increase database pool size"])

    # Fourth commit — add test
    (FIXTURE_DIR / "tests" / "test_validators.py").write_text(
        "def test_email_validation():\n    assert True\n"
    )
    run(["git", "add", "tests/test_validators.py"])
    run(["git", "commit", "-m", "test: add email validation tests"])

    # Create develop branch
    run(["git", "branch", "develop"])

    # Create staging branch
    run(["git", "branch", "staging"])

    # Create a feature branch
    run(["git", "checkout", "-b", "feature/redesign"])
    (FIXTURE_DIR / "src" / "components" / "navbar.tsx").write_text(
        "export const Navbar = () => <nav>Navbar</nav>;\n"
    )
    run(["git", "add", "src/components/navbar.tsx"])
    run(["git", "commit", "-m", "feat: add navbar component"])
    run(["git", "checkout", "main"])

    # Stage some changes (for git_diff_staged and git_commit)
    readme = FIXTURE_DIR / "README.md"
    readme.write_text(
        "# E2E Test Fixture\n\n"
        "This directory is a test fixture for end-to-end MCP testing.\n\n"
        "## Updated\n"
        "Added staging changes for testing.\n"
    )
    run(["git", "add", "README.md"])

    # Also leave some unstaged changes (for git_status)
    (FIXTURE_DIR / "src" / "main.py").write_text(
        (FIXTURE_DIR / "src" / "main.py").read_text().replace(
            "PORT = 3000", "PORT = 8080"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up e2e test fixture")
    parser.add_argument("--clean", action="store_true", help="Delete and recreate fixture")
    args = parser.parse_args()

    if args.clean and FIXTURE_DIR.exists():
        print(f"Cleaning {FIXTURE_DIR}...")
        shutil.rmtree(FIXTURE_DIR)

    if FIXTURE_DIR.exists():
        print(f"Fixture already exists at {FIXTURE_DIR}")
        print("Use --clean to recreate.")
        return

    print(f"Creating fixture at {FIXTURE_DIR}...")

    FIXTURE_DIR.mkdir(parents=True)
    create_directory_structure()
    print("  Created directory structure")

    create_files()
    print(f"  Created {len(list(FIXTURE_DIR.rglob('*')))} files and directories")

    setup_git_repo()
    commits = run(["git", "log", "--oneline"]).count("\n") + 1
    branches = run(["git", "branch"]).count("\n") + 1
    print(f"  Initialized git repo: {commits} commits, {branches} branches")

    staged = run(["git", "diff", "--cached", "--name-only"])
    unstaged = run(["git", "diff", "--name-only"])
    print(f"  Staged: {staged}")
    print(f"  Unstaged: {unstaged}")

    print("\nFixture ready. Run: python tests/e2e/run_e2e.py")


if __name__ == "__main__":
    main()
