#!/bin/bash
set -e

echo "🚀 Setting up Quaid development environment..."

# Check if mise is installed
if ! command -v mise &> /dev/null; then
  echo "📦 Installing mise..."

  # Detect OS
  OS=$(uname -s)

  case "$OS" in
  "Darwin")
    # macOS - try homebrew
    if command -v brew &> /dev/null; then
      echo "Using homebrew to install mise..."
      brew install mise
    else
      echo "Using curl installer..."
      curl https://mise.run | sh
      export PATH="$HOME/.local/share/mise/bin:$PATH"
    fi
    ;;
  "Linux")
    # Linux - detect distribution and try package managers
    if [ -f /etc/os-release ]; then
      . /etc/os-release
      case "$ID" in
      "ubuntu" | "debian")
        if command -v apt &> /dev/null; then
          echo "Using apt to install mise..."
          sudo apt update && sudo apt install -y mise
        else
          echo "Using curl installer..."
          curl https://mise.run | sh
          export PATH="$HOME/.local/share/mise/bin:$PATH"
        fi
        ;;
      "fedora" | "rhel" | "centos")
        if command -v dnf &> /dev/null; then
          echo "Using dnf to install mise..."
          sudo dnf install -y mise
        elif command -v yum &> /dev/null; then
          echo "Using yum to install mise..."
          sudo yum install -y mise
        else
          echo "Using curl installer..."
          curl https://mise.run | sh
          export PATH="$HOME/.local/share/mise/bin:$PATH"
        fi
        ;;
      "arch")
        if command -v pacman &> /dev/null; then
          echo "Using pacman to install mise..."
          sudo pacman -S mise
        else
          echo "Using curl installer..."
          curl https://mise.run | sh
          export PATH="$HOME/.local/share/mise/bin:$PATH"
        fi
        ;;
      *)
        echo "Using curl installer..."
        curl https://mise.run | sh
        export PATH="$HOME/.local/share/mise/bin:$PATH"
        ;;
      esac
    else
      echo "Using curl installer..."
      curl https://mise.run | sh
      export PATH="$HOME/.local/share/mise/bin:$PATH"
    fi
    ;;
  *)
    echo "Using curl installer..."
    curl https://mise.run | sh
    export PATH="$HOME/.local/share/mise/bin:$PATH"
    ;;
  esac

  # Add to shell profiles if using curl installer
  if [[ "$PATH" == *".local/share/mise/bin"* ]]; then
    echo 'export PATH="$HOME/.local/share/mise/bin:$PATH"' >> ~/.bashrc 2> /dev/null || true
    echo 'export PATH="$HOME/.local/share/mise/bin:$PATH"' >> ~/.zshrc 2> /dev/null || true
  fi
else
  echo "✅ mise is already installed"
fi

# Trust the project configuration
echo "🔐 Trusting project configuration..."
mise trust

# Install development tools
echo "🔧 Installing development tools..."
mise install

# Install workspace dependencies
echo "📚 Installing workspace dependencies with uv..."
uv sync

# Install git hooks
echo "🪝 Installing git hooks..."
mise exec lefthook -- lefthook install

echo "🎉 Setup complete! You can now run:"
echo "  uv run quaid --help    # Show CLI help"
echo "  uv run pytest         # Run tests"
echo "  uv run ruff check     # Run linter"
echo "  uv run ruff format    # Run formatter"
