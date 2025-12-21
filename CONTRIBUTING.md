# 🤝 Contributing to Antigravity

Thank you for your interest in contributing to Antigravity! This document provides guidelines for contributing.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/antigravity.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Test locally with `dfx start` and `dfx deploy`
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature`
8. Open a Pull Request

## 📋 Development Setup

### Prerequisites

- WSL (Windows) or Linux/macOS
- Python 3.10+
- dfx (ICP SDK)
- Kybra

### Local Development

```bash
# Install dependencies
pip install kybra
python -m kybra install-dfx-extension

# Start local replica
dfx start --clean --background

# Deploy and test
dfx deploy
dfx canister call antigravity_bot load_sample_data
```

## 🧪 Testing

Before submitting:
1. Run `dfx canister call antigravity_bot get_health`
2. Run `dfx canister call antigravity_bot load_sample_data`
3. Verify 45 pairs are analyzed

## 📝 Code Style

- Use descriptive variable names
- Add docstrings to functions
- Keep functions focused and small
- Comment complex logic

## 🐛 Reporting Issues

Please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- dfx version and OS

## 💡 Feature Requests

Open an issue with:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (optional)

---

Thank you for contributing! 🌌
