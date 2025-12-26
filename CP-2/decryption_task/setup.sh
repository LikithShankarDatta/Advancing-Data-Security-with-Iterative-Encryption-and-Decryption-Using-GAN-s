#!/bin/bash

echo "🚀 Setting up Secure Chat Application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application, run:"
echo "  python3 app.py"
echo ""
echo "Then open your browser to: http://127.0.0.1:5000"
