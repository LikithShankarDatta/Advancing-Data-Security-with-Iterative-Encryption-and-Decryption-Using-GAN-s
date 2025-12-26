# Secure Chat Application

A Flask-based encrypted chat application using Caesar cipher encryption.

## Features
- 🔐 End-to-end Caesar cipher encryption
- 👥 Multi-user support with authentication
- 💬 Real-time messaging
- 🔓 Message decryption on demand
- 💾 Persistent message storage
- 🎨 Modern, clean UI

## Requirements
- Python 3.9 or higher
- pip (Python package manager)

## Installation & Setup

### 1. Extract the Project
Unzip the project folder to your desired location.

### 2. Install Dependencies
Open a terminal in the project folder and run:
```bash
pip install flask
```

### 3. Run the Application
```bash
python3 app.py
```

### 4. Access the App
Open your browser and go to:
```
http://127.0.0.1:5000
```

## Usage

1. **Sign Up**: Create a new account with a username and password
2. **Login**: Sign in with your credentials
3. **Send Messages**: Type and send encrypted messages
4. **Decrypt Messages**: Click the 🔓 icon next to any message to decrypt it temporarily
5. **Clear Chat**: Click the "🗑️ Clear" button to delete all messages

## Project Structure
```
decryption_task/
├── app.py                    # Main Flask application
├── decryption_backend.py     # Encryption/decryption logic
├── templates/
│   ├── base.html            # Base template
│   ├── login.html           # Login/signup page
│   └── chat.html            # Chat interface
├── users.json               # User database (auto-created)
├── messages.json            # Message storage (auto-created)
└── README.md               # This file
```

## How It Works

### Encryption
Messages are encrypted using a Caesar cipher with a random key (1-26). The key is embedded in the encrypted text, allowing for decryption without sharing the key separately.

### Storage
- **Users**: Stored in `users.json`
- **Messages**: Stored in `messages.json`
- Both files are created automatically on first run

## Troubleshooting

### Port 5000 Already in Use
If you see "Address already in use", macOS AirPlay might be using port 5000. 

**Option 1**: Disable AirPlay Receiver
- Go to System Preferences → General → AirDrop & Handoff
- Turn off "AirPlay Receiver"

**Option 2**: Use a different port
Edit `app.py` line 140:
```python
app.run(debug=True, port=8080)  # Change 5000 to 8080
```
Then access at `http://127.0.0.1:8080`

### Missing Dependencies
If you get import errors, install the required package:
```bash
pip install flask
```

## Security Note
⚠️ This is a demonstration project using a simple Caesar cipher. It is **NOT** suitable for real secure communications. For production use, implement proper encryption (e.g., AES, RSA).

## Credits
Built with Flask and Caesar cipher encryption for educational purposes.
