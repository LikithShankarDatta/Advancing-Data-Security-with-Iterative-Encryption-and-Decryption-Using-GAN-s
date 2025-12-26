import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from decryption_backend import simple_caesar_encrypt, simple_caesar_decrypt

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_demo'

# Simple in-memory storage backed by file
MESSAGES = []
USERS_FILE = 'users.json'
MESSAGES_FILE = 'messages.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        try:
            return json.load(f)
        except:
            return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_messages():
    if not os.path.exists(MESSAGES_FILE):
        return []
    with open(MESSAGES_FILE, 'r') as f:
        try:
            return json.load(f)
        except:
            return []

def save_messages():
    with open(MESSAGES_FILE, 'w') as f:
        json.dump(MESSAGES, f, indent=2)

# Load messages on startup
MESSAGES = load_messages()

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        action = request.form.get('action')

        users = load_users()

        if action == 'signup':
            if username in users:
                return render_template('login.html', error="User already exists")
            users[username] = password
            save_users(users)
            session['user'] = username
            return redirect(url_for('chat'))
        
        elif action == 'signin':
            if username in users and users[username] == password:
                session['user'] = username
                return redirect(url_for('chat'))
            else:
                return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', username=session['user'])

@app.route('/api/messages', methods=['GET'])
def get_messages():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(MESSAGES)

@app.route('/api/send', methods=['POST'])
def send_message():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'Empty message'}), 400

    # Encrypt message
    encrypted_text = simple_caesar_encrypt(text)
    
    message = {
        'user': session['user'],
        'text': encrypted_text,
        'type': 'chat'
    }
    MESSAGES.append(message)
    save_messages()
    return jsonify({'status': 'success'})

@app.route('/api/decrypt', methods=['POST'])
def decrypt_message():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    decrypted_text = simple_caesar_decrypt(text)
    return jsonify({'decrypted': decrypted_text})

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global MESSAGES
    MESSAGES = []
    save_messages()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
