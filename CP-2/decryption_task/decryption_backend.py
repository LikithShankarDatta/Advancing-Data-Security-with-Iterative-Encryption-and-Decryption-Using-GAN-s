import os
import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow import keras
import tensorflow as tf

def load_data(path):
    """Load data from file - CORRECTED VERSION"""
    file_path = path
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data.split('\n')
def tokenize(x):
    """Tokenize text data"""
    x_tk = Tokenizer(char_level=False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    """Pad sequences to same length"""
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post', truncating='post')


def preprocess(x, y):
    """Preprocess input and output sequences"""
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def texts_to_token(text, tokenizer):
    """Convert text to tokens"""
    tokens = tokenizer.texts_to_sequences([text])
    return tokens[0] if tokens else []


def logits_to_text(logits, tokenizer):
    """Convert model output logits back to text"""
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def caesar_cipher_encrypt(text, shift):
    """Encrypt text using Caesar cipher"""
    encrypted = ''
    for char in text:
        if char.isalpha():
            ascii_offset = 97 if char.islower() else 65
            encrypted += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
        else:
            encrypted += char
    return encrypted


def caesar_cipher_decrypt(text, shift):
    """Decrypt Caesar cipher"""
    return caesar_cipher_encrypt(text, -shift)  # Decryption is just reverse shift


def map_last_letter_to_key(last_letter):
    """Map last letter of encrypted word to cipher key"""
    if last_letter.isalpha():
        return ord(last_letter.lower()) - ord('a') + 1
    return 0


def find_last_letter(word):
    """Extract the last letter (if it's an alphabet letter)"""
    last_letter = ""
    if word and word[-1].isalpha():
        last_letter = word[-1]
    return last_letter


def remove_last_letter(word):
    """Remove the last letter from the word"""
    return word[:-1] if word else word

def generate_dataset(num_sentences=1000):
    """Generate training dataset with Caesar cipher"""
    import random
    import nltk
    from nltk.corpus import words

    try:
        word_list = words.words()
    except:
        nltk.download('words')
        word_list = words.words()

    print(f"Generating {num_sentences} sentences...")

    # Generate plaintext sentences
    plaintexts = []
    for i in range(num_sentences):
        num_words = random.randint(3, 8)
        sentence = ' '.join(random.sample(word_list, num_words)).lower()
        plaintexts.append(sentence)

    # Save plaintext
    with open('extended_dataset.txt', 'w') as f:
        f.write('\n'.join(plaintexts))

    # Generate ciphertext for each key (1-26)
    for key in range(1, 27):
        ciphertexts = []
        for plaintext in plaintexts:
            # Encrypt and append key indicator
            encrypted = caesar_cipher_encrypt(plaintext, key)
            # Add key indicator as last character
            key_char = chr(ord('a') + key - 1)
            encrypted_with_key = encrypted + key_char
            ciphertexts.append(encrypted_with_key)

        # Save ciphertext
        with open(f'ciphertext_key{key}.txt', 'w') as f:
            f.write('\n'.join(ciphertexts))

        print(f"Key {key} dataset created")

    print("Dataset generation complete!")
    return plaintexts

def create_model(input_vocab_size, output_vocab_size, input_length, output_length):
    """Create LSTM model for decryption"""
    model = Sequential([
        Embedding(input_vocab_size, 128, input_length=input_length),
        Bidirectional(LSTM(256, return_sequences=True)),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dense(output_vocab_size, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def train_models_for_all_keys(epochs=50):
    """Train one model for each Caesar cipher key"""

    print("Loading dataset...")
    plaintext = load_data("extended_dataset.txt")

    for key in range(1, 27):
        print(f"\n{'='*70}")
        print(f"Training model for key {key}...")
        print(f"{'='*70}")

        # Load ciphertext for this key
        code = load_data(f'ciphertext_key{key}.txt')

        # Preprocess
        preproc_code_sentence, preproc_plaintext_sen, code_token, plaintext_token = preprocess(code, plaintext)

        # Create model
        model = create_model(
            input_vocab_size=len(code_token.word_index) + 1,
            output_vocab_size=len(plaintext_token.word_index) + 1,
            input_length=preproc_code_sentence.shape[1],
            output_length=preproc_plaintext_sen.shape[1]
        )

        # Train
        model.fit(
            preproc_code_sentence,
            preproc_plaintext_sen,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Save model in .keras format (Keras 3 compatible)
        model.save(f'model_{key}.keras')
        print(f"Model {key} saved!")

def load_all_models():
    """Load all 26 trained models"""
    loaded_models = []
    print("Loading models...")

    for key in range(1, 27):
        model_path = f'model_{key}.keras'
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            loaded_models.append(model)
            print(f"Model {key} loaded", end=" ")
        else:
            print(f"\nWarning: Model {key} not found at {model_path}")
            loaded_models.append(None)

    print("\nAll models loaded!")
    return loaded_models


def setup_decryption_system():
    """Setup complete decryption system"""
    print("Setting up decryption system...")

    # Load models
    loaded_models = load_all_models()

    # Load and preprocess all tokenizers
    print("\nPreparing tokenizers...")
    code_tokenizers = []
    plaintext_tokens = []
    preproc_plaintext_sentences = []

    plaintext = load_data("extended_dataset.txt")

    for key in range(1, 27):
        code = load_data(f'ciphertext_key{key}.txt')
        preproc_code_sentence, preproc_plaintext_sen, code_token, plaintext_token = preprocess(code, plaintext)

        code_tokenizers.append(code_token)
        plaintext_tokens.append(plaintext_token)
        preproc_plaintext_sentences.append(preproc_plaintext_sen)
        print(f"Tokenizer {key} ready")

    print("\nDecryption system ready!")

    return {
        'models': loaded_models,
        'code_tokenizers': code_tokenizers,
        'plaintext_tokens': plaintext_tokens,
        'preproc_plaintext_sentences': preproc_plaintext_sentences
    }

def decipher_sentence(ciphered_sentence, decryption_system):
    """
    Decipher an encrypted sentence using trained models

    Args:
        ciphered_sentence: The encrypted sentence
        decryption_system: Dictionary with models, tokenizers, etc.

    Returns:
        Deciphered plaintext
    """
    ciphered_sentence = str(ciphered_sentence).strip()

    # Extract components
    loaded_models = decryption_system['models']
    code_tokenizers = decryption_system['code_tokenizers']
    plaintext_tokens = decryption_system['plaintext_tokens']
    preproc_plaintext_sentences = decryption_system['preproc_plaintext_sentences']

    # Split into words
    ciphered_words = ciphered_sentence.split()
    deciphered_text = []

    for word in ciphered_words:
        is_punctuation = word in string.punctuation

        if is_punctuation:
            deciphered_text.append(word)
        else:
            # Extract key from last letter
            last_letter = find_last_letter(word)
            key = map_last_letter_to_key(last_letter)
            word_without_key = remove_last_letter(word)

            if key > 0 and key <= 26 and loaded_models[key-1] is not None:
                model = loaded_models[key - 1]
                code_tokenizer = code_tokenizers[key - 1]
                plaintext_token = plaintext_tokens[key - 1]

                # Tokenize and pad
                tokenized_word = texts_to_token(word_without_key, code_tokenizer)
                padded_word = pad([tokenized_word], preproc_plaintext_sentences[key-1].shape[1])
                padded_word = padded_word.reshape(-1, preproc_plaintext_sentences[key-1].shape[1], 1)

                # Predict
                prediction = model.predict(padded_word, verbose=0)
                prediction_word = logits_to_text(prediction[0], plaintext_token)
            else:
                prediction_word = word_without_key

            prediction_word = prediction_word.strip()
            deciphered_text.append(prediction_word)

    return ' '.join(deciphered_text)

def simple_caesar_encrypt(text, key=None):
    """Simple Caesar cipher for chat - adds key indicator"""
    if key is None:
        import random
        key = random.randint(1, 26)

    encrypted = caesar_cipher_encrypt(text, key)
    key_char = chr(ord('a') + key - 1)

    # Add key indicator to each word
    words = encrypted.split()
    words_with_key = [word + key_char for word in words]

    return ' '.join(words_with_key)


def simple_caesar_decrypt(text):
    """Simple Caesar cipher decryption - uses key indicator"""
    words = text.split()
    decrypted_words = []

    for word in words:
        if word and word[-1].isalpha():
            last_char = word[-1].lower()
            key = ord(last_char) - ord('a') + 1
            word_without_key = word[:-1]
            decrypted = caesar_cipher_decrypt(word_without_key, key)
            decrypted_words.append(decrypted)
        else:
            decrypted_words.append(word)

    return ' '.join(decrypted_words)



if __name__ == "__main__":
    print("="*70)
    print("DECRYPTION BACKEND INITIALIZATION")
    print("="*70)

    # Test simple encryption/decryption
    print("\nTesting simple Caesar cipher...")
    test_message = "hello world this is secret"
    encrypted = simple_caesar_encrypt(test_message)
    decrypted = simple_caesar_decrypt(encrypted)

    print(f"Original:  {test_message}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")

    print("\nBackend ready!")