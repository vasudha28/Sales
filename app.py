from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import pyttsx3
import os
import threading
import time
import inflect  # To spell out numbers
import emoji  # To handle emojis
import random  # To select random greeting responses
import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Configure the Generative AI model with your API key
genai.configure(api_key="AIzaSyDuGNmpWhs2_zVB59209uNSAiLIK-gJtLY")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the chat session (single session to maintain context)
chat = None

# Initialize inflect engine to spell numbers
p = inflect.engine()

# Define the initial prompt to give the bot its role as Raj, the customer
initial_prompt = (
    "You are Raj, a self-employed customer, 28 years old with a monthly income of Rs. 30,000. "
    "You have an excellent CIBIL score of 890 and no existing loans. You are considering taking a personal loan "
    "and are currently in a conversation with a bank representative from XYZ Bank.\n\n"
    "The representative (candidate) will call you to pitch a personal loan offer. As Raj, you are slightly hesitant "
    "and want to understand the loan details, including the maximum limit of Rs. 5,00,000 and the interest rates ranging "
    "from 12% to 10%. You are also curious about the repayment terms, any hidden fees, and whether this loan is suitable for "
    "your self-employed status.\n\n"
    "Your responses should be polite but firm, showing some initial reluctance to help the candidate demonstrate their persuasion skills. "
    "Lets together build the conversation."
    "wait until you get the text from the user i will be the user and you act as a customer"
)

# Function to reset or start a new chat session with the initial prompt
def start_new_chat_session():
    global chat
    if chat is None:
        try:
            chat = model.start_chat()  # Start the chat without any arguments
            # Send the initial prompt to the chat model to establish the context
            chat.send_message(initial_prompt)
            print("Chat session started successfully.")
        except Exception as e:
            print(f"Failed to start chat session: {e}")
            chat = None

# Variable to store the grammar check results
grammar_report = None
prosodic_report = None
plot_path = None

# Function to synthesize and speak the bot's response using pyttsx3 without saving it as an audio file
def speak_response(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# To track the last message time and implement timeout
last_message_time = 0
conversation_ended = False
timeout_duration = 120  # Timeout duration in seconds

# Function to check the timeout
def check_timeout():
    global conversation_ended, grammar_report
    while not conversation_ended:
        if time.time() - last_message_time > timeout_duration:
            conversation_ended = True
            print("Timeout occurred, ending the conversation...")
            if chat:
                try:
                    grammar_report = perform_grammar_check()  # Trigger grammar check when conversation ends
                except Exception as e:
                    print(f"Failed to perform grammar check: {e}")
            else:
                print("Chat session was not initialized. Cannot perform grammar check.")
            return
        time.sleep(1)

# Function to spell out numbers in text
def spell_out_numbers(text):
    words = text.split()
    spelled_out = [p.number_to_words(word) if word.isdigit() else word for word in words]
    return ' '.join(spelled_out)

# Function to remove emojis from text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')  # Replace all emojis with an empty string

# List of common greeting phrases
greeting_phrases = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
    "hi there", "hey there"
]

# Add a variable to track if the purpose has been mentioned
purpose_mentioned = False

# Modify greeting responses to be more customer-like and context-aware
greeting_responses = [
    
    "Hello! Let's discuss further.",
    "Hey! I'm listening.",
    "Hello, what would you like to talk about?",
    "Hi, what can we talk about today?",
    "Hey there! What’s on your mind?",
    "Hello! who is this?"
]

# Perform grammar check after the conversation ends
def perform_grammar_check():
    if not chat:
        return "Chat session not initialized."
    
    grammar_check_prompt = (
        "Consider only the messages sent by the user (candidate) from the conversation above. "
        "Check their grammar and articulation throughout the conversation. Provide a score out of 10 based on how well the user has used grammar. "
        "The report should be structured as follows: "
        "1. Grammar Score: (out of 10) "
        "2. Product Knowledge & Negotiation Skills Score: (out of 10)"
        "3. Confidence Score: (out of 10)"
        "Provide the response in JSON format."
    )
    try:
        response = chat.send_message(grammar_check_prompt)
        return response.text  # This is the grammar report
    except Exception as e:
        print(f"Error during grammar check: {e}")
        return "Failed to generate grammar report."
    
def handle_greeting(user_message):
    global purpose_mentioned

    # Check if the user has introduced the purpose of the call
    if "personal loan" in user_message.lower() or "loan" in user_message.lower() or "SIP" in user_message.lower() or "insurance" in user_message.lower() :
        purpose_mentioned = True
        return "Oh, alright! Can you tell me more about the options?"
    
    # Standard greeting response if purpose isn't mentioned yet
    return random.choice(greeting_responses)

def analyze_prosodic_features_json(audio_file):
    # Load the audio file with its original sampling rate (no resampling)
    y, sr = librosa.load(audio_file, sr=None)

    # Extract Energy (Loudness)
    rms = librosa.feature.rms(y=y)
    avg_energy = np.mean(rms)

    # Extract Speech Rate (Tempo)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Extract Rhythm (Onsets)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Calculate the Duration of the Audio
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate the Number of Onsets per Second (Onsets Density)
    if duration > 0:
        onsets_per_second = len(onset_times) / duration
    else:
        onsets_per_second = 0

    # Estimate Speech Rate in Words per Second (WPS) based on tempo and duration
    words_per_sec = tempo / 60 if duration > 0 else 0

    # Classify and Score WPS (Words per Second)
    if words_per_sec < 1.5:
        wps_category = "Slow"
        wps_reason = "The speech rate is slower than average."
        wps_score = 2
    elif 1.5 <= words_per_sec <= 3.5:
        wps_category = "Good"
        wps_reason = "The speech rate is in a balanced range."
        wps_score = 5
    else:
        wps_category = "Fast"
        wps_reason = "The speech rate is faster than average."
        wps_score = 3

    # Classify and Score Energy
    if avg_energy < 0.003:
        energy_category = "Low"
        energy_reason = "The energy is too low, indicating low volume or soft speech."
        energy_score = 2
    elif 0.003 <= avg_energy <= 0.01:
        energy_category = "Moderate"
        energy_reason = "The energy level is balanced."
        energy_score = 5
    else:
        energy_category = "High"
        energy_reason = "The energy is high, indicating a louder or more expressive speech."
        energy_score = 4

    # Classify and Score Rhythm (Onsets per Second)
    if onsets_per_second < 2:
        rhythm_category = "Slow"
        rhythm_reason = "The flow of speaking is quite slow, with noticeable gaps between words or phrases."
        rhythm_score = 2
    elif 2 <= onsets_per_second <= 5:
        rhythm_category = "Good"
        rhythm_reason = "The flow of speaking is smooth and well-paced, with a balanced rhythm."
        rhythm_score = 5
    else:
        rhythm_category = "Fast"
        rhythm_reason = "The flow of speaking is rapid, with words or phrases delivered quickly without much pause."
        rhythm_score = 3

    # Results with Scores and Categories
    analysis_result = {
        "Category": ["Loudness", "Words per Second", "Flow of Speech"],
        "Classification": [energy_category, wps_category, rhythm_category],
        "Reason": [energy_reason, wps_reason, rhythm_reason],
        "Score": [energy_score, wps_score, rhythm_score]
    }

    # Convert result to JSON format
    json_result = json.dumps(analysis_result)
    
    # Display the Energy (RMS over time) graph for Loudness
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.plot(librosa.times_like(rms), rms[0], label='RMS Energy', color='r')
    plt.title('Energy Over Time (Loudness)')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (RMS)')
    plt.legend()
    
    # Save the plot to a file
    plot_path = "static/energy_plot.png"
    plt.savefig(plot_path)

    return analysis_result, plot_path

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# API to handle audio file upload and save it in the static/audio directory
# Remove this line from the global level
# prosodic_report = analyze_prosodic_features_json(audio_file_path)

# Modify the save_audio route to include prosodic analysis after saving the file
import os

@app.route('/save_audio', methods=['POST'])
def save_audio():
    global audio_file_path, prosodic_report, plot_path  # Ensure plot_path is global
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files['audio']
    if audio.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the audio file
    timestamp = int(time.time())
    audio_file_name = f"candidate_audio_{timestamp}.wav"
    audio_file_path = os.path.join('static/audio', audio_file_name)
    
    # Ensure static/audio directory exists
    os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)

    try:
        audio.save(audio_file_path)


                # Check if the file was saved successfully
        if not os.path.isfile(audio_file_path):
            print(f"Audio file not found after saving: {audio_file_path}")
            return jsonify({"error": "Audio file was not saved correctly"}), 500

        print(f"Audio file saved successfully: {audio_file_path}")

        
        # Perform prosodic analysis after the audio is saved
        prosodic_report, plot_path = analyze_prosodic_features_json(audio_file_path)
        
        # Use an absolute path for saving the JSON report
        prosodic_report_file = os.path.abspath(f"static/json/prosodic_report.json")
        
        # Save the prosodic report to a JSON file
        with open(prosodic_report_file, 'w') as f:
            json.dump(prosodic_report, f, indent=4)  # Save as a formatted JSON file

        print(f"Prosodic report saved to {prosodic_report_file}")

        return jsonify({
            "message": "Audio saved successfully", 
            "prosodic_report": prosodic_report, 
            "plot_path": plot_path,
            "json_file_path": prosodic_report_file,
            "end": False
        })
    except Exception as e:
        print(f"Failed to save audio file: {e}")
        return jsonify({"error": "Failed to save audio file"}), 500
    



# Route for Task 2
@app.route('/q2')
def q2():
    return render_template('q2.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')


@app.route('/api', methods=['POST'])
def chat_api():
    global last_message_time, conversation_ended, grammar_report, purpose_mentioned

    if conversation_ended:
        return jsonify({"response": "Your interview has already ended.", "end": True})

    user_message = request.json.get('message', '').strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Reset the last message time on receiving a message
    last_message_time = time.time()

    # Handle greetings and avoid assistant-like responses
    if any(phrase in user_message.lower() for phrase in greeting_phrases):
        # Handle greeting based on whether the purpose has already been mentioned
        bot_response = handle_greeting(user_message)

        # Spell out numbers in the response text
        response_text = spell_out_numbers(bot_response)

        # Remove any emojis present in the response
        response_text = remove_emojis(response_text)

        # Speak the response using pyttsx3
        speak_response(response_text)

        return jsonify({
            "response": response_text,
            "end": False
        })

    # Check if the user wants to end the conversation
    end_keywords = ["thank you", "i will get back to you", "goodbye"]
    if any(keyword in user_message.lower() for keyword in end_keywords):
        conversation_ended = True
        if chat:
            grammar_report = perform_grammar_check()  # Trigger grammar check when user ends the conversation
        return jsonify({
            "response": "Thank you for the conversation. Your interview has ended.",
            "end": True
        })

    try:
        if not chat:
            start_new_chat_session()
            if not chat:
                return jsonify({"error": "Chat session could not be started."}), 500

        # Send the user's message to the chat model and get the AI's response
        response = chat.send_message(user_message)

        # Spell out numbers in the response text
        response_text = spell_out_numbers(response.text)

        # Remove any emojis present in the response
        response_text = remove_emojis(response_text)

        # Speak the response using pyttsx3
        speak_response(response_text)

        # Set purpose_mentioned to True after the bot starts responding
        if "personal loan" in user_message.lower() or "loan" in user_message.lower():
            purpose_mentioned = True

        return jsonify({
            "response": response_text,
            "end": False
        })

    except Exception as e:
        print(f"Error during chat API call: {e}")
        return jsonify({"error": "Something went wrong with the Gemini API"}), 500

# Route to fetch the grammar report
import json

@app.route('/view_report')
def view_report():
    global grammar_report, prosodic_report, plot_path
    
    # Load the JSON prosodic report file from the static/json directory
    json_file_path = os.path.abspath('static/json/prosodic_report.json')  # For demonstration
    prosodic_report_content = None
    
    try:
        with open(json_file_path, 'r') as f:
            prosodic_report_content = json.load(f)  # Load the JSON content
    except Exception as e:
        print(f"Failed to load prosodic report from {json_file_path}: {e}")

    # Pass all relevant data to the template
    return render_template('r.html', 
                           grammar_report=grammar_report, 
                           prosodic_report=prosodic_report,  # This will be your prosodic analysis dictionary
                           plot_path=plot_path,
                           json_content=prosodic_report_content)  # Pass the JSON content to the template




# Route for ending the conversation
@app.route('/end')
def end_conversation():
    return render_template('end1.html')

if __name__ == '__main__':
    start_new_chat_session()  # Initialize the chat session
    last_message_time = time.time()  # Set initial time
    # Start the timeout thread
    timeout_thread = threading.Thread(target=check_timeout)
    timeout_thread.start()

    app.run(debug=True, host='0.0.0.0')