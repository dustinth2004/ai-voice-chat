# AI Voice Chat

This project is a voice-based chat application that uses OpenAI's GPT-3.5 for conversation, OpenAI's Whisper for speech-to-text, and ElevenLabs or Google TTS for text-to-speech. It allows for a hands-free, voice-based interaction with a friendly AI assistant named Haven.

## How it Works

The application operates by listening for a user to press and hold the spacebar. While the spacebar is held, it records audio from the microphone. Upon release, the recorded audio is sent to OpenAI's Whisper API for transcription. The resulting text is then sent to the GPT-3.5 model to generate a response. This response is streamed back and converted into audio sentence by sentence using either ElevenLabs or Google TTS, and then played back to the user.

The application maintains conversation context within a single session and saves a summary of the conversation upon exit to provide context for the next session.

## Features

- **Push-to-talk**: Simple and intuitive press-and-hold spacebar interface for recording.
- **Realistic TTS**: High-quality, realistic voice responses using ElevenLabs.
- **Contextual Conversations**: Maintains the context of the conversation throughout a session.
- **Cross-Session Context**: Saves a summary of each conversation to provide context for future sessions.
- **Conversation Logging**: All conversations, including audio and text logs, are saved locally.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API keys:**
    Create a `.env` file in the root of the project and add your API keys:
    ```
    OPENAI_API_KEY=<your_openai_api_key>
    ELEVENLABS_API_KEY=<your_elevenlabs_api_key>
    ```
    *Note: You can use the application without an ElevenLabs API key, but you will be limited to the free tier and a smaller selection of voices.*

## Usage

1.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2.  **Run the application:**
    ```bash
    python main.py
    ```

3.  **Select a voice:**
    When the application starts, you will be prompted to select a voice. Enter the number corresponding to your choice.

4.  **Interact with the AI:**
    - **Hold Spacebar**: Record your message.
    - **Release Spacebar**: Send your message for transcription and response.
    - **Press `p`**: Pause the keyboard listener. This is useful if you need to use your keyboard in another application. Press Enter in the console to resume.
    - **Press `ESC`**: Exit the application.

5.  **Adjust TTS Settings (during playback):**
    - **Up/Down Arrows**: Adjust the similarity boost for ElevenLabs TTS.
    - **Left/Right Arrows**: Adjust the stability for ElevenLabs TTS.
    - **Page Up/Page Down**: Adjust the pause between sentences.

## Known Issues

- **Language Support**: The application is primarily tested in English and may not perform well in other languages.
- **Keyboard Exclusivity**: While the application is running, it captures all keyboard input, so you cannot use your keyboard in other applications unless you pause the listener.
- **TTS Latency**: There can be a noticeable delay in audio synthesis, which may result in pauses between sentences.
- **Limited Long-Term Memory**: The context between sessions is based on a summary, which may lead to the AI having a "poor memory" of past conversations.
- **Cost**: Use of the ElevenLabs API can be expensive and consume your character quota quickly. Google TTS is provided as a free but lower-quality alternative.
- **OS Compatibility**: The application has only been tested on Ubuntu.

## Dependencies

- **OpenAI**: For speech-to-text (Whisper) and chatbot responses (GPT-3.5).
- **ElevenLabs/gTTS**: For text-to-speech.
- **PyAudio**: For audio recording.
- **pynput**: for keyboard monitoring.
- **faster-whisper**: For local speech-to-text.
- **python-dotenv**: For managing environment variables.
- **playsound**: For playing audio files.
- **requests**: For making API requests.
