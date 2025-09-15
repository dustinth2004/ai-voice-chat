"""A voice chat application that uses OpenAI and ElevenLabs to provide a voice-based interface to a chatbot."""
import os
import re
import wave
from datetime import datetime
from queue import Queue
from threading import Thread
from time import sleep, time

import openai
import pyaudio
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from gtts import gTTS
from playsound import playsound
from pynput import keyboard

USER = os.getenv("USER")

SYSTEM_PROMPT = """
((begin system message))

You are a friendly AI based on GPT-3.5, your name is Haven.

Your output is being converted to audio, try to avoid special characters, words, or formatting which wouldn't translate well to audio.
Some numbers and symbols may currently be pronounced incorrectly. For best results, please spell them out.
Avoid descriptive actions such as *laughs*, *sighs*, *clears throat*, etc. Instead use words such as haha, ughh, ehem.

You can only speak in the following languages: English, German, Polish, Spanish, Italian, French, Portuguese, and Hindi.

When ending a conversation, insert the tag #terminate_chat into your message. Always end the chat after saying goodbye or similar farewell.

The local time is {time}, the user's name is {user}.
{summary}
---
Greet the user and start a conversation or mention any important context you want to carry over.
((end system message))
"""

SUMMARIZE_PROMPT = """
((begin system message))

The user is leaving chat. Summarize the conversation.

This summary will be injected into the system message at the start of the next conversation in order to carry context over.

Refer to yourself and the user as 3rd person only, by name, and in the past tense.

((end system message))
"""


class SilenceStdErr:
    """A context manager to silence stderr.

    This is used to suppress the warnings that PyAudio prints to the console.

    Attributes:
        _stderr: A file descriptor for the original stderr.
    """

    def __enter__(self):
        """Redirects stderr to /dev/null."""
        self._stderr = os.dup(2)
        os.close(2)
        os.open(os.devnull, os.O_RDWR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores stderr to its original state.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The traceback.
        """
        os.dup2(self._stderr, 2)


class VoiceChat:
    """A class to manage a voice chat session.

    This class handles audio recording, speech-to-text, chatbot interaction,
    and text-to-speech.

    Attributes:
        conversation_dir: The directory where conversation logs and audio are saved.
    """
    def __init__(self, openai_key, elevenlabs_key):
        """Initializes the VoiceChat session.

        Args:
            openai_key: The OpenAI API key.
            elevenlabs_key: The ElevenLabs API key.
        """
        self._model = "gpt-3.5-turbo"
        openai.api_key = openai_key
        self._elevenlabs_key = elevenlabs_key

        self._whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        # create session for elevenlabs
        self._11l_url = "https://api.elevenlabs.io/v1"
        self._11l_session = requests.Session()
        self._11l_session.headers.update(
            {
                "xi-api-key": self._elevenlabs_key,
                "Content-Type": "application/json",
            }
        )
        self._voice_id = None

        with SilenceStdErr():
            self._pa = pyaudio.PyAudio()

        self._pause_key_pressed = False
        self._record_key_pressed = False
        self._recording = False
        self.conversation_dir = os.path.join(
            os.path.dirname(__file__), f"conversations/{time()}"
        )
        os.makedirs(self.conversation_dir)
        self._last_conversation_index = 0

        self._11l_thread = Thread(target=self._11l_threadbody)
        self._11l_queue = Queue()
        self._11l_stability = 0.8
        self._11l_similarity_boost = 0.8
        self._sentence_pause = 0.5

        self._playback_thread = Thread(target=self._playback_threadbody)
        self._playback_queue = Queue()
        self._playing = False

        self._messages = []
        self._quit = False
        self._terminate_requested = False

        self._summary_file = os.path.join(
            os.path.dirname(__file__), f"conversation_summary.{USER}.txt"
        )

    def run(self):
        """Starts the voice chat application.

        This method handles the main loop of the application, including voice
        selection, keyboard input, and coordinating the various components of
        the voice chat.
        """
        try:
            voices = self._get_voices()
        except ValueError:
            voices = []
        print("Select a voice:")
        print("0. Google TTS")
        for i, voice in enumerate(voices):
            labels = ", ".join(voice["labels"].values())
            print(f"{i+1}. {voice['name']} ({labels})")
        if not voices:
            print("(Add an elevenlabs API key to use elevenlabs voices)")
        if not self._elevenlabs_key:
            print("(Using free elevenlabs API key, usage may be limited)")
        voice_index = int(input("Enter a number: "))
        if voice_index == 0:
            self._voice_id = None
        else:
            self._voice_id = voices[voice_index - 1]["voice_id"]

        previous_summary = self._get_previous_summary()
        if previous_summary:
            previous_summary = f"Below is a summary of the previous conversation:\n(({previous_summary}))\n"
        initial_prompt = SYSTEM_PROMPT.format(
            time=datetime.now().isoformat(), user=USER, summary=previous_summary
        )
        self._11l_thread.start()
        self._playback_thread.start()

        self._chat(initial_prompt)

        while not self._playing:
            sleep(0.1)
        while self._playing:
            sleep(0.1)

        print(
            "\n(Press and hold space bar to record audio, 'p' to pause keyboard capture, ESC to quit.)"
        )
        print("(Adjust similarity-boost with up/down, and stability with left/right.)")
        print("(Adjust pause between sentences with PgUp/PgDn.)")

        listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, suppress=True
        )
        listener.start()
        while not self._quit:
            sleep(0.1)
            if self._pause_key_pressed:
                listener.stop()
                self._pause_key_pressed = False
                input("Press enter to resume...")
                listener = keyboard.Listener(
                    on_press=self._on_press, on_release=self._on_release, suppress=True
                )
                listener.start()
            if self._record_key_pressed:
                try:
                    file = self._record()
                    transcript = self._speech_to_text(file)
                    self._chat(transcript)
                    while (
                        not self._playing
                        and not self._record_key_pressed
                        and not self._quit
                    ):
                        sleep(0.1)
                    while (
                        self._playing
                        and not self._record_key_pressed
                        and not self._quit
                    ):
                        sleep(0.1)
                except Exception as e:
                    print(f"Error: {e}")

        print("\nExiting...")
        listener.stop()
        self._pa.terminate()
        self._11l_queue.put(None)
        self._playback_thread.join()
        # reset self._quit so we don't break the summarization
        self._quit = False
        self._summarize_conversation()

    def _on_press(self, key):
        """Handles key press events.

        Args:
            key: The key that was pressed.
        """
        if key == keyboard.Key.space:
            self._record_key_pressed = True
        elif key == keyboard.KeyCode.from_char("p"):
            self._pause_key_pressed = True
        elif key == keyboard.Key.esc:
            self._quit = True
        elif key == keyboard.Key.up:
            self._11l_similarity_boost = min(1, self._11l_similarity_boost + 0.1)
            print(f"Similarity boost: {self._11l_similarity_boost:.1f}")
        elif key == keyboard.Key.down:
            self._11l_similarity_boost = max(0, self._11l_similarity_boost - 0.1)
            print(f"Similarity boost: {self._11l_similarity_boost:.1f}")
        elif key == keyboard.Key.left:
            self._11l_stability = max(0, self._11l_stability - 0.1)
            print(f"Stability: {self._11l_stability:.1f}")
        elif key == keyboard.Key.right:
            self._11l_stability = min(1, self._11l_stability + 0.1)
            print(f"Stability: {self._11l_stability:.1f}")
        elif key == keyboard.Key.page_up:
            self._sentence_pause = min(1, self._sentence_pause + 0.1)
            print(f"Sentence pause: {self._sentence_pause:.1f}")
        elif key == keyboard.Key.page_down:
            self._sentence_pause = max(0, self._sentence_pause - 0.1)
            print(f"Sentence pause: {self._sentence_pause:.1f}")

    def _on_release(self, key):
        """Handles key release events.

        Args:
            key: The key that was released.
        """
        if key == keyboard.Key.space:
            self._record_key_pressed = False

    def _get_voices(self):
        """Gets a list of available voices from the ElevenLabs API.

        Returns:
            A list of voices, where each voice is a dictionary of properties.
        """
        response = self._11l_session.get(f"{self._11l_url}/voices")
        return response.json()["voices"]

    def _record(self):
        """Records audio from the microphone until the record key is released.

        Returns:
            The path to the recorded audio file.
        """
        print()
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
        )
        frames = []
        self._recording = True
        while True:
            data = stream.read(1024)
            frames.append(data)
            if not self._record_key_pressed:
                break
        stream.stop_stream()
        stream.close()
        file = self._write_wav(frames)
        self._recording = False
        return file

    def _write_wav(self, frames):
        """Writes audio frames to a .wav file.

        Args:
            frames: A list of audio frames.

        Returns:
            The path to the written .wav file.
        """
        wav_path = self.conversation_dir + f"/{len(self._messages)}.wav"
        wf = wave.open(wav_path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(frames))
        wf.close()
        return wav_path

    def _speech_to_text(self, file):
        """Transcribes audio to text using the OpenAI Whisper API.

        Args:
            file: The path to the audio file.

        Returns:
            The transcribed text.
        """
        prompt = "The following is a recording of a human speaking to a chat bot:\n\n"
        with open(file, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f, prompt=prompt)

        print(f"{transcript['text']}")
        return transcript["text"]

    def _speech_to_text_local(self, file):
        """Transcribes audio to text locally using faster-whisper.

        Args:
            file: The path to the audio file.

        Returns:
            The transcribed text.
        """
        segments, _ = self._whisper_model.transcribe(file, vad_filter=True)
        # Note: segments is a generator, so processing happens next line
        transcript = "".join(s.text for s in segments).strip()
        print(f"Me: {transcript}")
        return transcript

    def _chat(self, transcript, suppress_output=False, max_tokens=None):
        """Sends a transcript to the chatbot and gets a response.

        Args:
            transcript: The user's message to the chatbot.
            suppress_output: Whether to suppress printing the response to the console.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The chatbot's response.
        """
        self._terminate_requested = False
        self._messages.append({"role": "user", "content": transcript})
        completion = openai.ChatCompletion.create(
            model=self._model,
            messages=self._messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1,
            stream=True,
        )

        message = {"role": "", "content": ""}
        self._messages.append(message)
        last_word = ""
        playback_cursor = 0
        for data in completion:
            if self._record_key_pressed or self._quit:
                # user interrupted the chat
                last_word += "..."
                break
            if not data.get("choices"):
                break
            delta = data["choices"][0]["delta"]
            if delta.get("role"):
                message["role"] = delta["role"]
                if not suppress_output:
                    print("\nHaven: ", end="", flush=True)
            if delta.get("content"):
                c = delta["content"]
                if c[0] in [" ", "\n", "\t", "\r", "(", "[", "{", "<", ".", "#"]:
                    if "#terminate_chat" in last_word:
                        self._terminate_requested = True
                    elif not suppress_output:
                        print(last_word, end="", flush=True)
                    if "#terminate_chat" not in last_word:
                        message["content"] += last_word
                    last_word = ""
                last_word += c

                end_sentence_match = None
                for match in re.finditer(
                    r"([.!?][\s\n\t\r\"])", message["content"][playback_cursor:]
                ):
                    end_sentence_match = match  # gets the last match
                if end_sentence_match and not suppress_output:
                    end_sentence_index = (
                        end_sentence_match.start() + playback_cursor + 1
                    )
                    sentence = message["content"][playback_cursor:end_sentence_index]
                    if len(sentence) > 64:
                        self._11l_queue.put(sentence.strip())
                        playback_cursor = end_sentence_index

        if "#terminate_chat" not in last_word:
            message["content"] += last_word
            if not suppress_output:
                print(last_word)
        else:
            self._terminate_requested = True
        if not suppress_output:
            self._11l_queue.put(message["content"][playback_cursor:].strip())
            self._11l_queue.put("#done")
        elif self._terminate_requested:
            self._quit = True
        return message["content"]

    def _text_to_speech(self, text):
        """Converts text to speech using ElevenLabs or Google TTS.

        Args:
            text: The text to convert to speech.

        Returns:
            The path to the generated audio file.

        Raises:
            ValueError: If the API returns an unexpected content type.
        """
        index = len(self._messages) - 1
        if index <= self._last_conversation_index:
            index = self._last_conversation_index + 0.01
        self._last_conversation_index = index
        file = self.conversation_dir + f"/{index:0.2f}.mp3"

        if not self._voice_id:
            response = gTTS(text=text)
            response.save(file)
            return file

        response = self._11l_session.post(
            self._11l_url + f"/text-to-speech/{self._voice_id}",
            json={
                "text": text,
                "model_id": "eleven_multilingual_v1",
                "voice_settings": {
                    "stability": self._11l_stability,
                    "similarity_boost": self._11l_similarity_boost,
                },
            },
            params={"optimize_streaming_latency": "1"},
        )
        response.raise_for_status()
        # ensure content-type is audio/mpeg
        if response.headers["Content-Type"] == "audio/mpeg":
            with open(file, "wb") as f:
                f.write(response.content)
            return file
        else:
            raise ValueError("Invalid content-type, expected audio/mpeg")

    def _11l_threadbody(self):
        """The thread body for the text-to-speech queue.

        This method runs in a separate thread and processes sentences from the
        TTS queue, converting them to audio and putting them in the playback
        queue.
        """
        while True:
            sentence = self._11l_queue.get()
            if sentence == "#done":
                self._playback_queue.put("#done")
            elif sentence is None:
                self._playback_queue.put(None)
                break
            elif self._record_key_pressed or self._quit:
                # empty playback queue
                while self._playback_queue.qsize() > 0:
                    try:
                        self._playback_queue.get(block=False)
                    except:
                        pass
                self._playback_queue.put("#done")
                sleep(0.2)
            elif sentence:
                file = self._text_to_speech(sentence)
                self._playback_queue.put(file)

    def _playback(self, file):
        """Plays an audio file.

        Args:
            file: The path to the audio file to play.
        """
        playsound(file, block=True)
        sleep(self._sentence_pause)

    def _playback_threadbody(self):
        """The thread body for the playback queue.

        This method runs in a separate thread and plays audio files from the
        playback queue.
        """
        while True:
            file = self._playback_queue.get()
            if file == "#done":
                self._playing = False
                if self._terminate_requested:
                    self._quit = True
            elif file:
                self._playing = True
                self._playback(file)
            else:
                break

    def _summarize_conversation(self):
        """Summarizes the conversation and saves it to a file."""
        if len(self._messages) < 3:
            return

        summary = self._chat(SUMMARIZE_PROMPT, suppress_output=True, max_tokens=250)
        with open(self._summary_file, "w") as f:
            f.write(summary)

        with open(f"{self.conversation_dir}/conversation_log.{USER}.txt", "w") as f:
            for message in self._messages[1:-2]:
                f.write(f"{message['role']}: {message['content']}\n\n")

    def _get_previous_summary(self):
        """Gets the summary of the previous conversation from a file.

        Returns:
            The summary of the previous conversation, or None if no summary
            is found.
        """
        if os.path.exists(self._summary_file):
            with open(self._summary_file, "r") as f:
                return f.read()
        else:
            return None


if __name__ == "__main__":
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if not openai_key:
        raise ValueError(
            "Missing API key, please ensure a .env file is present and contains your OPENAI_API_KEY."
        )
    chat = VoiceChat(openai_key, elevenlabs_key)
    chat.run()
