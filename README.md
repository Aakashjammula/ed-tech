# ed-tech

## 📝 Description
ed-tech Assistant is a voice-enabled chatbot that integrates with Google Generative AI (Gemini 2.0 flash) and rag using FastAPI as the backend and a simple HTML/CSS/JavaScript frontend. It allows users to interact with an AI assistant  using voice input and receive spoken responses, creating a seamless voice-based conversational experience and there is also a ui to upload your text book.

## 🚀 Features
- 🎙️ **Voice Recognition:** Converts speech to text using browser-based Web Speech API.
- 💬 **LLM Integration:** Utilizes the Gemini 2.0 flash model via LangChain's Google Generative AI integration with rag.
- 🔊 **Text-to-Speech:** Delivers AI responses using the browser's Speech Synthesis API.
- 🌐 **FastAPI Backend:** Efficiently handles API requests to the LLM.
- 🛠️ **Bootstrap UI:** Clean and responsive frontend design.
- 🧠 **Streaming Responses:** Provides real-time response streaming from the LLM.

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Backend:** FastAPI (Python)
- **AI Model:** Gemini 2.0 flash via LangChain

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
  
### Installation
1. **Clone the repository:**
```bash
git clone <repository-url>
cd ed-tech
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup environment variables:**
Create a `.env` file and add your API keys and configurations.
```plaintext
GOOGLE_API_KEY=your_google_genai_api_key
```

4. **Run the server:**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 3000
```
or
```bash
python server.py
```

5. **Access the app:**
Open `http://localhost:3000` in your browser.

## 💡 How to Use
1. Click **Start Listening** to begin voice input.
2. Speak your query to the AI assistant.
3. The AI will respond with both **text** and **speech**.
4. Use **Clear** to reset the conversation.
5. upload the textbook by drag and drop your pdf and click **upload textbook**

![ui](images/bot.png)
![ui](images/upload.png)

## 🤝 Contributing
Feel free to fork this project and submit pull requests!

## 📄 License
This project is licensed under the Apache-2.0 License.


