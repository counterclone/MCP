# ArXiv Paper Search with Gemini and Google Drive Integration

This application allows you to search for academic papers on arXiv and save the results both locally and to Google Drive using Google's Gemini AI model for natural language interaction.

## Prerequisites

- Python 3.10.7 or higher
- A Google Cloud project with the following APIs enabled:
  - Google Drive API
  - Gemini API
- Google Cloud credentials (credentials.json) for Google Drive access
- Gemini API key

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```

4. Set up Google Drive authentication:
   - Place your `credentials.json` file in the root directory
   - On first run, you'll be prompted to authenticate with Google Drive

## Usage

1. Run the chatbot:
   ```bash
   python chatbot.py
   ```

2. Enter your queries about research papers. The application will:
   - Search for relevant papers on arXiv
   - Save paper information locally
   - Upload paper information to Google Drive
   - Use Gemini AI to help you interact with the papers

## Features

- Natural language interaction using Gemini 2.0 Flash
- ArXiv paper search and metadata extraction
- Local storage of paper information
- Automatic Google Drive backup
- Interactive chat interface

## File Structure

- `chatbot.py`: Main application code
- `requirements.txt`: Python dependencies
- `.env`: Environment variables
- `credentials.json`: Google Drive API credentials
- `papers/`: Local storage directory for paper information

## Error Handling

The application includes error handling for:
- Google Drive API authentication
- File operations
- API rate limits
- Network connectivity issues

## Contributing

Feel free to submit issues and enhancement requests! 