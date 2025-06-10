import arxiv
import json
import os
import pickle
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
PAPER_DIR = "papers"

# Load environment variables
load_dotenv()
# Initialize the Gemini client with the API key from environment variables
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_drive_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    return service

def get_file_id_by_name(file_name: str) -> list:
    """
    Returns a list of file dicts matching the given file name.
    """
    service = get_drive_service()
    results = service.files().list(
        q=f"name='{file_name}'",
        fields="files(id, name, mimeType, size)"
    ).execute()
    files = results.get('files', [])
    return files  # List of dicts with id, name, mimeType, size

def extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
        return text if text.strip() else "No extractable text found in PDF."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_image(file_path: str) -> str:
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "No extractable text found in image."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def read_drive_file(file_id: str) -> str:
    """
    Downloads any file from Google Drive by file ID and saves it locally.
    Returns the content for text files, extracts text from PDFs/images, or the local file path for other types.
    """
    service = get_drive_service()
    file = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    file_name = file.get("name")
    mime_type = file.get("mimeType")
    request = service.files().get_media(fileId=file_id)
    local_path = os.path.join("downloads", file_name)
    os.makedirs("downloads", exist_ok=True)
    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    # Return content for text files, extract text for PDFs/images, otherwise return the file path
    if mime_type and mime_type.startswith("text/"):
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()
    elif mime_type == "application/pdf" or (file_name and file_name.lower().endswith(".pdf")):
        return extract_text_from_pdf(local_path)
    elif mime_type and mime_type.startswith("image/"):
        return extract_text_from_image(local_path)
    else:
        return f"File '{file_name}' (type: {mime_type}) downloaded to: {local_path}"

def read_drive_file_by_name(file_name: str) -> str:
    files = get_file_id_by_name(file_name)
    if not files:
        return f"No file found with name '{file_name}'."
    if len(files) == 1:
        return read_drive_file(files[0]['id'])
    # If multiple files, list them for user to choose
    return f"Multiple files found with name '{file_name}':\n" + json.dumps(files, indent=2)

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Convert max_results to int if it's a string
    if isinstance(max_results, str):
        max_results = int(max_results)
    
    # Ensure max_results is a positive integer
    max_results = max(1, min(max_results, 100))  # Limit between 1 and 100 results
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = list(client.results(search))  # Convert iterator to list
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."

# Define the tools schema using the correct format for Gemini API
tools = [
    {
        "function_declarations": [
            {
                "name": "search_papers",
                "description": "Search for papers on arXiv based on a topic and store their information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic to search for"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to retrieve (between 1 and 100)"
                        }
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "extract_info",
                "description": "Search for information about a specific paper across all topic directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {
                            "type": "string",
                            "description": "The ID of the paper to look for"
                        }
                    },
                    "required": ["paper_id"]
                }
            },
            {
                "name": "read_drive_file",
                "description": "Read a file from Google Drive using its file ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "The ID of the file to read from Google Drive"
                        }
                    },
                    "required": ["file_id"]
                }
            },
            {
                "name": "read_drive_file_by_name",
                "description": "Read a file from Google Drive by searching for its name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file to search for and read from Google Drive"
                        }
                    },
                    "required": ["file_name"]
                }
            }
        ]
    }
]

# Create the model
model = genai.GenerativeModel('gemini-2.0-flash')

def execute_tool(tool_name, tool_args):
    """Execute a tool with the given name and arguments."""
    if tool_name == "search_papers":
        return search_papers(**tool_args)
    elif tool_name == "extract_info":
        return extract_info(**tool_args)
    elif tool_name == "read_drive_file":
        return read_drive_file(**tool_args)
    elif tool_name == "read_drive_file_by_name":
        return read_drive_file_by_name(**tool_args)
    else:
        return f"Unknown tool: {tool_name}"

def process_query(query):
    """Process a user query using the Gemini model."""
    try:
        # Start a chat
        chat = model.start_chat(history=[])
        
        # Generate a response with tool calls
        response = chat.send_message(
            query,
            tools=tools,
            generation_config={"temperature": 0.7}
        )

        # Process the response
        if not response or not response.candidates:
            return "No response generated. Please try rephrasing your query."

        # Initialize result with empty string
        result = ""
        
        # Process all parts of the response
        for candidate in response.candidates:
            if not candidate or not candidate.content or not candidate.content.parts:
                continue
                
            for part in candidate.content.parts:
                # Handle text parts
                if hasattr(part, 'text') and part.text:
                    result += part.text + "\n"
                
                # Handle function calls
                if hasattr(part, 'function_call'):
                    tool_call = part.function_call
                    if not tool_call or not hasattr(tool_call, 'name'):
                        continue
                        
                    tool_name = tool_call.name
                    
                    # Convert MapComposite args to dict
                    if hasattr(tool_call, 'args'):
                        if isinstance(tool_call.args, dict):
                            tool_args = tool_call.args
                        else:
                            try:
                                # Convert MapComposite to dict
                                tool_args = dict(tool_call.args)
                            except Exception as e:
                                print(f"Warning: Could not convert tool args: {e}")
                                tool_args = {}
                    else:
                        tool_args = {}
                    
                    try:
                        # Execute the tool and get the result
                        tool_result = execute_tool(tool_name, tool_args)
                        
                        # Send the tool result back to continue the conversation
                        follow_up = chat.send_message(
                            f"Tool {tool_name} returned: {str(tool_result)}",
                            tools=tools
                        )
                        
                        # Process follow-up response parts
                        if follow_up and follow_up.candidates:
                            for follow_part in follow_up.candidates[0].content.parts:
                                if hasattr(follow_part, 'text') and follow_part.text:
                                    result += follow_part.text + "\n"
                    except Exception as e:
                        result += f"\nError executing tool {tool_name}: {str(e)}\n"

        return result.strip() if result.strip() else "I couldn't process that request. Please try rephrasing it."

    except Exception as e:
        return f"Error processing query: {str(e)}"

def chat_loop():
    """Main chat loop for interacting with the user."""
    print("Welcome to the Research Assistant! Type 'quit' to exit.")
    print("You can:")
    print("1. Search for papers on a topic (e.g., 'Find papers about machine learning')")
    print("2. Get information about a specific paper (e.g., 'Tell me about paper 2211.05071')")
    print("3. Read files from Google Drive (e.g., 'Read the file named example.pdf')")
    print("\nWhat would you like to do?")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            # Process the query and get response
            if user_input:
                response = process_query(user_input)
                print("\nAssistant:", response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    chat_loop()