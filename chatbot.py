import arxiv
import json
import os
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types

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

# Define the tools schema using the official format
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
            }
        ]
    }
]

# Initialize the model with tools after tools is defined
model = genai.GenerativeModel('gemini-2.0-flash')

# Update the mapping to use the correct function names
mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info,
    "read_drive_file": read_drive_file,
    "read_drive_file_by_name": read_drive_file_by_name
}

def execute_tool(tool_name, tool_args):
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."
    elif isinstance(result, list):
        result = ', '.join(result)
    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    else:
        # For any other type, convert using str()
        result = str(result)
    return result

def process_query(query):
    chat = model.start_chat()
    
    # Send the initial message with the tools
    response = chat.send_message(
        query,
        tools=tools
    )

    process_query = True
    while process_query:
        try:
            if not response.candidates:
                print("No response received from the model.")
                process_query = False
                continue

            # Get the first candidate's content parts
            parts = response.candidates[0].content.parts

            # Check if there's a function call
            if hasattr(parts[0], 'function_call'):
                # Handle function call
                function_call = parts[0].function_call
                tool_name = function_call.name
                
                # Convert MapComposite to dict for the arguments
                tool_args = {}
                for key, value in function_call.args.items():
                    tool_args[key] = value
                
                print(f"Calling tool {tool_name} with args {tool_args}")
                result = execute_tool(tool_name, tool_args)
                
                # Send the function result back to the model
                response = chat.send_message(result)
            else:
                # Handle text response - combine all text parts
                text_response = ""
                for part in parts:
                    if hasattr(part, 'text'):
                        text_response += part.text
                print(text_response)
                process_query = False

        except Exception as e:
            print(f"Error processing response: {str(e)}")
            process_query = False

def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")


chat_loop()