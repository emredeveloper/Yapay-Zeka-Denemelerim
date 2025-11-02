import os
import zipfile
import json
import time
import hashlib
import tempfile
import re
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

import gradio as gr
from google import genai
from google.genai import types

# Add new imports for enhanced features
import matplotlib.pyplot as plt
import networkx as nx
from functools import lru_cache
import plotly.graph_objects as go
import plotly.express as px

# Retrieve API key for Google GenAI from the environment variables.
GOOGLE_API_KEY = "AIz..xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual API key

# Initialize the client so that it can be reused across functions.
CLIENT = genai.Client(api_key=GOOGLE_API_KEY)

# Global variables
EXTRACTED_FILES = {}

# Store chat sessions
CHAT_SESSIONS = {}

# Cache for performance optimization
FILE_CONTENT_CACHE = {}
ANALYSIS_CACHE = {}

# App configuration 
TITLE = """<h1 align="center">✨ Gemini Code Analysis</h1>"""
AVATAR_IMAGES = (None, "https://media.roboflow.com/spaces/gemini-icon.png")

# Update model configuration and supported MIME types
MODEL_CONFIG = {
    "primary": "gemini-2.0-flash-001",
    "fallback": "gemini-1.5-flash-001",
    "last_resort": "gemini-1.0-pro"
}

# MIME type mapping - only use officially supported types for Gemini Flash
MIME_TYPE_MAPPING = {
    ".py": "text/plain",
    ".js": "text/plain",
    ".jsx": "text/plain",
    ".ts": "text/plain",
    ".tsx": "text/plain",
    ".html": "text/plain",
    ".css": "text/plain",
    ".json": "text/plain",  # Changed from application/json to text/plain
    ".jsonl": "text/plain",
    ".xml": "text/plain",
    ".svg": "text/plain",
    ".md": "text/plain",
    ".txt": "text/plain",
    ".csv": "text/plain",
}

# Add code analysis feature flags
ENABLE_SUMMARY = True
ENABLE_FILE_STATISTICS = True
ENABLE_AUTO_SUGGESTIONS = True
ENABLE_DEPENDENCY_GRAPH = True
ENABLE_SEARCH = True
ENABLE_EXPORT = True
ENABLE_FILE_TREE = True

# List of supported text extensions (alphabetically sorted)
TEXT_EXTENSIONS = [
    ".bat",
    ".c",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".php",
    ".ps1",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
]

# List of virtual environment directories to exclude
VENV_DIRS = [
    ".env",
    ".venv",
    "env",
    "venv",
    "ENV",
    "VENV",
    "__pycache__",
    "node_modules",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
]

# Add constants for file size limits
MAX_FILE_SIZE_BYTES = 500 * 1024  # 500KB per file
MAX_TOTAL_CONTENT_BYTES = 2 * 1024 * 1024  # 2MB total content
MAX_FILES_PER_REQUEST = 10  # Maximum files to send in one request

# ===== CORE FUNCTIONS (MOVE THESE TO THE TOP) =====

def extract_text_from_zip(zip_file_path: str) -> Dict[str, str]:
    """Extract text content from a ZIP file."""
    text_contents = {}

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                # Skip directories
                if file_info.filename.endswith("/"):
                    continue

                # Skip virtual environment directories
                if any(venv_dir in file_info.filename.split('/') for venv_dir in VENV_DIRS):
                    continue

                # Skip binary files and focus on text files
                file_ext = os.path.splitext(file_info.filename)[1].lower()

                if file_ext in TEXT_EXTENSIONS:
                    try:
                        with zip_ref.open(file_info) as file:
                            content = file.read().decode("utf-8", errors="replace")
                            text_contents[file_info.filename] = content
                    except Exception as e:
                        text_contents[file_info.filename] = (
                            f"Error extracting file: {str(e)}"
                        )
    except zipfile.BadZipFile:
        # Return empty dict if the file is not a valid ZIP
        return {}

    return text_contents


def is_valid_zip_file(file_path: str) -> bool:
    """Check if a file is a valid ZIP file and provide detailed error information."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Verify the zip file by checking the central directory
            zf.testzip()
            return True
    except zipfile.BadZipFile as e:
        print(f"BadZipFile error: {str(e)} for file {file_path}")
        return False
    except zipfile.LargeZipFile as e:
        print(f"LargeZipFile error: {str(e)} for file {file_path}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)} for file {file_path}")
        return False


def extract_text_from_single_file(file_path: str) -> Dict[str, str]:
    """Extract text content from a single file."""
    text_contents = {}
    filename = os.path.basename(file_path)
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext in TEXT_EXTENSIONS:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()
                text_contents[filename] = content
        except Exception as e:
            text_contents[filename] = f"Error reading file: {str(e)}"

    return text_contents


def get_file_statistics(files_dict):
    """Generate statistics about the extracted files."""
    stats = {
        "total_files": 0,
        "total_lines": 0,
        "extensions": {},
        "largest_file": {"name": "", "size": 0, "lines": 0},
        "smallest_file": {"name": "", "size": float('inf'), "lines": float('inf')}
    }
    
    for zip_name, files in files_dict.items():
        for filename, content in files.items():
            stats["total_files"] += 1
            
            # Count lines
            lines = content.count('\n') + 1
            stats["total_lines"] += lines
            
            # Track file size
            file_size = len(content.encode('utf-8'))
            
            # Track by extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in stats["extensions"]:
                stats["extensions"][ext] = {"count": 0, "lines": 0}
            stats["extensions"][ext]["count"] += 1
            stats["extensions"][ext]["lines"] += lines
            
            # Track largest file
            if file_size > stats["largest_file"]["size"]:
                stats["largest_file"] = {
                    "name": filename,
                    "size": file_size,
                    "lines": lines
                }
            
            # Track smallest file
            if file_size < stats["smallest_file"]["size"]:
                stats["smallest_file"] = {
                    "name": filename,
                    "size": file_size,
                    "lines": lines
                }
    
    # Remove inf if no files were processed
    if stats["smallest_file"]["size"] == float('inf'):
        stats["smallest_file"] = {"name": "N/A", "size": 0, "lines": 0}
    
    return stats


def format_statistics(stats):
    """Format statistics into a readable string."""
    result = "📊 **Code Statistics**\n\n"
    result += f"- Total Files: {stats['total_files']}\n"
    result += f"- Total Lines: {stats['total_lines']}\n\n"
    
    result += "**File Types:**\n"
    for ext, data in sorted(stats['extensions'].items()):
        result += f"- {ext}: {data['count']} files ({data['lines']} lines)\n"
    
    result += f"\n**Largest File:** {stats['largest_file']['name']} "
    result += f"({stats['largest_file']['size']/1024:.1f} KB, {stats['largest_file']['lines']} lines)\n"
    
    result += f"**Smallest File:** {stats['smallest_file']['name']} "
    result += f"({stats['smallest_file']['size']/1024:.1f} KB, {stats['smallest_file']['lines']} lines)\n"
    
    return result


def update_statistics():
    """Update the statistics display panel"""
    if not EXTRACTED_FILES:
        return "No files loaded yet. Upload files to see statistics."
    
    stats = get_file_statistics(EXTRACTED_FILES)
    return format_statistics(stats)


def upload_zip(files: Optional[List[str]], chatbot: List[Union[dict, gr.ChatMessage]]):
    """Process uploaded files (ZIP or individual text files) with improved error handling."""
    global EXTRACTED_FILES

    # Handle multiple file uploads
    if len(files) > 1:
        total_files_processed = 0
        total_files_extracted = 0
        file_types = set()

        # Process each file
        for file in files:
            filename = os.path.basename(file)
            file_ext = os.path.splitext(filename)[1].lower()

            # Process based on file type
            if file_ext == ".zip" and is_valid_zip_file(file):
                extracted_files = extract_text_from_zip(file)
                file_types.add("zip")
            else:
                extracted_files = extract_text_from_single_file(file)
                file_types.add("text")

            if extracted_files:
                total_files_extracted += len(extracted_files)
                # Store the extracted content in the global variable
                EXTRACTED_FILES[filename] = extracted_files

            total_files_processed += 1

        # Create a summary message for multiple files
        file_types_str = (
            "files"
            if len(file_types) > 1
            else ("ZIP files" if "zip" in file_types else "text files")
        )

        # Create a list of uploaded file names
        file_list = "\n".join([f"- {os.path.basename(file)}" for file in files])

        chatbot.append(
            gr.ChatMessage(
                role="user",
                content=f"<p>📚 Multiple {file_types_str} uploaded ({total_files_processed} files)</p><p>Extracted {total_files_extracted} text file(s) in total</p><p>Uploaded files:</p><pre>{file_list}</pre>",
            )
        )

    # Handle single file upload
    elif len(files) == 1:
        file = files[0]
        filename = os.path.basename(file)
        file_ext = os.path.splitext(filename)[1].lower()

        # Process based on file type
        if file_ext == ".zip":
            # Additional debugging for ZIP files
            file_size = os.path.getsize(file)
            print(f"Attempting to open ZIP file: {filename}, Size: {file_size/1024:.2f} KB")
            
            if is_valid_zip_file(file):
                extracted_files = extract_text_from_zip(file)
                file_type_msg = "📦 ZIP file"
            else:
                # More detailed error message with troubleshooting options
                error_message = f"""<p>❌ File uploaded: {filename} could not be processed as a ZIP file.</p>
<p>Possible reasons:</p>
<ul>
    <li>The file might be corrupted or incomplete</li>
    <li>The file might use an unsupported compression method</li>
    <li>The file might be password-protected</li>
</ul>
<p>Try to:</p>
<ul>
    <li>Re-download or re-create the ZIP file</li>
    <li>Use a different compression tool (like 7-Zip)</li>
    <li>Create a standard ZIP archive without encryption</li>
</ul>"""
                
                chatbot.append(
                    gr.ChatMessage(
                        role="user",
                        content=error_message,
                    )
                )
                return chatbot, update_statistics()
        else:
            extracted_files = extract_text_from_single_file(file)
            file_type_msg = "📄 File"

        if not extracted_files:
            chatbot.append(
                gr.ChatMessage(
                    role="user",
                    content=f"<p>{file_type_msg} uploaded: {filename}, but no text content was found or the file format is not supported.</p>",
                )
            )
        else:
            file_list = "\n".join([f"- {name}" for name in extracted_files.keys()])
            chatbot.append(
                gr.ChatMessage(
                    role="user",
                    content=f"<p>{file_type_msg} uploaded: {filename}</p><p>Extracted {len(extracted_files)} text file(s):</p><pre>{file_list}</pre>",
                )
            )

            # Store the extracted content in the global variable
            EXTRACTED_FILES[filename] = extracted_files

    # Clear cache if it's getting too large
    clear_cache_if_needed()
    
    # Update statistics
    stats_text = update_statistics()
    
    return chatbot, stats_text


def user(text_prompt: str, chatbot: List[gr.ChatMessage]):
    """Process user text input and ensure a response."""
    if not text_prompt.strip():
        return "", chatbot
    
    # Add user message to chat
    chatbot.append(gr.ChatMessage(role="user", content=text_prompt))
    
    try:
        # Handle special commands
        if text_prompt.startswith("/search ") and ENABLE_SEARCH:
            query = text_prompt[8:].strip()
            results = search_code(query)
            formatted_results = format_search_results(results)
            chatbot.append(gr.ChatMessage(role="assistant", content=formatted_results))
        elif text_prompt == "/deps" and ENABLE_DEPENDENCY_GRAPH:
            graph_html = generate_dependency_graph()
            chatbot.append(gr.ChatMessage(role="assistant", content=f"Dependency graph generated:\n\n{graph_html}"))
        elif text_prompt == "/tree" and ENABLE_FILE_TREE:
            tree_html = render_file_tree()
            chatbot.append(gr.ChatMessage(role="assistant", content=f"File tree:\n\n{tree_html}"))
        elif text_prompt == "/stats" and ENABLE_FILE_STATISTICS:
            if EXTRACTED_FILES:
                stats = get_file_statistics(EXTRACTED_FILES)
                formatted_stats = format_statistics(stats)
                chatbot.append(gr.ChatMessage(role="assistant", content=formatted_stats))
            else:
                chatbot.append(gr.ChatMessage(role="assistant", content="No files loaded yet. Upload files to see statistics."))
        else:
            # For any other input, provide a generic response
            chatbot.append(gr.ChatMessage(
                role="assistant", 
                content="I'm a code analysis assistant. You can use commands like /stats, /search, /deps, or /tree to analyze your code."
            ))
    except Exception as e:
        # Catch any errors and provide feedback
        chatbot.append(gr.ChatMessage(role="assistant", content=f"Error processing your request: {str(e)}"))
    
    return "", chatbot


def reset_app(chatbot):
    """Reset the application state."""
    global EXTRACTED_FILES, CHAT_SESSIONS, FILE_CONTENT_CACHE, ANALYSIS_CACHE

    # Clear all global variables
    EXTRACTED_FILES = {}
    CHAT_SESSIONS = {}
    FILE_CONTENT_CACHE = {}
    ANALYSIS_CACHE = {}

    # Reset the chatbot with a welcome message
    return [
        gr.ChatMessage(
            role="assistant",
            content="App has been reset. You can start a new conversation or upload new files.",
        )
    ], "No files loaded yet. Upload files to see statistics."


# Add function to clear cache if memory usage gets too high
def clear_cache_if_needed():
    """Clear the cache if it gets too large."""
    if len(FILE_CONTENT_CACHE) > 50:  # Arbitrary threshold
        FILE_CONTENT_CACHE.clear()
    
    if len(ANALYSIS_CACHE) > 50:
        ANALYSIS_CACHE.clear()

# Add a utility function to handle different archive formats
def try_alternate_extraction(file_path: str) -> Dict[str, str]:
    """Try to extract content using different methods if standard ZIP extraction fails."""
    # This is a placeholder for future implementation
    # Could use libraries like py7zr for 7z files, tarfile for tar archives, etc.
    print(f"Attempting alternate extraction methods for {file_path}")
    return {}

# ===== 1. NEW FEATURE: CODE SEARCH FUNCTIONALITY =====

def search_code(query: str) -> Dict[str, List[Dict]]:
    """Search through extracted files for a specific query and return matching results."""
    results = {}
    
    if not query or not EXTRACTED_FILES:
        return results
    
    # Convert query to lowercase for case-insensitive search
    query_lower = query.lower()
    
    for zip_name, files in EXTRACTED_FILES.items():
        results[zip_name] = {}
        for filename, content in files.items():
            matches = []
            for i, line in enumerate(content.split('\n')):
                if query_lower in line.lower():
                    # Find surrounding context (3 lines before and after)
                    lines = content.split('\n')
                    start = max(0, i - 3)
                    end = min(len(lines), i + 4)
                    context = '\n'.join(lines[start:end])
                    
                    matches.append({
                        'line_number': i + 1,
                        'context': context,
                        'full_line': line
                    })
            
            if matches:
                results[zip_name][filename] = matches
        
        # Remove empty zip entries
        if not results[zip_name]:
            del results[zip_name]
    
    return results

def format_search_results(results: Dict) -> str:
    """Format search results into a readable string."""
    if not results:
        return "No results found."
    
    formatted = "🔍 **Search Results:**\n\n"
    total_matches = 0
    
    for zip_name, files in results.items():
        for filename, matches in files.items():
            formatted += f"**{filename}** ({len(matches)} matches):\n"
            
            for match in matches[:5]:  # Limit to first 5 matches per file
                formatted += f"- Line {match['line_number']}: `{match['full_line'].strip()}`\n"
                formatted += "```\n" + match['context'] + "\n```\n"
            
            if len(matches) > 5:
                formatted += f"... and {len(matches) - 5} more matches in this file\n"
            
            formatted += "\n"
            total_matches += len(matches)
    
    formatted = f"Found {total_matches} total matches.\n\n" + formatted
    return formatted

# ===== 2. NEW FEATURE: FILE TREE VISUALIZATION =====

def build_file_tree():
    """Build a hierarchical tree structure from the extracted files."""
    if not EXTRACTED_FILES:
        return {"name": "root", "children": []}
    
    tree = {"name": "root", "children": []}
    
    for zip_name, files in EXTRACTED_FILES.items():
        zip_node = {"name": zip_name, "children": []}
        
        # Group files by directory structure
        directories = {}
        
        for filename in files.keys():
            parts = filename.split('/')
            curr_path = ""
            
            # Create directory nodes
            for i, part in enumerate(parts[:-1]):
                if i == 0:
                    parent_path = ""
                    curr_path = part
                else:
                    parent_path = curr_path
                    curr_path = f"{curr_path}/{part}"
                
                if curr_path not in directories:
                    directories[curr_path] = {
                        "name": part,
                        "path": curr_path,
                        "parent": parent_path,
                        "children": []
                    }
        
        # Create file nodes and add them to their parent directories
        for filename in files.keys():
            parts = filename.split('/')
            
            if len(parts) == 1:
                # File is in the root
                file_node = {"name": parts[0], "type": "file", "path": filename}
                zip_node["children"].append(file_node)
            else:
                # File is in a subdirectory
                parent_dir = '/'.join(parts[:-1])
                file_node = {"name": parts[-1], "type": "file", "path": filename}
                
                if parent_dir in directories:
                    directories[parent_dir]["children"].append(file_node)
        
        # Build the directory hierarchy
        root_dirs = []
        for dir_path, dir_info in directories.items():
            if dir_info["parent"] == "":
                root_dirs.append(dir_info)
            else:
                parent = directories.get(dir_info["parent"])
                if parent:
                    parent["children"].append(dir_info)
        
        # Add root directories to the zip node
        zip_node["children"].extend(root_dirs)
        tree["children"].append(zip_node)
    
    return tree

def get_file_icon(filename):
    """Get appropriate icon for file type."""
    ext = os.path.splitext(filename)[1].lower()
    
    icons = {
        ".py": "🐍",
        ".js": "📜",
        ".html": "🌐",
        ".css": "🎨",
        ".json": "📋",
        ".md": "📝",
        ".txt": "📄",
        ".xml": "🔖",
        ".yml": "⚙️",
        ".yaml": "⚙️",
        ".cpp": "©️",
        ".c": "©️",
        ".h": "🔧",
        ".java": "☕",
        ".rb": "💎",
        ".php": "🐘",
        ".go": "🔵",
        ".rs": "🦀",
        ".ts": "📘",
        ".tsx": "📘",
        ".jsx": "📙",
    }
    
    return icons.get(ext, "📄")

def get_language_for_extension(ext: str) -> str:
    """Map file extension to language for syntax highlighting."""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".md": "markdown",
        ".xml": "xml",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".sh": "bash",
        ".bat": "batch",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".java": "java",
        ".php": "php",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust"
    }
    
    return language_map.get(ext, "")

def render_file_tree():
    """Render the file tree as HTML with robust file clicking."""
    tree = build_file_tree()
    
    if not tree["children"]:
        return "<p>No files loaded.</p>"
    
    html = "<div class='file-tree'>"
    html += "<ul class='tree'>"
    
    def render_node(node, indent=0):
        node_html = ""
        
        if node.get("type") == "file":
            icon = get_file_icon(node["name"])
            file_path = node.get('path', '')
            # Simplify the click handling with a direct approach
            node_html += f'''<li class='file'>
                <a href="#" onclick="event.preventDefault(); handleFileClick('{file_path}');">
                    {icon} {node['name']}
                </a>
            </li>'''
        else:
            node_html += f"<li class='directory'><span class='folder'>📁 {node['name']}</span>"
            
            if node.get("children"):
                node_html += "<ul>"
                for child in sorted(node["children"], key=lambda x: (x.get("type", "") != "file", x["name"])):
                    node_html += render_node(child, indent + 1)
                node_html += "</ul>"
            
            node_html += "</li>"
        
        return node_html
    
    for node in tree["children"]:
        html += render_node(node)
    
    html += "</ul></div>"
    
    # Use a more direct approach for file handling with a simplified global function
    html = """
    <style>
    .file-tree { 
        max-height: 300px; 
        overflow: auto; 
        background: #2d333b; 
        padding: 10px; 
        border-radius: 5px; 
        color: #e6edf3; 
        font-family: monospace;
    }
    .tree { list-style-type: none; padding-left: 10px; margin: 0; }
    .tree ul { list-style-type: none; padding-left: 20px; margin: 0; }
    .file { 
        cursor: pointer; 
        margin: 4px 0; 
        padding: 3px 5px;
        border-radius: 3px;
    }
    .file:hover { background-color: #444c56; }
    .file a {
        color: #e6edf3;
        text-decoration: none;
        display: block;
    }
    .file.selected { 
        background-color: #0366d6; 
    }
    .file.selected a {
        color: white;
    }
    .folder { 
        cursor: pointer; 
        font-weight: bold; 
        color: #e6edf3;
        padding: 3px 5px;
        border-radius: 3px;
        display: inline-block;
    }
    .folder:hover { background-color: #444c56; }
    .directory { margin: 4px 0; }
    </style>
    <script>
    // Global file selector function - simpler and more direct
    function handleFileClick(filePath) {
        console.log("File clicked:", filePath);
        
        // Store the selected file path in a global variable
        window.selectedFilePath = filePath;
        
        // Use a custom event to communicate with Python backend
        const event = new CustomEvent('file-selected', { detail: filePath });
        document.dispatchEvent(event);
        
        // Update the display directly using fetch API
        fetch('/file-content', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: filePath })
        })
        .then(response => response.json())
        .then(data => {
            // Update the code viewer with the file content
            const codeViewer = document.getElementById('code-viewer');
            const titleElement = document.getElementById('code-viewer-title');
            
            if (codeViewer) {
                codeViewer.innerHTML = data.content;
            }
            if (titleElement) {
                titleElement.innerHTML = data.title;
            }
            
            // Update visual selection
            document.querySelectorAll('.file').forEach(el => {
                el.classList.remove('selected');
            });
            document.querySelectorAll('.file a').forEach(el => {
                if (el.textContent.includes(filePath.split('/').pop())) {
                    el.parentElement.classList.add('selected');
                }
            });
        })
        .catch(error => {
            console.error('Error loading file:', error);
        });
        
        // Also try to update using Gradio's methods
        try {
            const inputElement = document.querySelector('#selected_file_path input');
            if (inputElement) {
                inputElement.value = filePath;
                inputElement.dispatchEvent(new Event('input', { bubbles: true }));
                
                setTimeout(() => {
                    const submitButton = document.querySelector('#file_path_submit button');
                    if (submitButton) submitButton.click();
                }, 100);
            }
        } catch(e) {
            console.log("Alternative file selection method failed:", e);
        }
    }
    
    // Set up the listener once the page is loaded
    document.addEventListener('DOMContentLoaded', function() {
        console.log("File tree initialized with click handlers");
    });
    </script>
    """ + html
    
    return html

def handle_file_selection(file_path):
    """Handle file selection from the file tree - more robust implementation."""
    if not file_path or not file_path.strip():
        return "No file selected. Please click on a file in the tree view.", "No file selected."
    
    if not EXTRACTED_FILES:
        return "No files have been loaded yet. Please upload files first.", "No files loaded."
    
    print(f"File selection request received for: {file_path}")
    
    # Find the file in the extracted files - more thorough search
    for zip_name, files in EXTRACTED_FILES.items():
        if file_path in files:
            content = files[file_path]
            print(f"File found in {zip_name}: {file_path}")
            
            # Get file extension for syntax highlighting
            ext = os.path.splitext(file_path)[1].lower()
            language = get_language_for_extension(ext)
            
            # Format code for display with proper markdown
            formatted_content = f"```{language}\n{content}\n```"
            
            return formatted_content, f"File: {file_path}"
        
        # Try alternative path formats (Windows/Unix path conversion)
        normalized_path = file_path.replace('\\', '/')
        if normalized_path in files:
            content = files[normalized_path]
            print(f"File found with normalized path in {zip_name}: {normalized_path}")
            ext = os.path.splitext(normalized_path)[1].lower()
            language = get_language_for_extension(ext)
            formatted_content = f"```{language}\n{content}\n```"
            return formatted_content, f"File: {normalized_path}"
    
    # List all available files to help debugging
    all_files = []
    for zip_name, files in EXTRACTED_FILES.items():
        all_files.extend(list(files.keys()))
    
    print(f"File not found. Available files: {all_files}")
    return f"File not found: {file_path}. Please try selecting a different file.", "File Not Found"

# ===== 3. NEW FEATURE: DEPENDENCY GRAPH GENERATION =====

@lru_cache(maxsize=32)
def extract_dependencies():
    """Extract dependencies between files based on imports/includes."""
    if not EXTRACTED_FILES:
        return {}
    
    dependencies = {}
    files_content = {}
    
    # Flatten all files
    for zip_name, files in EXTRACTED_FILES.items():
        for filename, content in files.items():
            files_content[filename] = content
    
    # Process Python files
    for filename, content in files_content.items():
        if not filename.endswith('.py'):
            continue
            
        dependencies[filename] = []
        
        # Extract imports
        import_patterns = [
            r'^import\s+([\w\.]+)',  # import module
            r'^from\s+([\w\.]+)\s+import',  # from module import
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                module = match.group(1)
                
                # Check if this is a local import
                module_parts = module.split('.')
                potential_file = module_parts[0] + '.py'
                
                # Check if file exists in our codebase
                for other_file in files_content.keys():
                    other_basename = os.path.basename(other_file)
                    if other_basename == potential_file:
                        dependencies[filename].append(other_file)
    
    # Process JavaScript/TypeScript files
    for filename, content in files_content.items():
        if not filename.endswith(('.js', '.jsx', '.ts', '.tsx')):
            continue
            
        if filename not in dependencies:
            dependencies[filename] = []
        
        # Extract imports
        import_patterns = [
            r'import.*from\s+[\'"](.+)[\'"]',  # import x from 'module'
            r'require\([\'"](.+)[\'"]\)',  # require('module')
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                module = match.group(1)
                
                # Handle relative imports
                if module.startswith('./') or module.startswith('../'):
                    # Resolve the relative path
                    base_dir = os.path.dirname(filename)
                    module_path = os.path.normpath(os.path.join(base_dir, module))
                    
                    # Check for extensions
                    for ext in ['.js', '.jsx', '.ts', '.tsx', '']:
                        test_path = module_path + ext
                        if test_path in files_content:
                            dependencies[filename].append(test_path)
                            break
    
    return dependencies

def generate_dependency_graph():
    """Generate dependency graph visualization with improved file handling."""
    dependencies = extract_dependencies()
    
    if not dependencies:
        return "No dependencies found among the uploaded files."
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for file, deps in dependencies.items():
        file_short = os.path.basename(file)
        G.add_node(file_short)
        
        for dep in deps:
            dep_short = os.path.basename(dep)
            G.add_edge(file_short, dep_short)
    
    # If the graph is empty, return a message
    if not G.nodes():
        return "No dependencies found among the uploaded files."
    
    # Create a temporary file for the graph image with absolute path
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time())
    filename = f"dependency_graph_{timestamp}.png"
    graph_path = os.path.join(temp_dir, filename)
    
    # Draw the graph with improved styling
    plt.figure(figsize=(12, 8))
    plt.title("Code Dependency Graph", fontsize=16)
    plt.margins(0.1)
    
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    # Draw nodes with improved visuals
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=3000, 
                          alpha=0.9,
                          linewidths=1,
                          edgecolors='navy')
    
    # Draw edges with better styling
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          width=2,
                          alpha=0.7,
                          arrowsize=20,
                          arrowstyle='->')
    
    # Add labels with better fonts
    nx.draw_networkx_labels(G, pos, 
                           font_size=11, 
                           font_family='sans-serif',
                           font_weight='bold')
    
    # Save with higher quality
    plt.savefig(graph_path, format="PNG", dpi=200, bbox_inches='tight', transparent=False)
    plt.close()
    
    # Generate HTML with the image and ensure it's visible
    # Use a data URL to embed the image directly
    try:
        with open(graph_path, "rb") as img_file:
            import base64
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            img_src = f"data:image/png;base64,{img_data}"
    except Exception as e:
        print(f"Error reading graph image: {e}")
        img_src = ""
    
    html = f"""
    <div style="text-align: center; margin: 20px 0; background-color: white; padding: 20px; border-radius: 8px;">
        <h3 style="color: #333; margin-bottom: 15px;">Dependency Graph</h3>
        <div style="max-width: 100%; overflow: auto;">
            <img src="{img_src}" alt="Dependency Graph" style="max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
        </div>
        <p style="font-size: 0.9em; margin-top: 15px; color: #555;">Graph shows dependencies between files based on imports/requires.</p>
    </div>
    """
    
    return html

# ===== 4. NEW FEATURE: PERFORMANCE OPTIMIZATION WITH CACHING =====

@lru_cache(maxsize=128)
def cached_extract_text_from_zip(zip_file_path: str) -> Dict[str, str]:
    """Cached version of extract_text_from_zip for performance."""
    # Generate a cache key based on the file path and last modified time
    file_stat = os.stat(zip_file_path)
    cache_key = f"{zip_file_path}_{file_stat.st_mtime}"
    
    if cache_key in FILE_CONTENT_CACHE:
        return FILE_CONTENT_CACHE[cache_key]
    
    # If not in cache, extract content as before
    result = extract_text_from_zip(zip_file_path)
    
    # Store in cache
    FILE_CONTENT_CACHE[cache_key] = result
    
    return result

# ===== 5. NEW FEATURE: EXPORT/SAVE CONVERSATION =====

def export_conversation(chatbot):
    """Export the conversation and analysis to a file with better error handling."""
    if not chatbot:
        return "No conversation to export."
    
    # Create a dictionary with all the relevant data
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation": [
            {"role": msg.role if hasattr(msg, "role") else msg.get("role"),
             "content": msg.content if hasattr(msg, "content") else msg.get("content")}
            for msg in chatbot
        ],
        "files_analyzed": len(EXTRACTED_FILES),
        "statistics": get_file_statistics(EXTRACTED_FILES) if EXTRACTED_FILES else None
    }
    
    # Convert to JSON with proper encoding
    export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    # Create a unique filename with absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    export_file = os.path.join(desktop_path, f"code_analysis_{timestamp}.json")
    
    try:
        # Ensure the directory exists
        os.makedirs(desktop_path, exist_ok=True)
        
        # Write to file with explicit encoding
        with open(export_file, "w", encoding="utf-8") as f:
            f.write(export_json)
        
        print(f"File exported successfully to: {export_file}")
        return f"Conversation exported to:\n{export_file}"
    except Exception as e:
        error_msg = f"Error exporting file: {str(e)}"
        print(error_msg)
        return error_msg

# Add the export button handler
def handle_export(chatbot):
    """Export the current conversation and analysis to a file."""
    result = export_conversation(chatbot)
    
    # Add a message to the chatbot about the export
    chatbot.append(
        gr.ChatMessage(
            role="assistant",
            content=f"💾 {result}",
        )
    )
    
    return chatbot

# Add a new function to create a REST API endpoint for file content
def file_content_api(request: gr.Request):
    """API endpoint to get file content."""
    try:
        request_json = request.json
        file_path = request_json.get("file_path", "")
        content, title = handle_file_selection(file_path)
        return {"content": content, "title": title}
    except Exception as e:
        return {"content": f"Error loading file: {str(e)}", "title": "Error"}

# Define the Gradio UI components
chatbot_component = gr.Chatbot(
    label="Gemini Code Analysis",
    type="messages",
    bubble_full_width=False,
    avatar_images=AVATAR_IMAGES,
    scale=2,
    height=350,
)
text_prompt_component = gr.Textbox(
    placeholder="Ask a question or upload code files to analyze...",
    show_label=False,
    autofocus=True,
    scale=28,
)
upload_zip_button_component = gr.UploadButton(
    label="Upload",
    file_count="multiple",
    file_types=[".zip"] + TEXT_EXTENSIONS,
    scale=1,
    min_width=80,
)

send_button_component = gr.Button(
    value="Send", variant="primary", scale=1, min_width=80
)
reset_button_component = gr.Button(
    value="Reset", variant="stop", scale=1, min_width=80
)

# Add new components for enhanced features
search_component = gr.Textbox(
    placeholder="Search in code...",
    label="Code Search",
    show_label=True,
)

search_results_component = gr.Markdown(
    value="Enter a search term and click 'Search' to find text in your code files.",
    label="Search Results",
    visible=True
)

file_tree_component = gr.HTML(
    value="<p>Upload files to see the file structure.</p>",
    label="File Structure"
)

dependency_graph_component = gr.HTML(
    value="<p>Upload Python, JavaScript, or TypeScript files to see dependencies.</p>",
    label="Dependency Graph"
)

# Add statistics component
stats_component = gr.Markdown(
    value="No files loaded yet. Upload files to see statistics.",
    label="Code Statistics",
    visible=True
)

# Add export button
export_button_component = gr.Button(
    value="💾 Export", variant="secondary", scale=1, min_width=80
)

# Add code viewer component
code_viewer = gr.Markdown(
    value="Select a file from the file tree to view its content.",
    label="Code Viewer",
    elem_id="code-viewer"
)

code_viewer_title = gr.Markdown(
    value="No file selected.",
    elem_id="code-viewer-title"
)

# Define input lists for button chaining
user_inputs = [text_prompt_component, chatbot_component]

# Create tabs for different analysis views
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.HTML(TITLE)
    
    # Status bar with commands
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 10px;">
    <span style="font-size: 0.9em;">Commands: 
        <code>/stats</code> - Show code statistics | 
        <code>/search [query]</code> - Search in code | 
        <code>/deps</code> - View dependencies | 
        <code>/tree</code> - View file structure
    </span>
    </div>
    """)
    
    # Main chat and input area
    with gr.Column():
        # Chat area
        chatbot_component.render()
        
        # Input area
        with gr.Row(equal_height=True):
            text_prompt_component.render()
            send_button_component.render()
            upload_zip_button_component.render()
            reset_button_component.render()
            export_button_component.render()
    
    # Create the hidden input with improved structure
    with gr.Column(visible=False) as hidden_components:
        file_path_input = gr.Textbox(
            label="Selected File Path", 
            elem_id="selected_file_path",
            value=""
        )
        file_path_submit = gr.Button("Load File", elem_id="file_path_submit")
    
    # Analysis tabs
    with gr.Tabs():
        with gr.TabItem("Statistics"):
            stats_component.render()
        
        with gr.TabItem("File Explorer"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### File Structure")
                    file_tree_component.render()
                with gr.Column(scale=2):
                    code_viewer_title.render()
                    code_viewer.render()
        
        with gr.TabItem("Search"):
            with gr.Row():
                search_component.render()
                search_button = gr.Button(value="🔍 Search", variant="primary")
            search_results_component.render()
        
        with gr.TabItem("Dependency Graph"):
            dependency_graph_component.render()
            generate_graph_button = gr.Button(value="Generate Graph", variant="primary")

    # Fix the search button click handler
    search_button.click(
        fn=lambda query: format_search_results(search_code(query)),
        inputs=[search_component],
        outputs=[search_results_component]
    )
    
    generate_graph_button.click(
        fn=generate_dependency_graph,
        inputs=[],
        outputs=[dependency_graph_component]
    )
    
    export_button_component.click(
        fn=handle_export,
        inputs=[chatbot_component],
        outputs=[chatbot_component]
    )
    
    # Fix the prompt submission with both options
    send_button_component.click(
        fn=user,
        inputs=[text_prompt_component, chatbot_component],
        outputs=[text_prompt_component, chatbot_component],
        api_name="send_message"
    )
    
    # Also connect the Enter key in the text box
    text_prompt_component.submit(
        fn=user,
        inputs=[text_prompt_component, chatbot_component],
        outputs=[text_prompt_component, chatbot_component]
    )
    
    # Make file selection more robust by connecting both change and click events
    file_path_input.change(
        fn=handle_file_selection,
        inputs=[file_path_input],
        outputs=[code_viewer, code_viewer_title],
        api_name="update_file_content"
    )
    
    file_path_submit.click(
        fn=handle_file_selection,
        inputs=[file_path_input],
        outputs=[code_viewer, code_viewer_title],
        api_name="load_file_content"
    )
    
    # Add a REST API endpoint for file content
    demo.add_api_route("/file-content", file_content_api, methods=["POST"])
    
    # Update the upload handler to also update the file tree
    upload_zip_button_component.upload(
        fn=lambda files, chatbot: (
            upload_zip(files, chatbot)[0],
            upload_zip(files, chatbot)[1],
            render_file_tree(),
            "Upload Python or JavaScript files to see dependencies.",
            "Select a file from the file tree to view its content.",
            "No file selected."
        ),
        inputs=[upload_zip_button_component, chatbot_component],
        outputs=[
            chatbot_component, 
            stats_component, 
            file_tree_component, 
            dependency_graph_component,
            code_viewer,
            code_viewer_title
        ],
        queue=False
    )
    
    # Update all panels when resetting
    reset_button_component.click(
        fn=lambda chatbot: (
            reset_app(chatbot)[0],
            reset_app(chatbot)[1],
            "<p>Upload files to see the file structure.</p>",
            "<p>Upload Python or JavaScript files to see dependencies.</p>",
            "Select a file from the file tree to view its content.",
            "No file selected."
        ),
        inputs=[chatbot_component],
        outputs=[
            chatbot_component, 
            stats_component, 
            file_tree_component, 
            dependency_graph_component,
            code_viewer,
            code_viewer_title
        ],
        queue=False
    )

# Launch the demo interface with more debugging options
demo.queue(max_size=99, api_open=True).launch(
    debug=True,
    show_error=True,
    server_port=9595,
    server_name="localhost",
    share=False  # Set to True if you want to create a public link
)

