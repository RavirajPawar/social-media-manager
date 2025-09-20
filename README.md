# Social Media Content Manager

This is a production-ready social media content manager application built using FastAPI, LangChain, and LangGraph. The app is designed to run locally using Docker, including support for running LLM models locally.

## Features
- Content creation using LLMs
- Content scheduling
- Content analysis (tone, sentiment, etc.)
- Local storage for posts
- Fully containerized with Docker

## Tech Stack
- **FastAPI**: Backend framework
- **LangChain**: LLM workflows
- **LangGraph**: Workflow visualization
- **SQLite**: Local database
- **Docker**: Containerization

## Setup Instructions

### Prerequisites
- Python 3.11+
- Docker

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RavirajPawar/social-media-manager.git
   cd social-media-manager
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Access the API documentation at `http://127.0.0.1:8000/docs`.

### Docker Setup
1. Build the Docker image:
   ```bash
   docker-compose build
   ```

2. Run the application:
   ```bash
   docker-compose up
   ```

## License
This project is licensed under the MIT License.
