# 3D Interior Visualization System - FastAPI Project

This is a simple FastAPI project serving as a foundation for building web APIs. The main application is defined in the `main.py` file.

## Features

- REST API endpoints built with FastAPI
- Automatic API documentation available at `/docs` (Swagger UI) and `/redoc` (ReDoc)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/KappaAITeam/kappa-ai-video-insight-be.git
   cd kappa-ai
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The application will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API Documentation

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.

