# Simple Django Project

This is a simple Django project with a basic app that displays "Hello, World!" when you visit the homepage.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/simple_django_project.git
    ```

2. Navigate to the project directory:
    ```bash
    cd simple_django_project
    ```

3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. Run the migrations:
    ```bash
    python manage.py migrate
    ```

6. Start the development server:
    ```bash
    python manage.py runserver
    ```

7. Open your browser and go to `http://127.0.0.1:8000/` to see "Hello, World!".

## Requirements

- Python 3.x
- Django 4.x
