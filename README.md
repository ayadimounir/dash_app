
# Project Setup

This guide will walk you through setting up a virtual environment and installing the required packages from `requirements.txt` for both Linux and Windows systems.

## Prerequisites

Make sure you have Python installed on your machine. You can check by running:

```bash
python --version
```

If Python is not installed, download and install it from [python.org](https://www.python.org/downloads/).

---

## 1. Setting up a Virtual Environment

### Linux / macOS

1. **Create a virtual environment:**

    Open your terminal and run the following command:

    ```bash
    python3 -m venv venv
    ```

2. **Activate the virtual environment:**

    ```bash
    source venv/bin/activate
    ```

3. **Install dependencies:**

    After activating the virtual environment, install the required packages using:

    ```bash
    pip install -r requirements.txt
    ```

4. **Deactivate the virtual environment:**

    Once you're done, you can deactivate the virtual environment by running:

    ```bash
    deactivate
    ```

### Windows

1. **Create a virtual environment:**

    Open Command Prompt or PowerShell and run:

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment:**

    In Command Prompt:

    ```bash
    venv\Scripts\activate
    ```

    In PowerShell:

    ```bash
    .\venv\Scripts\Activate
    ```

3. **Install dependencies:**

    Once the virtual environment is active, install the packages:

    ```bash
    pip install -r requirements.txt
    ```

4. **Deactivate the virtual environment:**

    When you're finished, deactivate the environment by typing:

    ```bash
    deactivate
    ```

---

## 2. Running the Application

After setting up the virtual environment and installing dependencies, you can run your application as usual:

```bash
python app.py
```

---

## Notes

- Always activate your virtual environment before running the application.
- Make sure to keep your `requirements.txt` up to date by running `pip freeze > requirements.txt` whenever new packages are added.
