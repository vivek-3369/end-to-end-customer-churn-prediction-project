# 1. Creating Base image 
FROM python:3.12-slim

# 2. Set the Working Directory inside the container
WORKDIR /app

# 3. Copy all code files into container 
COPY . /app

# 4. To install the requirements in container
RUN pip install -r requirements.txt

# 5. Expose the port for the streamlit app
EXPOSE 8501

# 6. Command to the run the app in the container
CMD ["streamlit", "run", "app.py"]