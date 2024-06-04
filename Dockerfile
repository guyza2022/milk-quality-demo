
# Use an official Python runtime as a parent image
FROM python:3.8.18

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . /app

# Expose port 8501 for the Streamlit app
EXPOSE 80

# Command to run the Streamlit app
CMD ["streamlit", "run", "Main.py","--server.enableCORS=false"]