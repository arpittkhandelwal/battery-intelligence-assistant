# Use official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the python scripts and the dataset into the container
COPY loader.py preprocess.py features.py qa.py plot.py bot.py ./
COPY ["Dataset Nasa.csv", "./"]

# Run the bot
CMD ["python", "bot.py"]
