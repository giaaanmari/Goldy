FROM ubuntu:20.04

#Install necessary libraries
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && \
    DEBIAN_FRONTEND="noninteractive" TZ="Asia/Taipei" apt-get install -y libenchant-2-dev

#Install Python3 and pip3
RUN apt install -y python3 python3-pip

# Setup a working directory
WORKDIR /app

# Copy the contents of the current directory to the container at /app
COPY . /app


# Install required dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Set the command to run when the container starts
CMD ["python3", "app.py"]