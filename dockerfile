# Use Ubuntu as the base image
FROM ubuntu:24.04

# Install dependencies and clean up package cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gcc \
    python3 \
    python3-venv \
    python3-pip \
    make \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for LKH-3
WORKDIR /LKH-3

# Download and extract LKH-3
RUN wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz && \
    tar -xvzf LKH-3.0.13.tgz && \
    rm LKH-3.0.13.tgz

# Build LKH-3
WORKDIR /LKH-3/LKH-3.0.13
RUN make

# Ensure the LKH binary is in the PATH
RUN ln -s /LKH-3/LKH-3.0.13/LKH /usr/local/bin/LKH

# Create and activate a virtual environment, then install dependencies
WORKDIR /LKH-3/LKH-3.0.13
RUN python3 -m venv /venv
RUN /venv/bin/pip install --upgrade pip

# Set the PATH to ensure the virtual environment is used
ENV PATH="/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API script and environment file
COPY app.py /LKH-3/LKH-3.0.13/app.py
COPY .env /LKH-3/LKH-3.0.13/.env

# Expose port 8000
EXPOSE 8000

# Run the API server using the virtual environment
CMD ["/venv/bin/uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]
