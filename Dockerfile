FROM python:3.12-slim

# Install MPI
RUN apt-get update && apt-get install -y \
    mpich \
    build-essential \
    && rm -rf /var/lib/api/lists/*

WORKDIR /app

# Copy the requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Default command
CMD ["mpiexec", "-n", "4", "python", "-m", "src.main"]
