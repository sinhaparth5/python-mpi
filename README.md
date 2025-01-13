# MPI K-means Clustering with Docker

This project implements a distributed K-means clustering algorithm using MPI (Message Passing Interface) and Docker for easy deployment and scaling.

## Features
- Distributed processing using MPI
- Docker containerization
- Configurable parameters via YAML and environment variables
- Comprehensive logging
- Error handling and recovery
- Unit tests
- Data preprocessing and validation

## Requirements
- Docker
- Docker Compose
- Python 3.9+
- MPICH

## Quick Start
1. Clone the repository
2. Build the Docker image:
   ```bash
   docker-compose build
   ```
3. Run the clustering:
   ```bash
   docker-compose up
   ```

## Configuration
You can modify the clustering parameters in either:
- config.yaml
- .env file
- Environment variables when running Docker

## Project Structure
- src/: Source code
- tests/: Unit tests
- data/: Data files
- logs/: Log files (created at runtime)
- results/: Clustering results (created at runtime)

## Testing
Run the tests using:
```bash
docker-compose run mpi_master pytest
```

## Logging
Logs are stored in the logs/ directory and also output to console.