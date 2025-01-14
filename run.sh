#!/bin/bash
# run.sh

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting pipeline execution...${NC}"

# Create necessary directories
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p data results result_img logs

# Function to check if a command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success: $1${NC}"
    else
        echo -e "${RED}✗ Error: $1 failed${NC}"
        exit 1
    fi
}

# 1. Install dependencies using uv
echo -e "\n${YELLOW}Installing dependencies with uv...${NC}"
uv pip install kagglehub pandas numpy matplotlib seaborn scikit-learn mpi4py python-dotenv pyyaml
check_status "Dependencies installation"

# 2. Run data fetcher scripts
echo -e "\n${YELLOW}Running data fetcher scripts...${NC}"
uv run data_config/data_fetcher.py
check_status "Data fetcher execution"

echo -e "\n${YELLOW}Fetching credit card data...${NC}"
uv run data_config/credit_card_data.py
check_status "Credit card data fetcher"

# 3. Verify data
echo -e "\n${YELLOW}Verifying downloaded data...${NC}"
uv run data_config/verify_data.py
check_status "Data verification"

# 4. Run Docker compose
echo -e "\n${YELLOW}Running Docker compose...${NC}"
docker-compose down  # Stop any running containers
docker-compose up --build -d
check_status "Docker compose"

# Wait for Docker containers to be ready
echo "Waiting for containers to be ready..."
sleep 10

# 5. Run visualization
echo -e "\n${YELLOW}Running visualization script...${NC}"
uv run data_config/visualize_results.py
check_status "Visualization generation"

echo -e "\n${GREEN}Pipeline completed successfully!${NC}"

# Print locations of results
echo -e "\n${YELLOW}Results can be found in:${NC}"
echo "- Visualizations: result_img/"
echo "- Clustering results: results/"
echo "- Logs: logs/"

# Optional: Display any warnings or issues
if [ -f "logs/warnings.log" ]; then
    echo -e "\n${YELLOW}Warnings during execution:${NC}"
    cat logs/warnings.log
fi