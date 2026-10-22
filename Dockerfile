FROM python:3.11-slim

WORKDIR /app

# Set timezone to Indian Standard Time (IST)
ENV TZ="Asia/Kolkata"

# Install system dependencies & timezone support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make start script executable
RUN chmod +x start.sh

EXPOSE 8501

CMD ["bash", "start.sh"]
