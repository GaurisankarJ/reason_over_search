# Retriever

## Environment Setup

1. Create a conda environment

```
conda create -n retriever python=3.10 -y
conda activate retriever
```

2. Install requirements

```
pip install -r requirements.txt
```

## Run Retriever

```
# TEST
python retriever_serving.py --config retriever_config_mini.yaml --num_retriever 1 --port 3005

python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 3005
```

## API Endpoints

### Health

```bash
curl -X GET "http://127.0.0.1:3005/health"
```

### Search

```bash
curl -X POST "http://127.0.0.1:3005/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who wrote The Lord of the Rings?",
    "top_n": 3,
    "return_score": false
  }'
```

### Search (with scores)

```bash
curl -X POST "http://127.0.0.1:3000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who wrote The Lord of the Rings?",
    "top_n": 3,
    "return_score": true
  }'
```

### Batch Search

```bash
curl -X POST "http://127.0.0.1:3005/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      "capital of France",
      "largest planet in our solar system"
    ],
    "top_n": 2,
    "return_score": false
  }'
```

### Batch Search (with scores)

```bash
curl -X POST "http://127.0.0.1:3000/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      "capital of France",
      "largest planet in our solar system"
    ],
    "top_n": 2,
    "return_score": true
  }'
```