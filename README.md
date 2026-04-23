# Reason Over Search

## Papers:

1. [ReSearch](https://www.alphaxiv.org/abs/2503.19470)
2. [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

## Instructions Milestone 1: Baseline [Search-R1](https://www.alphaxiv.org/abs/2503.09516)

Goal: Reproduce the Search-R1 3B baseline (both base and instruct variants) on four benchmarks:
- Bamboogle
- 2WikiMultiHopQA
- HotpotQA
- MuSiQue

### Step-by-step

1. Prepare the runtime environment on Vast.ai.
   - Use this Docker image: [pantomiman/reason-over-search-v1](https://hub.docker.com/r/pantomiman/reason-over-search-v1)
   - Or build and push your own image using `docker/reason-over-search-v1/README.md`.
   - Create a Vast.ai custom template from that image and start an instance.

2. Set up the local retriever.
   - Follow `local_retriever/README.md`.
   - If you are using the Docker image above, required environments are already installed, so you can skip environment creation.
   - Download:
     - corpus
     - indexes
     - embedding model

3. Run and validate the retriever service.
   - Start the retriever using `local_retriever/README.md`.
   - Run the health/search test calls to confirm it is serving correctly.
   - Resource note: the retriever runs on CPU and needs about `~80 GB` CPU RAM.

4. Set up evaluation models and SGLang.
   - Go to `evaluation_search_r1/`.
   - Activate the evaluation environment.
   - Download both Search-R1 model variants (base and instruct).
   - Start the SGLang server for the model being evaluated.

5. Run baseline evaluations.
   - Benchmarks: Bamboogle, 2WikiMultiHopQA, HotpotQA, MuSiQue.
   - Run each benchmark `3` times per model variant.
   - Report the average score across the 3 runs for each benchmark.