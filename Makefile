
run-triton:
	docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 --shm-size=4gb -v /Users/nikolay/PycharmProjects/pythonProject2/models:/models d4d4c3b477e7 tritonserver --model-repository=/models