IMAGE := memory-server
PORT := 8000
DB_PATH := ./

.PHONY: run build

run: build
	docker run --rm -it \
		-p $(PORT):8000 \
		-v $(DB_PATH):/data \
		$(IMAGE) \
		--config /etc/memory/config.toml \
		--listen-addr 0.0.0.0:8000 \
		--db-path /data/memory.db

build:
	docker build --build-arg GIT_SHA=$$(git rev-parse --short HEAD) -t $(IMAGE) .
