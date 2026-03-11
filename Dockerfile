FROM rust:1.91-trixie AS chef
RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz -O /tmp/ort.tgz \
    && tar -xzf /tmp/ort.tgz -C /opt \
    && rm /tmp/ort.tgz
ENV ORT_LIB_LOCATION=/opt/onnxruntime-linux-x64-1.23.2/lib
ENV ORT_PREFER_DYNAMIC_LINK=1
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
ARG GIT_SHA=unknown
RUN MEMORY_GIT_SHA=${GIT_SHA} cargo build --release --bin memory-server

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so* /usr/lib/
COPY --from=builder /app/target/release/memory-server /usr/local/bin/memory-server
COPY config.toml /etc/memory/config.toml
EXPOSE 8000
ENTRYPOINT ["memory-server"]
CMD ["--config", "/etc/memory/config.toml"]
