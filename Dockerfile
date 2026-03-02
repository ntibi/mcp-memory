FROM ubuntu:24.04 AS chef
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.91.0
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install cargo-chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bin memory-server

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/memory-server /usr/local/bin/memory-server
COPY config.toml /etc/memory/config.toml
EXPOSE 8000
ENTRYPOINT ["memory-server", "--config", "/etc/memory/config.toml"]
