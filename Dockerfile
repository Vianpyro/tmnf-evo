# ── Stage: builder ────────────────────────────────────────────────────────────
# Compiles the Rust extension and produces a Python wheel.
# Cargo.lock must be committed for reproducible builds.
FROM python:3.10-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir maturin

WORKDIR /build

# Copy manifests before source so cargo dependency compilation is cached
# independently of application code changes.
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY pyproject.toml ./
COPY python/ ./python/

RUN maturin build --release


# ── Stage: test ───────────────────────────────────────────────────────────────
# Extends builder; runs Rust unit tests at build time (fails fast on regression)
# then exposes pytest as the default command.
FROM builder AS test

RUN cargo test --release
RUN pip install --no-cache-dir pytest

CMD ["pytest", "python/tests/", "-v"]


# ── Stage: runtime ────────────────────────────────────────────────────────────
# Minimal image: wheel + Python scripts only.
FROM python:3.10-slim AS runtime

WORKDIR /app

COPY --from=builder /build/target/wheels/*.whl ./
RUN pip install --no-cache-dir *.whl tminterface && rm *.whl

COPY python/ ./python/
COPY scripts/ ./scripts/

# Override with TMI_HOST=host.docker.internal when running from Docker Desktop.
# On Linux, docker-compose.yml provides extra_hosts so the name resolves.
ENV TMI_HOST=host.docker.internal

CMD ["python", "python/runner.py"]
