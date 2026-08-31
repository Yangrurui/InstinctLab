# syntax=docker/dockerfile:1.7

# The wheel builder is intentionally separate from the simulator image. It
# proves the four published distributions can be built from the checked-out
# repository without inheriting undeclared sibling sources.
ARG INSTINCTLAB_WHEEL_BUILDER_IMAGE
ARG INSTINCTLAB_RUNTIME_IMAGE
FROM ${INSTINCTLAB_WHEEL_BUILDER_IMAGE} AS wheel-builder

ARG INSTINCTLAB_WHEEL_BUILDER_IMAGE
WORKDIR /src/instinctlab
COPY . .
RUN python -c 'import re,sys; assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", sys.argv[1]), "wheel builder must use an immutable digest"' "${INSTINCTLAB_WHEEL_BUILDER_IMAGE}" && \
    python scripts/build_release.py --output /release --expected-version 0.1.0

# Isaac Sim's redistributable/runtime terms and the host driver boundary make
# the simulator layer an externally supplied artifact. The base must already
# contain both locked SDK stacks and /opt/instinctlab-runtime/runtime_provenance.json.
# The final verifier rejects mutable image tags, wrong commits, missing SDKs,
# and package-version drift before the image can be produced.
FROM ${INSTINCTLAB_RUNTIME_IMAGE} AS runtime

ARG INSTINCTLAB_RUNTIME_IMAGE
ARG INSTINCTLAB_RUNTIME_PROVENANCE=/opt/instinctlab-runtime/runtime_provenance.json

LABEL org.opencontainers.image.title="InstinctLab dual-backend operator runtime" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/project-instinct/instinctlab" \
      org.project-instinct.isaaclab.commit="f73c33173801f5f8afea4142482e47b7710c2b75" \
      org.project-instinct.mjlab.commit="08090e8a77228e733373f3b5c54f8b5a68d19d9d"

USER root
COPY --from=wheel-builder /release /opt/instinctlab/release
COPY docker/runtime-lock.json /opt/instinctlab/runtime-lock.json
COPY datasets/manifest.json /opt/instinctlab/dataset-manifest.json
COPY scripts/verify_container_runtime.py /opt/instinctlab/bin/verify_container_runtime.py
COPY scripts/verify_datasets.py /opt/instinctlab/bin/verify_datasets.py
COPY scripts/train.py /opt/instinctlab/bin/train.py
COPY scripts/play.py /opt/instinctlab/bin/play.py

RUN python -m pip install --no-deps /opt/instinctlab/release/*.whl && \
    python /opt/instinctlab/bin/verify_container_runtime.py \
      --lock /opt/instinctlab/runtime-lock.json \
      --runtime-provenance "${INSTINCTLAB_RUNTIME_PROVENANCE}" \
      --wheel-dir /opt/instinctlab/release \
      --base-image "${INSTINCTLAB_RUNTIME_IMAGE}"

ENV INSTINCTLAB_DATA_ROOT=/datasets \
    PYTHONUNBUFFERED=1
WORKDIR /workspace

# Dataset verification is a separate operator action because datasets are
# mounted read-only at runtime, not copied into the image.
CMD ["bash"]
