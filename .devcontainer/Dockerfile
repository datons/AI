FROM mambaorg/micromamba:latest

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN micromamba shell init -s zsh

USER root

WORKDIR /workspace

COPY environment.yml ./

RUN micromamba install --file environment.yml --name base && \
    micromamba clean --all --yes

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    zsh \
    direnv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure direnv hook for zsh
RUN echo 'eval "$(direnv hook zsh)"' >> /root/.zshrc && \
    echo 'eval "$(direnv hook zsh)"' >> /home/mambauser/.zshrc

CMD ["zsh"]