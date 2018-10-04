FROM tensorflow/tensorflow:nightly-devel-gpu-py3

RUN apt update && \
    apt-get install python3-matplotlib -y && \
	pip install --upgrade pip

ARG user_id
RUN useradd --uid $user_id --group video --shell /bin/bash --create-home patrick
USER patrick

WORKDIR /home/patrick
RUN mkdir /home/patrick/src
WORKDIR /home/patrick/src

COPY --chown=patrick:users requirements.txt /home/patrick/src

ENV path="/home/patrick/.local/bin/:$PATH"
RUN pip install -r requirements.txt --user

VOLUME ["/home/patrick/src"]

COPY --chown=patrick:users docker-entrypoint.sh /home/patrick
RUN chmod 755 ~/docker-entrypoint.sh
ENTRYPOINT ["/home/patrick/docker-entrypoint.sh"]
