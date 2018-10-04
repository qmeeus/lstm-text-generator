#!/bin/sh

exec tensorboard --logdir ./logs &
exec python3 main.py "$@"
