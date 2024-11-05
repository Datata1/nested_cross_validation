#!/bin/bash

SESSION_NAME="JankaSession"
ROOT_DIR="$HOME/Documents/Janka"

tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME -c "$ROOT_DIR"
    
    tmux send-keys -t $SESSION_NAME "htop" C-m
    
    tmux split-window -v -t $SESSION_NAME -c "$ROOT_DIR"
    tmux send-keys -t $SESSION_NAME "cd $ROOT_DIR && pipenv run prefect server start --port 3000" C-m
    
    tmux split-window -h -t $SESSION_NAME:0.1 -c "$ROOT_DIR"
    tmux send-keys -t $SESSION_NAME "cd $ROOT_DIR && pipenv run jupyter lab" C-m
    
    tmux select-pane -t $SESSION_NAME:0.0
fi

tmux attach -t $SESSION_NAME
