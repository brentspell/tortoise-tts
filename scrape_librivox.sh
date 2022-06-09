#!/bin/bash

PART_COUNT=8

tmux new-session -d -s scrape-librivox
tmux split-window -h
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 2
tmux split-window -v
tmux select-pane -t 4
tmux split-window -v
tmux select-pane -t 4
tmux split-window -v
tmux select-pane -t 6
tmux split-window -v

for i in $(seq 0 $(($PART_COUNT - 1))); do
   tmux select-pane -t $i
   tmux send-keys "python scrape_librivox.py $i $PART_COUNT" 'C-m'
done

exec tmux -2 attach-session -t scrape-librivox
