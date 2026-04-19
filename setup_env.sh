#!/bin/bash
# Source this file to activate the codeit environment:
#   source setup_env.sh
module unload anaconda3/2025.06 2>/dev/null
unset CPATH LIBRARY_PATH
export LD_LIBRARY_PATH=/scratch/cy2941/codeit_env/lib
export PATH=/scratch/cy2941/codeit_env/bin:$PATH
export PYTHONPATH=/home/cy2941/codeit:${PYTHONPATH:-}
echo "codeit env activated. python: $(which python)"
