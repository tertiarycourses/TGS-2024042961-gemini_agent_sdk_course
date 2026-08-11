#!/bin/bash
# Load environment variables from .env and run adk web

set -a  # automatically export all variables
source .env
set +a

adk web "$@"
