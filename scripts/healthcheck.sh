#!/bin/bash
curl -f http://localhost:8000/health || exit 1
echo "CoreC is healthy"