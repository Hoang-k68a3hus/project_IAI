#!/usr/bin/env bash
set -euo pipefail

WEB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$WEB_ROOT/server"
CLIENT_DIR="$WEB_ROOT/client"

echo "VieComRec web setup"

echo
echo "Checking prerequisites..."
node --version
npm --version

echo
echo "Preparing env files..."
if [[ ! -f "$SERVER_DIR/.env" && -f "$SERVER_DIR/.env.example" ]]; then
  cp "$SERVER_DIR/.env.example" "$SERVER_DIR/.env"
  echo "Created server/.env from example"
fi

if [[ ! -f "$CLIENT_DIR/.env" && -f "$CLIENT_DIR/.env.example" ]]; then
  cp "$CLIENT_DIR/.env.example" "$CLIENT_DIR/.env"
  echo "Created client/.env from example"
fi

echo
echo "Starting MongoDB with Docker..."
docker compose -f "$WEB_ROOT/docker-compose.yml" up -d mongodb

echo
echo "Installing server dependencies..."
(cd "$SERVER_DIR" && npm install)

echo
echo "Installing client dependencies..."
(cd "$CLIENT_DIR" && npm install)

echo
echo "Setup complete."
echo "Server: cd web/server && npm start"
echo "Client: cd web/client && npm start"
echo "VieComRec API: http://localhost:8000"
echo "Web API: http://localhost:5000/api/health"
echo "Web app: http://localhost:3000"
