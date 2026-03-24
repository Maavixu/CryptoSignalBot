#!/bin/bash
# setup_vps.sh — Run once on a fresh Ubuntu 22.04 VPS as root
# Usage: bash setup_vps.sh
set -e

echo "=== CryptoSignalBot VPS Setup ==="

# Update system
apt update && apt upgrade -y
apt install -y python3.11 python3.11-venv python3-pip git nginx ufw curl

# Create bot user
if ! id "botuser" &>/dev/null; then
    adduser --disabled-password --gecos "" botuser
    echo "botuser created"
fi

# Set up project directory
mkdir -p /home/botuser/crypto_bot
chown -R botuser:botuser /home/botuser/crypto_bot

echo ""
echo "=== Next steps ==="
echo "1. Upload project files to /home/botuser/crypto_bot/"
echo "   scp -r crypto_bot/* botuser@YOUR_IP:/home/botuser/crypto_bot/"
echo ""
echo "2. Switch to botuser and install dependencies:"
echo "   su - botuser"
echo "   cd /home/botuser/crypto_bot"
echo "   python3.11 -m venv .venv"
echo "   source .venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Install systemd service:"
echo "   sudo cp deploy/cryptobot.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable cryptobot"
echo "   sudo systemctl start cryptobot"
echo ""
echo "4. Configure firewall:"
echo "   sudo ufw allow 22/tcp"
echo "   sudo ufw allow 8000/tcp"
echo "   sudo ufw enable"
echo ""
echo "5. (Optional) Set up nginx + HTTPS:"
echo "   sudo cp deploy/nginx.conf /etc/nginx/sites-available/cryptobot"
echo "   sudo ln -s /etc/nginx/sites-available/cryptobot /etc/nginx/sites-enabled/"
echo "   sudo nginx -t && sudo systemctl reload nginx"
echo "   sudo apt install -y certbot python3-certbot-nginx"
echo "   sudo certbot --nginx -d your-domain.com"
echo ""
echo "Setup complete."
