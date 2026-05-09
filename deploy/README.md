# Deploy FastAPI master API (`master.scheduler`) to AWS VPS

## What CI/CD does

GitHub Actions workflow [`.github/workflows/deploy-master-api.yml`](../.github/workflows/deploy-master-api.yml):

1. **CI**: installs [`deploy/requirements-api.txt`](requirements-api.txt), runs `compileall`, imports `master.scheduler`.
2. **CD** (on `main`/`master` push or **workflow_dispatch**): **rsync** repo to `VPS_DEPLOY_PATH` (default `/opt/cse354-api`), creates **venv**, **pip install**, **`systemctl restart cse354-master-api`**.

The API listens on **`127.0.0.1:8000`**; [**nginx**](../nginx/nginx.conf) on the same host should reverse-proxy **HTTPS → upstream**.

## One-time VPS setup

1. **Create deploy directory** (as root or with sudo):

   ```bash
   sudo mkdir -p /opt/cse354-api
   sudo chown "$USER:$USER" /opt/cse354-api
   ```

2. **System user for the service** (matches [`systemd/cse354-master-api.service`](systemd/cse354-master-api.service)):

   ```bash
   sudo useradd --system --home /opt/cse354-api --shell /bin/bash cse354 || true
   sudo chown -R cse354:cse354 /opt/cse354-api
   ```

3. **Install systemd unit**:

   ```bash
   sudo cp deploy/systemd/cse354-master-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable cse354-master-api
   ```

4. **Secrets file** (never commit):

   ```bash
   sudo -u cse354 nano /opt/cse354-api/.env
   ```

   Minimal:

   ```env
   REDIS_URL=redis://127.0.0.1:6379/0
   ```

   Ensure Redis is running (`redis-server`) and bound appropriately.

5. **`sudo` for restart**: the deploy user (e.g. `ubuntu`) must restart systemd **without password**:

   ```bash
   echo 'ubuntu ALL=(ALL) NOPASSWD: /bin/systemctl restart cse354-master-api, /bin/systemctl status cse354-master-api' | sudo tee /etc/sudoers.d/cse354-deploy
   sudo chmod 440 /etc/sudoers.d/cse354-deploy
   ```

   Adjust username to match **`VPS_USER`** in GitHub secrets.

## GitHub repository configuration

| Name | Type | Value |
|------|------|--------|
| `VPS_HOST` | Secret | e.g. `ec2-xx-xx-xx-xx.compute.amazonaws.com` |
| `VPS_USER` | Secret | e.g. `ubuntu` |
| `VPS_SSH_KEY` | Secret | Private SSH key PEM for that user |
| `VPS_DEPLOY_PATH` | Variable (optional) | e.g. `/opt/cse354-api` |

Add the **repository public key** to `~/.ssh/authorized_keys` on the VPS if you use a dedicated deploy key pair.

## Manual smoke test on VPS

```bash
curl -s http://127.0.0.1:8000/health
```

Through nginx (after TLS + `proxy_pass`):

```bash
curl -s https://api.yourdomain.com/health
```

## Notes

- Full repo **`requirements.txt`** includes GPU/Ray stacks; production API host should use only **`deploy/requirements-api.txt`** (what CI installs).
- **`main.py`** uses `reload=True` — production should use **systemd + uvicorn** without reload (see unit file).
