## Deploying & Testing New Code

### One-time setup (editable install)
Do this once after cloning. It makes git pull immediately effective without reinstalling.
```bash
source /venvs/mini_daemon/bin/activate
sudo /venvs/mini_daemon/bin/pip install -e /home/pollen/reachy-submodule
```

### Normal deploy cycle
```bash
# on the robot:
git pull
sudo systemctl restart reachy-mini-daemon
# wait ~10s for the daemon to initialize, then wake it up:
curl -X POST 'http://localhost:8000/api/daemon/start?wake_up=true'
```

### how to activate the venv or install something
source /venvs/mini_daemon/bin/activate
sudo /venvs/mini_daemon/bin/pip install aiortc


### Verify WebRTC is running
```bash
ss -tlnp | grep 8443        # should show 0.0.0.0:8443
ps aux | grep webrtc        # should show a python process
```

### When things go wrong

#### Check journal logs to find errors with the reachy-mini-daemon
```sudo journalctl -u reachy-mini-daemon -f```

**Service fails to start (status=203/EXEC)**
The service file has a hardcoded path that goes stale. Fix it permanently:
```bash
# The source service file is the one that gets deployed on every start.
# Edit it directly — don't edit /etc/systemd/system/ (launcher.sh overwrites it):
nano /home/pollen/reachy-submodule/src/reachy_mini/daemon/app/services/wireless/reachy-mini-daemon.service
# Set ExecStart and WorkingDirectory to the /home/pollen/reachy-submodule/src/... path
sudo systemctl daemon-reload
sudo systemctl start reachy-mini-daemon
```

**Service hangs on stop (takes 90s)**
GStreamer webrtcsink blocks during teardown. Force it faster:
```bash
sudo systemctl kill reachy-mini-daemon   # immediate, skips teardown
sudo systemctl start reachy-mini-daemon
```
To make it stop quickly by default, add `TimeoutStopSec=10` to the service file.

**Port 8000 not up after start**
The daemon does a pip reinstall on first run after certain changes — takes ~30s.
Wait and retry:
```bash
watch -n2 'curl -s http://localhost:8000/api/daemon/status'
```

**Port 8443 (WebRTC) not up after wake**
Wake_up hasn't started the GStreamer pipeline yet, or it crashed. Check:
```bash
sudo journalctl -u reachy-mini-daemon -n 80 --no-pager
```

**Corrupted pip packages (warnings about ~eachy-mini, ~yee, etc.)**
Previous pip installs were interrupted. Clean up:
```bash
sudo rm -rf /venvs/mini_daemon/lib/python3.12/site-packages/\~*
sudo /venvs/mini_daemon/bin/pip install -e /home/pollen/reachy-submodule
```




### Quick Look
After [installing the SDK](https://huggingface.co/docs/reachy_mini/SDK/installation), once your robot is awake, you can control it in just **a few lines of code**:

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # Look up and tilt head
    mini.goto_target(
        head=create_head_pose(z=10, roll=15, degrees=True, mm=True),
        duration=1.0
    )
```

<br>

## 🛠 Hardware Overview

Reachy Mini robots are sold as kits and generally take **2 to 3 hours** to assemble. Detailed step-by-step guides are available in the platform-specific folders linked above.

* **Reachy Mini (Wireless):** Runs onboard (RPi 4), autonomous, includes IMU. [See specs](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/hardware).
* **Reachy Mini Lite:** Runs on your PC, powered via wall outlet. [See specs](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/hardware).

<br>
