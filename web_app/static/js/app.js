/**
 * DRIVER GUARDIAN - High-Tech Automotive Cockpit Application Logic
 * SOTA Low-Light Drowsiness Detection & Explainable AI Engine
 */

class DriverGuardianApp {
  constructor() {
    // Mode & State
    this.mode = 'simulation'; // 'simulation' | 'webcam' | 'upload'
    this.activeScenario = 'normal';
    this.isMonitoring = true;
    this.audioEnabled = true;
    this.lastAlarmLevel = 0;
    this.lastSoundTime = 0;
    this.yawnCount = 0;
    this.inYawnState = false;

    // Web Audio Synthesizer
    this.audioCtx = null;
    this.activeAlarmOscillator = null;

    // DOM Elements
    this.canvas = document.getElementById('hudCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.videoEl = document.getElementById('webcamVideo');
    this.criticalStrobe = document.getElementById('criticalStrobe');
    
    // Status & Pill Elements
    this.alarmPill = document.getElementById('alarmPill');
    this.alarmStatusText = document.getElementById('alarmStatusText');
    this.quickClass = document.getElementById('quickClass');
    this.quickFps = document.getElementById('quickFps');
    this.bufferCount = document.getElementById('bufferCount');
    this.deviceText = document.getElementById('deviceText');

    // Gauges Elements
    this.gaugeProgressArc = document.getElementById('gaugeProgressArc');
    this.fatigueValue = document.getElementById('fatigueValue');
    this.fatigueTierTag = document.getElementById('fatigueTierTag');
    
    this.stateIcon = document.getElementById('stateIcon');
    this.stateTitle = document.getElementById('stateTitle');
    this.stateDescription = document.getElementById('stateDescription');
    this.headPitchVal = document.getElementById('headPitchVal');
    this.headPoseStatus = document.getElementById('headPoseStatus');

    this.perclosBar = document.getElementById('perclosBar');
    this.perclosValue = document.getElementById('perclosValue');
    this.earVal = document.getElementById('earVal');
    this.closureDurationVal = document.getElementById('closureDurationVal');

    this.marBar = document.getElementById('marBar');
    this.marValue = document.getElementById('marValue');
    this.oralStatusVal = document.getElementById('oralStatusVal');
    this.yawnCountVal = document.getElementById('yawnCountVal');

    // XAI Elements
    this.gradCamImg = document.getElementById('gradCamImg');
    this.llformerTabImg = document.getElementById('llformerTabImg');
    this.llformerSplit = document.getElementById('llformerSplit');
    this.llformerImg = document.getElementById('llformerImg');
    this.toggleEnhance = document.getElementById('toggleEnhance');
    this.toggleMesh = document.getElementById('toggleMesh');
    this.temporalCanvas = document.getElementById('temporalTimelineCanvas');
    this.temporalCtx = this.temporalCanvas.getContext('2d');
    this.sparklineCanvas = document.getElementById('telemetrySparkline');
    this.sparklineCtx = this.sparklineCanvas.getContext('2d');

    // Receipts & Incidents
    this.receiptTierBadge = document.getElementById('receiptTierBadge');
    this.receiptTime = document.getElementById('receiptTime');
    this.receiptTitle = document.getElementById('receiptTitle');
    this.receiptReasons = document.getElementById('receiptReasons');
    this.receiptAction = document.getElementById('receiptAction');
    this.receiptAuditId = document.getElementById('receiptAuditId');
    this.incidentsList = document.getElementById('incidentsList');
    this.incidentCounter = document.getElementById('incidentCounter');

    // Offscreen Canvas for Webcam Encoding
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = 640;
    this.offscreenCanvas.height = 480;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');

    // Rolling Telemetry Array
    this.telemetryHistory = [];

    this.init();
  }

  async init() {
    this.bindEvents();
    await this.fetchSystemStatus();
    this.startSimulationLoop();
    this.fetchTelemetryPeriodically();
  }

  // =========================================================================
  // Event Listeners
  // =========================================================================
  bindEvents() {
    // Mode Switch Buttons
    document.getElementById('modeSimBtn').addEventListener('click', () => this.setMode('simulation'));
    document.getElementById('modeWebcamBtn').addEventListener('click', () => this.setMode('webcam'));
    document.getElementById('modeUploadBtn').addEventListener('click', () => this.openUploadModal());

    // Scenario Selection Pills
    document.querySelectorAll('.scenario-pill').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.scenario-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.switchScenario(btn.getAttribute('data-scenario'));
      });
    });

    // Model Switch Dropdown
    document.getElementById('modelSelect').addEventListener('change', (e) => {
      this.switchModel(e.target.value);
    });

    // Audio Mute Toggle
    document.getElementById('audioToggleBtn').addEventListener('click', () => {
      this.toggleAudio();
    });

    // Test Siren Button
    document.getElementById('testAlarmBtn').addEventListener('click', () => {
      this.initAudioContext();
      this.playAlarmTone(3, true);
    });

    // Snapshot & XAI Button
    document.getElementById('snapshotBtn').addEventListener('click', () => {
      this.triggerSnapshotXAI();
    });

    // Export Report Button
    document.getElementById('exportReportBtn').addEventListener('click', () => {
      this.exportReport();
    });

    // Refresh XAI
    document.getElementById('refreshXaiBtn').addEventListener('click', () => {
      this.triggerSnapshotXAI();
    });

    // XAI Tabs
    document.querySelectorAll('.xai-tab').forEach(tabBtn => {
      tabBtn.addEventListener('click', () => {
        document.querySelectorAll('.xai-tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.xai-tab-content').forEach(c => c.classList.remove('active'));
        tabBtn.classList.add('active');
        const targetId = `tab-${tabBtn.getAttribute('data-tab')}`;
        document.getElementById(targetId)?.classList.add('active');
      });
    });

    // LLFormer Split Toggle
    this.toggleEnhance.addEventListener('change', (e) => {
      if (e.target.checked) {
        this.llformerSplit.classList.remove('hidden');
      } else {
        this.llformerSplit.classList.add('hidden');
      }
    });

    // Upload Modal Handling
    const modal = document.getElementById('uploadModal');
    document.getElementById('closeModalBtn').addEventListener('click', () => {
      modal.classList.add('hidden');
    });
    document.getElementById('browseFileBtn').addEventListener('click', () => {
      document.getElementById('mediaFileInput').click();
    });
    document.getElementById('mediaFileInput').addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.handleFileUpload(e.target.files[0]);
      }
    });

    // Drag & drop on dropZone
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-hover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-hover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-hover');
      if (e.dataTransfer.files.length > 0) {
        this.handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }

  // =========================================================================
  // Web Audio Synthesizer: 3-Tier Multi-Frequency Acoustic Alarms
  // =========================================================================
  initAudioContext() {
    if (!this.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtx();
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  toggleAudio() {
    this.audioEnabled = !this.audioEnabled;
    const textEl = document.getElementById('audioStateText');
    const btn = document.getElementById('audioToggleBtn');
    if (this.audioEnabled) {
      textEl.innerText = 'SIREN: ON';
      btn.querySelector('.btn-icon').innerText = '🔊';
      this.initAudioContext();
    } else {
      textEl.innerText = 'SIREN: MUTED';
      btn.querySelector('.btn-icon').innerText = '🔇';
      this.stopActiveAlarm();
    }
  }

  stopActiveAlarm() {
    if (this.activeAlarmOscillator) {
      try {
        this.activeAlarmOscillator.stop();
      } catch (e) {}
      this.activeAlarmOscillator = null;
    }
  }

  playAlarmTone(level, force = false) {
    if (!this.audioEnabled && !force) return;
    this.initAudioContext();
    if (!this.audioCtx) return;

    const now = this.audioCtx.currentTime;
    // Throttle repeat sounds within 1.5s unless critical level 3
    if (!force && level < 3 && (Date.now() - this.lastSoundTime < 1500)) return;
    this.lastSoundTime = Date.now();

    if (level === 1) {
      // Tier 1 Caution: Gentle chime (440Hz -> 660Hz)
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(660, now + 0.15);
      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.35);
    } else if (level === 2) {
      // Tier 2 Warning: Pulsed cautionary double-beep (880Hz)
      [0, 0.15].forEach(delay => {
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = 'square';
        osc.frequency.setValueAtTime(880, now + delay);
        gain.gain.setValueAtTime(0.25, now + delay);
        gain.gain.exponentialRampToValueAtTime(0.01, now + delay + 0.10);
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(now + delay);
        osc.stop(now + delay + 0.10);
      });
    } else if (level === 3) {
      // Tier 3 Critical: Emergency Warble Siren (1600Hz <-> 2400Hz)
      this.stopActiveAlarm();
      const osc = this.audioCtx.createOscillator();
      const mod = this.audioCtx.createOscillator();
      const modGain = this.audioCtx.createGain();
      const masterGain = this.audioCtx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(1800, now);

      // Modulator for siren warble effect
      mod.type = 'sine';
      mod.frequency.setValueAtTime(6.0, now); // 6 Hz warble speed
      modGain.gain.setValueAtTime(500, now);

      mod.connect(osc.frequency);
      masterGain.gain.setValueAtTime(0.45, now);
      masterGain.gain.exponentialRampToValueAtTime(0.01, now + 1.2);

      osc.connect(masterGain);
      masterGain.connect(this.audioCtx.destination);

      mod.start(now);
      osc.start(now);
      mod.stop(now + 1.2);
      osc.stop(now + 1.2);
      this.activeAlarmOscillator = osc;
    }
  }

  // =========================================================================
  // Mode Controller: Simulation vs Live Camera vs Upload
  // =========================================================================
  async setMode(newMode) {
    this.mode = newMode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    const pillGroup = document.getElementById('scenarioPills');

    if (newMode === 'simulation') {
      document.getElementById('modeSimBtn').classList.add('active');
      pillGroup.style.display = 'flex';
      this.stopWebcam();
      this.startSimulationLoop();
    } else if (newMode === 'webcam') {
      document.getElementById('modeWebcamBtn').classList.add('active');
      pillGroup.style.display = 'none';
      this.stopSimulationLoop();
      await this.startWebcam();
    }
  }

  async switchScenario(scenarioName) {
    this.activeScenario = scenarioName;
    try {
      await fetch('/api/simulation/set_scenario', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenarioName })
      });
    } catch (e) {
      console.warn('Could not switch scenario:', e);
    }
  }

  async switchModel(modelId) {
    try {
      const resp = await fetch('/api/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });
      const data = await resp.json();
      if (data.status === 'success') {
        this.alarmStatusText.innerText = `MODEL SWITCHED: ${modelId.toUpperCase()}`;
      }
    } catch (e) {
      console.error('Model switch failed:', e);
    }
  }

  // =========================================================================
  // Simulation Loop
  // =========================================================================
  startSimulationLoop() {
    this.isMonitoring = true;
    const loop = async () => {
      if (!this.isMonitoring || this.mode !== 'simulation') return;
      try {
        const resp = await fetch('/api/simulation/frame');
        const data = await resp.json();
        if (data.image) {
          const img = new Image();
          img.onload = () => {
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            this.drawHudOverlay(data.metrics);
            this.updateTelemetry(data.metrics);
          };
          img.src = data.image;
        }
      } catch (err) {
        console.warn('Simulation frame fetch error:', err);
      }
      setTimeout(loop, 45); // ~22 FPS
    };
    loop();
  }

  stopSimulationLoop() {
    this.isMonitoring = false;
  }

  // =========================================================================
  // Live Browser WebCam Pipeline (High-Performance Decoupled Architecture)
  // =========================================================================
  async startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 30, max: 60 } },
        audio: false
      });
      this.videoEl.srcObject = stream;
      await this.videoEl.play();

      this.isMonitoring = true;
      this.isProcessingFrame = false;

      // Downsampled offscreen canvas for fast transmission (320x240 = 4x lighter)
      this.offscreenCanvas.width = 320;
      this.offscreenCanvas.height = 240;

      // 1. Decoupled 60 FPS Native Camera Display Loop (Zero Lag)
      const renderLoop = () => {
        if (!this.isMonitoring || this.mode !== 'webcam') return;
        if (this.videoEl.readyState >= 2) {
          this.ctx.drawImage(this.videoEl, 0, 0, this.canvas.width, this.canvas.height);
          if (this.lastMetrics) {
            this.drawHudOverlay(this.lastMetrics);
          }
        }
        requestAnimationFrame(renderLoop);
      };
      requestAnimationFrame(renderLoop);

      // 2. Background Asynchronous AI Inference Loop (Non-blocking)
      const inferenceLoop = async () => {
        if (!this.isMonitoring || this.mode !== 'webcam') return;

        if (!this.isProcessingFrame && this.videoEl.readyState >= 2) {
          this.isProcessingFrame = true;
          this.offscreenCtx.drawImage(this.videoEl, 0, 0, 320, 240);
          const b64 = this.offscreenCanvas.toDataURL('image/jpeg', 0.60);

          try {
            const resp = await fetch('/api/process_frame', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image: b64, request_xai: false })
            });
            const res = await resp.json();
            if (res.status === 'success') {
              const m = res.metrics;
              // Rescale bounding box & landmark coordinates (320x240 -> 640x480)
              if (m.bbox) {
                m.bbox = [m.bbox[0] * 2, m.bbox[1] * 2, m.bbox[2] * 2, m.bbox[3] * 2];
              }
              if (m.landmarks) {
                m.landmarks = m.landmarks.map(pt => ({
                  ...pt,
                  x: pt.x * 2,
                  y: pt.y * 2
                }));
              }
              this.lastMetrics = m;
              this.updateTelemetry(m);
            }
          } catch (e) {
            console.warn('Webcam inference notice:', e);
          } finally {
            this.isProcessingFrame = false;
          }
        }

        setTimeout(inferenceLoop, 80); // ~12 FPS AI refresh cadence
      };
      inferenceLoop();
    } catch (err) {
      alert(`Camera Access Error: ${err.message}. Falling back to Simulated Drive.`);
      this.setMode('simulation');
    }
  }

  stopWebcam() {
    this.isProcessingFrame = false;
    if (this.videoEl.srcObject) {
      this.videoEl.srcObject.getTracks().forEach(track => track.stop());
      this.videoEl.srcObject = null;
    }
  }

  // =========================================================================
  // HUD Canvas Overlay Rendering (Face Boxes, Mesh, Crosshairs)
  // =========================================================================
  drawHudOverlay(metrics) {
    if (!metrics) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // 1. Alert Level Strobe & Border Color
    let hudColor = '#00e676'; // Safe Green
    if (metrics.alarm_level === 3) {
      hudColor = '#ff1744'; // Danger Red
      this.criticalStrobe.classList.add('active');
    } else {
      this.criticalStrobe.classList.remove('active');
      if (metrics.alarm_level === 2) hudColor = '#ff6d00'; // Warning Orange
      else if (metrics.alarm_level === 1) hudColor = '#ffab00'; // Caution Yellow
    }

    // 2. Draw Facial Bounding Box with Cyberpunk Corner Accents
    if (metrics.bbox) {
      const [bx, by, bw, bh] = metrics.bbox;
      ctx.strokeStyle = hudColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);

      // Corner Brackets
      const cLen = 14;
      ctx.lineWidth = 4;
      ctx.beginPath();
      // Top Left
      ctx.moveTo(bx, by + cLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cLen, by);
      // Top Right
      ctx.moveTo(bx + bw - cLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cLen);
      // Bottom Left
      ctx.moveTo(bx, by + bh - cLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cLen, by + bh);
      // Bottom Right
      ctx.moveTo(bx + bw - cLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cLen);
      ctx.stroke();

      // Face Tag
      ctx.fillStyle = hudColor;
      ctx.fillRect(bx, by - 22, 120, 22);
      ctx.fillStyle = '#070a13';
      ctx.font = 'bold 11px Rajdhani, sans-serif';
      ctx.fillText(`DRIVER [${metrics.class_name.toUpperCase()}]`, bx + 6, by - 7);
    }

    // 3. Draw Landmark Points & Mesh
    if (this.toggleMesh.checked && metrics.landmarks) {
      ctx.fillStyle = '#00f0ff';
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
      ctx.lineWidth = 1;

      // Draw connections between landmarks
      ctx.beginPath();
      for (let i = 0; i < metrics.landmarks.length; i++) {
        const pt = metrics.landmarks[i];
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();

      // Draw Keypoint Nodes
      metrics.landmarks.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // 4. Cockpit Compass & Crosshairs
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2 - 20, h / 2); ctx.lineTo(w / 2 + 20, h / 2);
    ctx.moveTo(w / 2, h / 2 - 20); ctx.lineTo(w / 2, h / 2 + 20);
    ctx.stroke();
  }

  // =========================================================================
  // Telemetry Dashboard & Biometrics Synchronization
  // =========================================================================
  updateTelemetry(metrics) {
    if (!metrics) return;

    // Status Pill
    this.alarmStatusText.innerText = metrics.status_text;
    this.alarmPill.className = `status-pill level-${metrics.alarm_level}`;
    this.quickClass.innerText = metrics.class_name;
    this.quickFps.innerText = metrics.fps.toFixed(1);

    // Audio Alert Trigger
    if (metrics.alarm_level > 0 && metrics.alarm_level !== this.lastAlarmLevel) {
      this.playAlarmTone(metrics.alarm_level);
    }
    this.lastAlarmLevel = metrics.alarm_level;

    // Radial Fatigue Gauge
    const fatiguePct = metrics.smoothed_fatigue * 100;
    this.fatigueValue.innerText = fatiguePct.toFixed(1);
    this.fatigueTierTag.innerText = `TIER ${metrics.alarm_level}`;

    // SVG Arc Dashoffset (251.3 total circumference for semi-circle arc)
    const arcOffset = 251.3 - (251.3 * (metrics.smoothed_fatigue || 0));
    this.gaugeProgressArc.style.strokeDashoffset = Math.max(0, arcOffset);

    // Dynamic Arc Color Shift
    if (metrics.alarm_level === 3) this.gaugeProgressArc.style.stroke = '#ff1744';
    else if (metrics.alarm_level === 2) this.gaugeProgressArc.style.stroke = '#ff6d00';
    else if (metrics.alarm_level === 1) this.gaugeProgressArc.style.stroke = '#ffab00';
    else this.gaugeProgressArc.style.stroke = '#00e676';

    // Behavioral State Card
    const stateIcons = {
      0: '🟢',
      1: '🟡',
      2: '🥱',
      3: '🔻',
      4: '🚨'
    };
    this.stateIcon.innerText = stateIcons[metrics.predicted_class] || '🟢';
    this.stateTitle.innerText = metrics.class_name.toUpperCase();

    const descriptions = {
      0: 'Eyes tracking road ahead. Normal blinking rate.',
      1: 'Slow eyelid closure velocity detected. Early fatigue.',
      2: 'Frequent oral cavity expansion (Yawning detected).',
      3: 'Downward head pitch acceleration (Nodding / Head drop).',
      4: 'Sustained eye closure (>1.5s). CRITICAL MICROSLEEP EVENT!'
    };
    this.stateDescription.innerText = descriptions[metrics.predicted_class] || 'Attentive driving.';
    this.headPitchVal.innerText = `${metrics.head_pitch.toFixed(1)}°`;
    this.headPoseStatus.innerText = Math.abs(metrics.head_pitch) > 16 ? 'HEAD NOD' : 'UPRIGHT';

    // PERCLOS & Eye Dynamics
    const perclosPct = metrics.perclos * 100;
    this.perclosValue.innerText = `${perclosPct.toFixed(1)}%`;
    this.perclosBar.style.width = `${Math.min(100, perclosPct * 3.5)}%`;
    this.earVal.innerText = metrics.ear.toFixed(2);
    this.closureDurationVal.innerText = `${metrics.closure_duration.toFixed(2)}s`;
    if (metrics.closure_duration > 1.5) {
      this.closureDurationVal.style.color = '#ff1744';
    } else {
      this.closureDurationVal.style.color = '#00f0ff';
    }

    // Mouth & Yawn Dynamics
    this.marValue.innerText = metrics.mar.toFixed(2);
    this.marBar.style.width = `${Math.min(100, metrics.mar * 120)}%`;
    if (metrics.mar > 0.55) {
      this.oralStatusVal.innerText = 'YAWNING';
      this.oralStatusVal.style.color = '#ff6d00';
      if (!this.inYawnState) {
        this.yawnCount++;
        this.yawnCountVal.innerText = this.yawnCount;
        this.inYawnState = true;
      }
    } else {
      this.oralStatusVal.innerText = 'Closed';
      this.oralStatusVal.style.color = '#cbd5e1';
      this.inYawnState = false;
    }

    // Safety Receipt Update
    this.receiptTierBadge.innerText = `TIER ${metrics.alarm_level} - ${metrics.status_text.toUpperCase()}`;
    this.receiptTime.innerText = new Date().toLocaleTimeString();
    this.receiptTitle.innerText = `STATUS: ${metrics.class_name.toUpperCase()}`;
    if (metrics.alarm_level >= 2) {
      this.receiptAction.innerText = 'Immediate Pull-Over & Driver Rotation Required.';
      this.receiptAction.style.color = '#ff1744';
    } else {
      this.receiptAction.innerText = 'Maintain safe speed & continue attentive monitoring.';
      this.receiptAction.style.color = '#00e676';
    }
  }

  // =========================================================================
  // Multi-Modal Explainable AI (XAI) Studio
  // =========================================================================
  async triggerSnapshotXAI() {
    try {
      this.alarmStatusText.innerText = 'COMPUTING XAI ATTRIBUTIONS...';
      const resp = await fetch('/api/diagnose_snapshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: null })
      });
      const data = await resp.json();
      if (data.xai) {
        this.renderXaiData(data.xai);
        this.alarmStatusText.innerText = 'XAI DIAGNOSIS CARD GENERATED';
      }
    } catch (err) {
      console.error('XAI Snapshot trigger failed:', err);
    }
  }

  renderXaiData(xai) {
    if (!xai) return;

    // 1. Grad-CAM Overlay
    if (xai.grad_cam_b64) {
      this.gradCamImg.src = `data:image/jpeg;base64,${xai.grad_cam_b64}`;
    }

    // 2. LLFormer Restoration
    if (xai.llformer_enhanced_b64) {
      this.llformerTabImg.src = `data:image/jpeg;base64,${xai.llformer_enhanced_b64}`;
      this.llformerImg.src = `data:image/jpeg;base64,${xai.llformer_enhanced_b64}`;
    }

    // 3. Regional SHAP Values
    const shap = xai.shap_attribution || {};
    const le = shap['Left Eye Region'] || 38.5;
    const re = shap['Right Eye Region'] || 34.2;
    const m = shap['Mouth / Yawning'] || 18.1;
    const p = shap['Head Pose & Motion'] || 9.2;

    document.getElementById('shapLeftEye').innerText = `${le.toFixed(1)}%`;
    document.getElementById('shapLeftEyeBar').style.width = `${le}%`;
    document.getElementById('shapRightEye').innerText = `${re.toFixed(1)}%`;
    document.getElementById('shapRightEyeBar').style.width = `${re}%`;
    document.getElementById('shapMouth').innerText = `${m.toFixed(1)}%`;
    document.getElementById('shapMouthBar').style.width = `${m}%`;
    document.getElementById('shapPose').innerText = `${p.toFixed(1)}%`;
    document.getElementById('shapPoseBar').style.width = `${p}%`;

    // 4. Temporal Sequence Canvas
    const temp = xai.temporal_behavior || {};
    const probs = temp.drowsiness_probabilities || [0.1, 0.15, 0.2, 0.35, 0.5, 0.7, 0.85];
    this.drawTemporalCurve(probs);

    // 5. Structured Alarm Card
    const card = xai.alarm_card || {};
    if (card.primary_reasons && card.primary_reasons.length > 0) {
      this.receiptReasons.innerHTML = card.primary_reasons
        .map(r => `<li>${r.charAt(0).toUpperCase() + r.slice(1)}</li>`)
        .join('');
    }
  }

  drawTemporalCurve(probabilities) {
    const ctx = this.temporalCtx;
    const w = this.temporalCanvas.width;
    const h = this.temporalCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background Grid
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let y = 20; y < h; y += 30) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Critical Threshold Line (at 85%)
    const critY = h - (h * 0.85);
    ctx.strokeStyle = 'rgba(255, 23, 68, 0.5)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, critY); ctx.lineTo(w, critY); ctx.stroke();
    ctx.setLineDash([]);

    // Plot Temporal Confidence Curve
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();

    const step = w / (probabilities.length - 1);
    probabilities.forEach((prob, idx) => {
      const x = idx * step;
      const y = h - (prob * (h - 15)) - 8;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill Gradient under curve
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
    grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // =========================================================================
  // Telemetry Sparkline & Incident Audit Feed
  // =========================================================================
  async fetchTelemetryPeriodically() {
    const fetchLoop = async () => {
      try {
        const [telResp, incResp] = await Promise.all([
          fetch('/api/telemetry'),
          fetch('/api/incidents')
        ]);
        const telData = await telResp.json();
        const incData = await incResp.json();

        if (telData.telemetry) {
          this.drawSparkline(telData.telemetry);
        }
        if (incData.incidents) {
          this.renderIncidents(incData.incidents);
        }
      } catch (e) {
        // silent background poll
      }
      setTimeout(fetchLoop, 3000);
    };
    fetchLoop();
  }

  drawSparkline(telemetry) {
    if (!telemetry || telemetry.length < 2) return;
    const ctx = this.sparklineCtx;
    const w = this.sparklineCanvas.width;
    const h = this.sparklineCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const step = w / (telemetry.length - 1);
    telemetry.forEach((pt, idx) => {
      const x = idx * step;
      const y = h - (pt.fatigue / 100 * (h - 6)) - 3;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  renderIncidents(incidents) {
    this.incidentCounter.innerText = `${incidents.length} EVENTS`;
    if (incidents.length === 0) return;

    this.incidentsList.innerHTML = incidents.slice(-8).reverse().map(inc => {
      const badgeClass = inc.level === 3 ? 'danger-red' : (inc.level === 2 ? 'warning-orange' : 'caution-yellow');
      return `
        <div class="incident-entry level-${inc.level}">
          <div class="incident-entry-info">
            <span class="incident-timestamp">${inc.timestamp}</span>
            <span class="incident-msg">${inc.status}</span>
          </div>
          <span class="incident-badge" style="background: rgba(255,255,255,0.1); color: var(--${badgeClass});">
            ${inc.fatigue_score}% FATIGUE
          </span>
        </div>
      `;
    }).join('');
  }

  // =========================================================================
  // Media Upload & Report Export
  // =========================================================================
  openUploadModal() {
    document.getElementById('uploadModal').classList.remove('hidden');
  }

  async handleFileUpload(file) {
    const progress = document.getElementById('uploadProgress');
    progress.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch('/api/upload_media', {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();
      if (data.status === 'success') {
        progress.classList.add('hidden');
        document.getElementById('uploadModal').classList.add('hidden');
        this.updateTelemetry(data.metrics);
        if (data.xai) {
          this.renderXaiData(data.xai);
        }
        alert(`Analysis Complete for ${file.name}! Check the XAI Studio panel.`);
      }
    } catch (err) {
      alert(`Upload error: ${err.message}`);
      progress.classList.add('hidden');
    }
  }

  async exportReport() {
    try {
      const resp = await fetch('/api/export_report');
      const data = await resp.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `driver_guardian_audit_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert('Report export failed: ' + e.message);
    }
  }

  async fetchSystemStatus() {
    try {
      const resp = await fetch('/api/status');
      const data = await resp.json();
      this.deviceText.innerText = `DEVICE: ${data.gpu_available ? data.gpu_name : 'CPU'}`;
    } catch (e) {
      console.warn('System status fetch failed:', e);
    }
  }
}

// Instantiate App once DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
  window.driverApp = new DriverGuardianApp();
});
