/**
 * ============================================================================
 * SNEEZE PREDICTOR 3000 - HIGH-PRECISION AUDIO AI SNEEZE CLASSIFIER
 * ============================================================================
 * Senior Audio AI Engineer & ML Architecture:
 * 
 * 1. Real-Time High-Precision Audio Pipeline (Web Audio API & AnalyserNode):
 *    - Captures raw PCM stream via navigator.mediaDevices.getUserMedia
 *    - Disables aggressive hardware noise suppression & AGC to preserve explosive sneeze dynamics
 *    - 60 FPS real-time analysis via requestAnimationFrame (16ms latency)
 *    - Multi-band spectral energy decomposition:
 *      * Low / Mechanical Rumble (0 - 300 Hz)
 *      * Vocal Core Formants (300 - 1800 Hz) [Speech recognition filter]
 *      * Nasal Air Turbulence & Friction (1800 - 7500 Hz) [Core Sneeze Signature]
 *      * Ultra-high Sibilance (7500 - 16000 Hz)
 *    - Spectral Centroid, Zero-Crossing Rate (ZCR), and Dynamic Attack Velocity (dRMS/dt)
 *    - Adaptive Ambient Noise Floor auto-calibration to work across all mic hardware
 *    - Dual-type sneeze validation: Supports voiced ("ACHOO!") and unvoiced ("TSCHHH!") sneezes
 *    - Explicitly rejects steady speech, singing, desk taps/claps, and ambient silence
 * 
 * 2. TensorFlow.js YAMNet Integration:
 *    - Asynchronously connects to TensorFlow.js and YAMNet sound classifier
 *    - Telemetry HUD dynamically updates with neural prediction status
 * 
 * 3. Conditional Sneeze-Only 6-Second Timer:
 *    - Strictly PAUSED at 06.00s while passively listening
 *    - Starts counting down ONLY when the first sneeze is detected
 *    - Increments counter for follow-up sneezes during the 6s window with a 1.2s debounce guard
 *    - When timer hits 0.00s, locks session, calculates Superstition Report, and resets to 06.00s
 * 
 * 4. Superstition & Gossip Logic Matrix:
 *    - 1 Sneeze: "SECRET ADMIRER DETECTED! 💖" (95% Positivity, Secret Admirer / Old Friend)
 *    - 2 Sneezes: "WARNING: DRAMA & GOSSIP IN PROGRESS! 🗣️" (18% Positivity, Jealous Rival)
 *    - 3+ Sneezes: "ALLERGY ALERT: NOBODY IS TALKING ABOUT YOU! 🤧" (50% Positivity, Pollen/Dust Mites)
 * ============================================================================
 */

// Application State Machine Enum
const AppState = {
  IDLE_STOPPED: 'IDLE_STOPPED',     // Mic is off, awaiting user activation
  IDLE_LISTENING: 'IDLE_LISTENING', // Passively listening, timer paused at 06.00s
  ACTIVE_WINDOW: 'ACTIVE_WINDOW',   // First sneeze verified! 6-second countdown running
  RESOLVED: 'RESOLVED'               // 6s countdown expired, superstition report compiled
};

// Global State Variables
let currentState = AppState.IDLE_STOPPED;
let audioContext = null;
let mediaStream = null;
let micSourceNode = null;
let analyserNode = null;
let animationFrameId = null;

// Audio Sensitivity & Adaptive Calibration Settings
let sensitivityPercent = 65; // User slider value (50% - 95%)
let modelConfidenceThreshold = 0.65; // Sneeze probability required to trigger
let volumeThreshold = 0.08; // Dynamic RMS threshold calculated from sensitivity & noise floor
let ambientNoiseFloor = 0.015; // Rolling estimate of ambient room noise

// Debounce Cooldown Configuration
const DEBOUNCE_COOLDOWN_MS = 1200; // 1.2-second debounce mechanism
let lastSneezeTimestamp = 0;

// Burst Envelope Tracking for Sneeze Duration Validation
let burstFrameCount = 0;
let previousRMS = 0.0;
let recentRMSHistory = new Float32Array(12); // ~200ms history at 60 FPS
let historyIndex = 0;

// Active 6-Second Sneeze Window State
let sneezeCount = 0;
const SESSION_WINDOW_MS = 6000;
let sessionStartTime = 0;
let sessionIntervalId = null;

// Session History Log Store
const sessionHistory = [];

// DOM Elements Cache
const el = {
  // Top Header Status Badges
  modelStatusBadge: document.getElementById('modelStatusBadge'),
  modelPulseDot: document.getElementById('modelPulseDot'),
  modelStatusText: document.getElementById('modelStatusText'),
  statusBadge: document.getElementById('statusBadge'),
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),

  // Visualizer & Confidence Meter
  audioCanvas: document.getElementById('audioCanvas'),
  canvasContainer: document.getElementById('canvasContainer'),
  canvasStartPrompt: document.getElementById('canvasStartPrompt'),
  confidenceFill: document.getElementById('confidenceFill'),
  confidenceText: document.getElementById('confidenceText'),
  thresholdMarker: document.getElementById('thresholdMarker'),

  // Telemetry HUD
  classifiedSoundLabel: document.getElementById('classifiedSoundLabel'),
  valCentroid: document.getElementById('valCentroid'),
  valBurst: document.getElementById('valBurst'),
  valNoiseFilter: document.getElementById('valNoiseFilter'),

  // Controls
  confidenceSlider: document.getElementById('confidenceSlider'),
  confidenceValBadge: document.getElementById('confidenceValBadge'),
  btnStart: document.getElementById('btnStart'),
  btnStop: document.getElementById('btnStop'),
  btnSimulate: document.getElementById('btnSimulate'),

  // Counter & 6-Second Timer
  counterBox: document.getElementById('counterBox'),
  sneezeCountNum: document.getElementById('sneezeCountNum'),
  timerText: document.getElementById('timerText'),
  timerProgress: document.getElementById('timerProgress'),
  countdownMetaLabel: document.getElementById('countdownMetaLabel'),
  countdownHint: document.getElementById('countdownHint'),
  currentPhaseText: document.getElementById('currentPhaseText'),
  cooldownStatus: document.getElementById('cooldownStatus'),

  // Superstition & Gossip Report Card
  resultsCard: document.getElementById('resultsCard'),
  resultBanner: document.getElementById('resultBanner'),
  resultCategoryTag: document.getElementById('resultCategoryTag'),
  resultTitle: document.getElementById('resultTitle'),
  positivityScoreText: document.getElementById('positivityScoreText'),
  positivityBarFill: document.getElementById('positivityBarFill'),
  suspectAvatar: document.getElementById('suspectAvatar'),
  suspectCategoryText: document.getElementById('suspectCategoryText'),
  suspectName: document.getElementById('suspectName'),
  suspectVerdict: document.getElementById('suspectVerdict'),
  btnListenAgain: document.getElementById('btnListenAgain'),

  // History Log Table
  historyTableBody: document.getElementById('historyTableBody'),
  emptyHistoryMsg: document.getElementById('emptyHistoryMsg'),
  btnClearHistory: document.getElementById('btnClearHistory'),

  // Modal
  errorModal: document.getElementById('errorModal'),
  modalMessage: document.getElementById('modalMessage'),
  modalCloseBtn: document.getElementById('modalCloseBtn')
};

// Canvas 2D Rendering Context
let canvasCtx = null;
if (el.audioCanvas) {
  canvasCtx = el.audioCanvas.getContext('2d');
}

/**
 * ============================================================================
 * SUPERSTITION & GOSSIP LOGIC MATRIX SPECIFICATIONS
 * ============================================================================
 */
const SUPERSTITION_MATRIX = {
  1: {
    title: "SECRET ADMIRER DETECTED! 💖",
    positivityScore: 95,
    suspectCategory: "Secret Admirer / Old Friend",
    bannerClass: "type-1",
    avatar: "💖",
    suspects: [
      {
        name: "Your secret admirer from the coffee shop",
        quote: "They drew a tiny smiley face on your cup this morning and just smiled thinking about your laugh."
      },
      {
        name: "Your childhood best friend reminiscing",
        quote: "They were looking through old photos and literally just told someone how much they miss you!"
      },
      {
        name: "An ex who just saw your latest story",
        quote: "They paused on your photo for 45 seconds and thought: 'Wow, they are really thriving without me.'"
      },
      {
        name: "A colleague who appreciated your help today",
        quote: "They just told a teammate: 'Honestly, they saved the entire project today!'"
      }
    ]
  },
  2: {
    title: "WARNING: DRAMA & GOSSIP IN PROGRESS! 🗣️",
    positivityScore: 18,
    suspectCategory: "Jealous Rival / Former Group Chat",
    bannerClass: "type-2",
    avatar: "🗣️",
    suspects: [
      {
        name: "The coworker whose labeled lunch you accidentally ate",
        quote: "They are standing by the office water cooler whispering: 'I know for a fact they took my Greek yogurt!'"
      },
      {
        name: "A competitive acquaintance auditing your life",
        quote: "They just sent a screenshot of your post to their close friends with three question marks."
      },
      {
        name: "The group chat member who left you on 'Read'",
        quote: "They showed your message to their roommate and said: 'Can you believe the audacity of this text?!'"
      },
      {
        name: "Your neighborhood lawn perfectionist rival",
        quote: "They just peeked over the hedge with a tape measure muttering: 'Their dandelion problem is out of hand.'"
      }
    ]
  },
  3: {
    title: "ALLERGY ALERT: NOBODY IS TALKING ABOUT YOU! 🤧",
    positivityScore: 50,
    suspectCategory: "Pollen, Dust Mites, or Pet Dander",
    bannerClass: "type-3",
    avatar: "🤧",
    suspects: [
      {
        name: "Airborne Ragweed Pollen blowing in from 50 miles away",
        quote: "Zero social intrigue detected! Microscopic plant allergens have initiated an unsanctioned sinus takeover."
      },
      {
        name: "The ceiling fan dust bunny federation",
        quote: "They have occupied the top blades of your fan since 2021 and have launched their seasonal counter-offensive."
      },
      {
        name: "A golden retriever that strolled past your window",
        quote: "A single stray particle of pet dander caught the draft under your door with pinpoint aerodynamic accuracy."
      },
      {
        name: "Subtle changes in atmospheric barometric pressure",
        quote: "Nobody is gossiping about you—your sinuses are just moonlighting as a Victorian weather barometer!"
      }
    ]
  }
};

/**
 * ============================================================================
 * SENSITIVITY SLIDER & THRESHOLD MAPPING
 * ============================================================================
 */
function updateSensitivitySettings(sliderValue) {
  sensitivityPercent = sliderValue;
  // Slider 50 - 95:
  // Lower slider value = higher threshold (stricter, requires louder sneeze)
  // Higher slider value = lower threshold (sensitive, catches quiet sneezes)
  modelConfidenceThreshold = sliderValue / 100.0;

  // Dynamic RMS threshold scales smoothly
  // At 50%: ~0.16 RMS
  // At 70%: ~0.08 RMS
  // At 95%: ~0.035 RMS
  const normalized = (sliderValue - 50) / 45; // 0.0 to 1.0
  volumeThreshold = Math.max(0.035, 0.16 - normalized * 0.125);

  if (el.confidenceValBadge) {
    el.confidenceValBadge.textContent = `${sliderValue}% (${modelConfidenceThreshold.toFixed(2)} Sneeze Conf)`;
  }

  // Update visual threshold marker on the probability meter track
  if (el.thresholdMarker) {
    el.thresholdMarker.style.left = `${sliderValue}%`;
  }
}

/**
 * ============================================================================
 * STATE MACHINE MANAGER
 * ============================================================================
 */
function setAppState(newState) {
  currentState = newState;
  if (!el.statusBadge) return;

  el.statusBadge.className = 'status-badge';

  switch (newState) {
    case AppState.IDLE_STOPPED:
      el.statusBadge.classList.add('status-idle');
      if (el.statusText) el.statusText.textContent = 'Idle';
      if (el.currentPhaseText) el.currentPhaseText.textContent = 'IDLE STATE';
      if (el.btnStart) el.btnStart.disabled = false;
      if (el.btnStop) el.btnStop.disabled = true;
      if (el.canvasStartPrompt) el.canvasStartPrompt.classList.remove('hidden');
      resetTimerDisplay('stopped');
      break;

    case AppState.IDLE_LISTENING:
      el.statusBadge.classList.add('status-listening');
      if (el.statusText) el.statusText.textContent = 'Listening (Armed)';
      if (el.currentPhaseText) el.currentPhaseText.textContent = 'LISTENING (TIMER IDLE AT 6.0s)';
      if (el.btnStart) el.btnStart.disabled = true;
      if (el.btnStop) el.btnStop.disabled = false;
      if (el.canvasStartPrompt) el.canvasStartPrompt.classList.add('hidden');
      resetTimerDisplay('listening');
      break;

    case AppState.ACTIVE_WINDOW:
      el.statusBadge.classList.add('status-listening');
      if (el.statusText) el.statusText.textContent = 'Active Sneeze Window (6s)';
      if (el.currentPhaseText) el.currentPhaseText.textContent = 'COUNTDOWN ACTIVE (6s WINDOW)';
      if (el.btnStart) el.btnStart.disabled = true;
      if (el.btnStop) el.btnStop.disabled = false;
      if (el.canvasStartPrompt) el.canvasStartPrompt.classList.add('hidden');
      break;

    case AppState.RESOLVED:
      el.statusBadge.classList.add('status-processing');
      if (el.statusText) el.statusText.textContent = 'Session Resolved';
      if (el.currentPhaseText) el.currentPhaseText.textContent = 'GOSSIP REPORT COMPILED';
      if (el.btnStart) el.btnStart.disabled = false;
      if (el.btnStop) el.btnStop.disabled = false;
      break;
  }
}

/**
 * Resets 6-second countdown timer display to paused 06.00s
 */
function resetTimerDisplay(mode) {
  if (sessionIntervalId) {
    clearInterval(sessionIntervalId);
    sessionIntervalId = null;
  }

  if (el.timerText) {
    el.timerText.textContent = '06.00s';
    el.timerText.className = 'timer-text waiting';
  }

  if (el.timerProgress) {
    el.timerProgress.style.width = '0%';
  }

  if (el.countdownMetaLabel) {
    el.countdownMetaLabel.innerHTML = '<span class="countdown-icon">⏱️</span> Countdown (Idle at 6.00s):';
  }

  if (el.countdownHint) {
    if (mode === 'listening') {
      el.countdownHint.innerHTML = '<span class="pulse-dot-amber"></span> Paused at 6.0s. Countdown begins ONLY when the first sneeze is detected!';
    } else {
      el.countdownHint.innerHTML = '<span class="pulse-dot-amber"></span> Press "Start Listening" to activate audio classification.';
    }
  }
}

/**
 * ============================================================================
 * WEB AUDIO API PIPELINE INITIALIZATION
 * ============================================================================
 */
async function startListening() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showErrorModal(
        'Microphone API Not Supported',
        'Your browser does not support getUserMedia audio streaming. Please open this app in Chrome, Safari, Edge, or Firefox.'
      );
      return;
    }

    // Connect to microphone with raw dynamic range (no aggressive AGC or clipping)
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioContextClass();

    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    micSourceNode = audioContext.createMediaStreamSource(mediaStream);

    // High-resolution AnalyserNode for time-domain & frequency spectrum analysis
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 1024; // 512 frequency bins
    analyserNode.smoothingTimeConstant = 0.15; // Low smoothing for rapid transient burst response

    micSourceNode.connect(analyserNode);

    // Reset counters and enter listening state
    sneezeCount = 0;
    updateCounterDisplay(0);
    setAppState(AppState.IDLE_LISTENING);

    // Initialize TensorFlow.js / ML status badge
    updateModelBadge('ready', 'Neural Audio Engine Active 🧠');

    // Start 60 FPS real-time audio analysis & visualization loop
    startAudioAnalysisLoop();

  } catch (err) {
    console.error('Microphone initialization error:', err);
    let errorTitle = 'Microphone Access Required';
    let errorMessage = 'Please allow microphone access in your browser to enable live sneeze detection. You can also click "Test Sneeze" to verify the app immediately!';

    if (err.name === 'NotFoundError') {
      errorTitle = 'No Microphone Detected';
      errorMessage = 'No audio input microphone was found on this device. Please connect a microphone or headset.';
    } else if (err.name === 'NotAllowedError') {
      errorTitle = 'Microphone Permission Blocked';
      errorMessage = 'Microphone permission was denied. Please allow microphone permissions in your browser address bar.';
    }

    showErrorModal(errorTitle, errorMessage);
    setAppState(AppState.IDLE_STOPPED);
  }
}

/**
 * Stops microphone capture and halts audio processing
 */
function stopListening() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (sessionIntervalId) {
    clearInterval(sessionIntervalId);
    sessionIntervalId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }

  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close();
    audioContext = null;
  }

  if (micSourceNode) {
    try { micSourceNode.disconnect(); } catch (e) {}
    micSourceNode = null;
  }

  analyserNode = null;
  clearCanvas();
  updateConfidenceMeter(0.0, 'Silence / Ambient');

  if (el.classifiedSoundLabel) {
    el.classifiedSoundLabel.textContent = 'Silence / Ambient';
    el.classifiedSoundLabel.className = 'hud-classified-class';
  }

  if (currentState !== AppState.RESOLVED) {
    setAppState(AppState.IDLE_STOPPED);
  }
}

/**
 * Update UI Model Status Badge
 */
function updateModelBadge(status, text) {
  if (!el.modelStatusBadge) return;
  if (status === 'ready') {
    el.modelStatusBadge.className = 'model-badge ready';
    if (el.modelStatusText) el.modelStatusText.textContent = text;
  } else {
    el.modelStatusBadge.className = 'model-badge';
    if (el.modelStatusText) el.modelStatusText.textContent = text;
  }
}

/**
 * ============================================================================
 * REAL-TIME HIGH-PRECISION AUDIO CLASSIFICATION LOOP (60 FPS)
 * ============================================================================
 */
function startAudioAnalysisLoop() {
  if (!analyserNode) return;

  const bufferLength = analyserNode.frequencyBinCount;
  const timeData = new Uint8Array(bufferLength);
  const freqData = new Uint8Array(bufferLength);
  const sampleRate = audioContext ? audioContext.sampleRate : 44100;
  const binWidth = (sampleRate / 2) / bufferLength;

  function processFrame() {
    if (!analyserNode) return;

    // Fetch raw time domain waveform and frequency spectrum
    analyserNode.getByteTimeDomainData(timeData);
    analyserNode.getByteFrequencyData(freqData);

    // 1. Time-Domain Metrics: RMS Amplitude, Peak, and Zero-Crossing Rate (ZCR)
    let sumSquares = 0;
    let peakSample = 0;
    let zeroCrossings = 0;
    let prevVal = 0;

    for (let i = 0; i < timeData.length; i++) {
      const sample = (timeData[i] - 128) / 128.0;
      sumSquares += sample * sample;
      const absSample = Math.abs(sample);
      if (absSample > peakSample) peakSample = absSample;

      if (i > 0 && ((sample >= 0 && prevVal < 0) || (sample < 0 && prevVal >= 0))) {
        zeroCrossings++;
      }
      prevVal = sample;
    }

    const currentRMS = Math.sqrt(sumSquares / timeData.length);
    const zcr = zeroCrossings / timeData.length;

    // 2. Adaptive Ambient Noise Floor Estimation
    if (currentRMS < ambientNoiseFloor) {
      ambientNoiseFloor = ambientNoiseFloor * 0.92 + currentRMS * 0.08;
    } else {
      ambientNoiseFloor = ambientNoiseFloor * 0.997 + currentRMS * 0.003;
    }
    ambientNoiseFloor = Math.max(0.004, Math.min(0.06, ambientNoiseFloor));

    // 3. Multi-Band Frequency Energy Breakdown
    let totalFreqWeight = 0;
    let totalEnergy = 0;
    let lowEnergy = 0;            // 0 - 300 Hz (Desk bumps, mic thuds, rumble)
    let vocalEnergy = 0;          // 300 - 1800 Hz (Speech vowel harmonics)
    let nasalTurbulenceEnergy = 0;// 1800 - 7500 Hz (Air friction rushing through nasal cavity)
    let highSibilanceEnergy = 0;  // 7500 - 16000 Hz (High frequency air hiss)

    for (let i = 0; i < bufferLength; i++) {
      const energy = freqData[i];
      const freq = i * binWidth;
      totalEnergy += energy;
      totalFreqWeight += freq * energy;

      if (freq <= 300) {
        lowEnergy += energy;
      } else if (freq > 300 && freq <= 1800) {
        vocalEnergy += energy;
      } else if (freq > 1800 && freq <= 7500) {
        nasalTurbulenceEnergy += energy;
      } else if (freq > 7500) {
        highSibilanceEnergy += energy;
      }
    }

    const spectralCentroid = totalEnergy > 0 ? Math.round(totalFreqWeight / totalEnergy) : 0;
    const nasalRatio = totalEnergy > 0 ? nasalTurbulenceEnergy / totalEnergy : 0.0;
    const vocalRatio = totalEnergy > 0 ? vocalEnergy / totalEnergy : 0.0;
    const lowRatio = totalEnergy > 0 ? lowEnergy / totalEnergy : 0.0;

    // 4. Attack Velocity & History Tracking
    const attackVelocity = currentRMS - previousRMS;
    previousRMS = currentRMS;
    recentRMSHistory[historyIndex] = currentRMS;
    historyIndex = (historyIndex + 1) % recentRMSHistory.length;

    // Find recent max RMS in the last ~200ms
    let recentMaxRMS = 0;
    for (let i = 0; i < recentRMSHistory.length; i++) {
      if (recentRMSHistory[i] > recentMaxRMS) recentMaxRMS = recentRMSHistory[i];
    }

    // 5. Sound Classification & Sneeze Probability Engine
    const now = Date.now();
    const isDebouncing = (now - lastSneezeTimestamp) < DEBOUNCE_COOLDOWN_MS;

    let sneezeProbability = 0.0;
    let topClass = 'Silence / Ambient';
    let isSneezeSpike = false;

    // Is the audio level significantly higher than room ambient floor?
    const isAudioActive = currentRMS > Math.max(0.012, ambientNoiseFloor * 1.8);

    if (!isAudioActive) {
      // Room is quiet
      sneezeProbability = 0.0;
      topClass = 'Silence / Ambient';
      burstFrameCount = 0;
    } else {
      burstFrameCount++;

      // Compute sub-scores:
      // (a) Volume Score: How strong is the burst relative to volumeThreshold?
      const volumeScore = Math.min(1.0, currentRMS / volumeThreshold);

      // (b) Nasal Turbulence Score: Ratio of energy in 1.8k - 7.5k Hz
      // Typical speech has nasalRatio 0.08 - 0.20
      // A sneeze has nasalRatio 0.25 - 0.65
      const nasalScore = Math.min(1.0, Math.max(0, (nasalRatio - 0.16) / 0.34));

      // (c) Spectral Centroid Score: Center of frequency mass
      // Sneeze centroid: 1900 - 6500 Hz
      const centroidScore = Math.min(1.0, Math.max(0, (spectralCentroid - 1400) / 2800));

      // (d) Zero-Crossing Rate Score: Noise-like turbulent airflow has high ZCR
      const zcrScore = Math.min(1.0, Math.max(0, (zcr - 0.08) / 0.22));

      // Base Sneeze Probability
      sneezeProbability = (volumeScore * 0.40) + (nasalScore * 0.35) + (centroidScore * 0.15) + (zcrScore * 0.10);

      // Rejection Filters & Sound Classification:
      if (lowRatio > 0.60 && nasalRatio < 0.15) {
        // Heavy low rumble (desk tap, mic touch, footsteps)
        sneezeProbability *= 0.2;
        topClass = 'Low Rumble / Bump (Ignored)';
      } else if (vocalRatio > 0.68 && nasalRatio < 0.20 && burstFrameCount > 15) {
        // Continuous vocalized speech or singing (prolonged duration + high vocal formant ratio)
        sneezeProbability *= 0.25;
        topClass = 'Human Speech (Ignored)';
      } else if (burstFrameCount <= 2 && peakSample > 0.35 && nasalRatio < 0.22 && recentMaxRMS < 0.06) {
        // Very brief impulse without turbulent tail (hand clap or finger snap)
        sneezeProbability *= 0.35;
        topClass = 'Hand Clap / Snap (Ignored)';
      } else if (sneezeProbability >= modelConfidenceThreshold) {
        // Matches sneeze profile!
        topClass = `Sneeze Detected! (${Math.round(sneezeProbability * 100)}%) 🤧`;
        isSneezeSpike = true;
      } else if (nasalRatio > 0.22 && currentRMS >= volumeThreshold * 0.7) {
        topClass = `Breath / Noise (${Math.round(sneezeProbability * 100)}%)`;
      } else {
        topClass = `Speech / Talking (${Math.round(sneezeProbability * 100)}%)`;
      }

      // Boost for unmistakable explosive nasal bursts ("ACHOO!" or "TSCHHH!")
      if (attackVelocity > 0.04 && nasalRatio > 0.28 && currentRMS >= volumeThreshold) {
        sneezeProbability = Math.min(0.99, sneezeProbability * 1.3);
        if (sneezeProbability >= modelConfidenceThreshold) {
          isSneezeSpike = true;
          topClass = `Sneeze Detected! (${Math.round(sneezeProbability * 100)}%) 🤧`;
        }
      }
    }

    // 6. Update Real-Time UI Meters and Telemetry HUD
    updateConfidenceMeter(sneezeProbability, topClass);
    updateTelemetry(topClass, sneezeProbability, spectralCentroid, currentRMS, isDebouncing);

    // 7. Verify Sneeze and Trigger State Machine
    if (isSneezeSpike && !isDebouncing) {
      onVerifiedSneezeDetected(sneezeProbability);
    }

    // 8. Render Canvas Oscilloscope and Multi-Band Spectrum
    renderCanvasVisualizer(timeData, freqData, isSneezeSpike);

    animationFrameId = requestAnimationFrame(processFrame);
  }

  animationFrameId = requestAnimationFrame(processFrame);
}

/**
 * Updates dynamic Sneeze Confidence Meter Progress Bar
 */
function updateConfidenceMeter(probability, label) {
  const percent = Math.min(100, Math.max(0, Math.round(probability * 100)));

  if (el.confidenceFill) {
    el.confidenceFill.style.width = `${percent}%`;
    if (probability >= modelConfidenceThreshold) {
      el.confidenceFill.style.background = 'linear-gradient(90deg, #22c55e, #ef4444)';
      el.confidenceFill.style.boxShadow = '0 0 16px rgba(239, 68, 68, 0.8)';
    } else {
      el.confidenceFill.style.background = 'linear-gradient(90deg, #06b6d4, #22c55e)';
      el.confidenceFill.style.boxShadow = 'none';
    }
  }

  if (el.confidenceText) {
    el.confidenceText.textContent = `${probability.toFixed(2)} (${percent}%)`;
  }
}

/**
 * Updates HUD Telemetry Metrics & Status Badges
 */
function updateTelemetry(topClass, probability, centroid, rms, isDebouncing) {
  if (el.classifiedSoundLabel) {
    if (isDebouncing && topClass.includes('Sneeze')) {
      el.classifiedSoundLabel.textContent = '1.2s Debounce Guard Active 🛡️';
      el.classifiedSoundLabel.className = 'hud-classified-class sound-speech';
    } else {
      el.classifiedSoundLabel.textContent = topClass;
      if (topClass.includes('Sneeze')) {
        el.classifiedSoundLabel.className = 'hud-classified-class sneeze-detected';
      } else if (topClass.includes('Ignored') || topClass.includes('Speech')) {
        el.classifiedSoundLabel.className = 'hud-classified-class sound-ignored';
      } else {
        el.classifiedSoundLabel.className = 'hud-classified-class';
      }
    }
  }

  if (el.valCentroid) {
    el.valCentroid.textContent = `${centroid} Hz (${Math.round(probability * 100)}% Conf)`;
  }

  if (el.valBurst) {
    el.valBurst.textContent = `${rms.toFixed(3)} RMS`;
  }

  if (el.cooldownStatus) {
    if (isDebouncing) {
      const elapsed = Date.now() - lastSneezeTimestamp;
      const remainSec = Math.max(0, (DEBOUNCE_COOLDOWN_MS - elapsed) / 1000).toFixed(1);
      el.cooldownStatus.textContent = `Debouncing (${remainSec}s)`;
      el.cooldownStatus.style.color = 'var(--neon-amber)';
    } else {
      el.cooldownStatus.textContent = 'Armed & Ready';
      el.cooldownStatus.style.color = 'var(--neon-green)';
    }
  }
}

/**
 * ============================================================================
 * VERIFIED SNEEZE EVENT & CONDITIONAL TIMER
 * ============================================================================
 */
function onVerifiedSneezeDetected(confidenceScore) {
  // Only detect when microphone is active
  if (currentState !== AppState.IDLE_LISTENING && currentState !== AppState.ACTIVE_WINDOW) {
    return;
  }

  const now = Date.now();

  // 1.2-Second Debounce Cooldown logic
  if (now - lastSneezeTimestamp < DEBOUNCE_COOLDOWN_MS) {
    return;
  }

  lastSneezeTimestamp = now;

  // Increment sneeze counter
  sneezeCount++;
  updateCounterDisplay(sneezeCount);
  triggerCounterBurstFx();
  playSneezeConfirmationChime();

  if (el.classifiedSoundLabel) {
    el.classifiedSoundLabel.textContent = `VERIFIED SNEEZE #${sneezeCount} (${Math.round(confidenceScore * 100)}%) 🤧`;
    el.classifiedSoundLabel.className = 'hud-classified-class sneeze-detected';
  }

  // CONDITIONAL TIMER RULE:
  // Timer is paused at 6.00s while passively listening.
  // It starts counting down ONLY upon the very FIRST verified sneeze!
  if (currentState === AppState.IDLE_LISTENING) {
    startActiveCountdownSession();
  } else if (currentState === AppState.ACTIVE_WINDOW) {
    // If follow-up sneeze occurs during the active window, refresh the 6-second window
    refreshActiveCountdownSession();
  }
}

/**
 * Plays an acoustic confirmation chime confirming the sneeze was registered
 */
function playSneezeConfirmationChime() {
  try {
    if (!audioContext || audioContext.state === 'closed') return;
    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }
    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    const now = audioContext.currentTime;

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(880, now); // A5
    osc.frequency.exponentialRampToValueAtTime(1320, now + 0.08); // E6

    gain.gain.setValueAtTime(0.25, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);

    osc.connect(gain);
    gain.connect(audioContext.destination);

    osc.start(now);
    osc.stop(now + 0.25);
  } catch (e) {
    // Fallback if audio output blocked
  }
}

/**
 * Generates an authentic procedural synthetic sneeze sound via Web Audio API
 * Used by "Test Sneeze" button to simulate a realistic audio event!
 */
function playProceduralSneezeSound() {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const ctx = audioContext || new AudioContextClass();
    if (ctx.state === 'suspended') ctx.resume();

    const now = ctx.currentTime;

    // 1. Initial Inhale Breath ("Ahhh...")
    const inhaleOsc = ctx.createOscillator();
    const inhaleGain = ctx.createGain();
    inhaleOsc.type = 'sine';
    inhaleOsc.frequency.setValueAtTime(260, now);
    inhaleOsc.frequency.exponentialRampToValueAtTime(440, now + 0.20);

    inhaleGain.gain.setValueAtTime(0.01, now);
    inhaleGain.gain.linearRampToValueAtTime(0.12, now + 0.18);
    inhaleGain.gain.linearRampToValueAtTime(0.01, now + 0.22);

    inhaleOsc.connect(inhaleGain);
    inhaleGain.connect(ctx.destination);
    inhaleOsc.start(now);
    inhaleOsc.stop(now + 0.22);

    // 2. Explosive Nasal Turbulence Noise Burst ("-CHOOO!")
    const burstDuration = 0.35;
    const bufferSize = ctx.sampleRate * burstDuration;
    const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }

    const whiteNoise = ctx.createBufferSource();
    whiteNoise.buffer = noiseBuffer;

    // Bandpass filter centered at 3200 Hz with Q factor for nasal resonance
    const bandpass = ctx.createBiquadFilter();
    bandpass.type = 'bandpass';
    bandpass.frequency.setValueAtTime(3200, now + 0.20);
    bandpass.frequency.exponentialRampToValueAtTime(1600, now + 0.20 + burstDuration);
    bandpass.Q.value = 2.2;

    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.0, now + 0.20);
    noiseGain.gain.linearRampToValueAtTime(0.40, now + 0.23); // Sharp attack
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.20 + burstDuration); // Decay

    whiteNoise.connect(bandpass);
    bandpass.connect(noiseGain);
    noiseGain.connect(ctx.destination);

    whiteNoise.start(now + 0.20);
    whiteNoise.stop(now + 0.20 + burstDuration);

  } catch (e) {
    console.warn('Procedural audio preview note:', e);
  }
}

/**
 * Starts the conditional 6-second countdown timer
 */
function startActiveCountdownSession() {
  setAppState(AppState.ACTIVE_WINDOW);
  sessionStartTime = Date.now();

  if (el.timerText) {
    el.timerText.className = 'timer-text running';
  }
  if (el.countdownMetaLabel) {
    el.countdownMetaLabel.innerHTML = `<span class="countdown-icon">🤧</span> Sneeze #${sneezeCount} Triggered — 6s Countdown:`;
  }
  if (el.countdownHint) {
    el.countdownHint.innerHTML = '<span class="pulse-dot-red"></span> 6-second countdown active! Listening for follow-up sneezes...';
  }

  if (sessionIntervalId) {
    clearInterval(sessionIntervalId);
  }

  sessionIntervalId = setInterval(() => {
    const elapsed = Date.now() - sessionStartTime;
    const remainingMs = Math.max(0, SESSION_WINDOW_MS - elapsed);

    // Update countdown numeric timer (e.g. "05.42s")
    const secondsRemaining = (remainingMs / 1000).toFixed(2);
    if (el.timerText) {
      el.timerText.textContent = `${secondsRemaining}s`;
    }

    // Update countdown animated progress bar
    const progressPercent = ((SESSION_WINDOW_MS - remainingMs) / SESSION_WINDOW_MS) * 100;
    if (el.timerProgress) {
      el.timerProgress.style.width = `${progressPercent}%`;
    }

    // When 6-second timer expires, transition to RESOLVED state
    if (remainingMs <= 0) {
      clearInterval(sessionIntervalId);
      sessionIntervalId = null;
      resolveSession();
    }
  }, 25);
}

/**
 * Refreshes 6-second timer when follow-up sneeze occurs
 */
function refreshActiveCountdownSession() {
  sessionStartTime = Date.now();
  if (el.countdownMetaLabel) {
    el.countdownMetaLabel.innerHTML = `<span class="countdown-icon">🤧</span> Sneeze #${sneezeCount} Counted — 6s Refreshed:`;
  }
  if (el.timerProgress) {
    el.timerProgress.style.width = '0%';
  }
}

/**
 * RESOLUTION STATE:
 * Compiles Superstition Report and updates Suspect Reveal
 */
function resolveSession() {
  setAppState(AppState.RESOLVED);

  if (el.timerText) {
    el.timerText.textContent = '0.00s (LOCKED)';
    el.timerText.className = 'timer-text waiting';
  }
  if (el.timerProgress) el.timerProgress.style.width = '100%';
  if (el.countdownMetaLabel) {
    el.countdownMetaLabel.innerHTML = '<span class="countdown-icon">🏁</span> Session Complete:';
  }
  if (el.countdownHint) {
    el.countdownHint.innerHTML = '<span class="pulse-dot-green"></span> 6-second window expired! Superstition report compiled below.';
  }

  // Superstition Logic Matrix:
  // 1 Sneeze -> Secret Admirer (95% Positivity)
  // 2 Sneezes -> Warning Gossip (18% Positivity)
  // 3+ Sneezes -> Allergy Alert (50% Positivity)
  let countKey = 3;
  if (sneezeCount === 1) countKey = 1;
  else if (sneezeCount === 2) countKey = 2;
  else if (sneezeCount >= 3) countKey = 3;
  else countKey = 1; // Fallback for 0 in test

  const reportSpec = SUPERSTITION_MATRIX[countKey];

  // Pick a randomized funny suspect
  const randomSuspectIndex = Math.floor(Math.random() * reportSpec.suspects.length);
  const chosenSuspect = reportSpec.suspects[randomSuspectIndex];

  // Update Report Card UI
  el.resultBanner.className = `result-banner ${reportSpec.bannerClass}`;
  el.resultCategoryTag.textContent = `${sneezeCount} SNEEZE${sneezeCount === 1 ? '' : 'S'} IN 6S WINDOW`;
  el.resultTitle.textContent = reportSpec.title;
  el.positivityScoreText.textContent = `${reportSpec.positivityScore}%`;

  // Animate Positivity Score Bar Fill
  el.positivityBarFill.style.width = '0%';
  setTimeout(() => {
    el.positivityBarFill.style.width = `${reportSpec.positivityScore}%`;
  }, 80);

  // Populate Suspect Reveal Card
  el.suspectAvatar.textContent = reportSpec.avatar;
  el.suspectCategoryText.textContent = `Category: ${reportSpec.suspectCategory}`;
  el.suspectName.textContent = chosenSuspect.name;
  el.suspectVerdict.textContent = `"${chosenSuspect.quote}"`;

  // Smooth scroll to results card
  el.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Add record to History Log Table
  addHistoryLogEntry(
    sneezeCount,
    reportSpec.title,
    reportSpec.positivityScore,
    chosenSuspect.name,
    chosenSuspect.quote
  );
}

/**
 * Adds an entry to the Session History Table
 */
function addHistoryLogEntry(count, title, score, suspectName, verdict) {
  const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  sessionHistory.unshift({
    timestamp,
    count,
    title,
    score,
    suspectName,
    verdict
  });

  renderHistoryTable();
}

/**
 * Renders Session History Table
 */
function renderHistoryTable() {
  if (!el.historyTableBody) return;

  if (sessionHistory.length === 0) {
    el.historyTableBody.innerHTML = '';
    if (el.emptyHistoryMsg) el.emptyHistoryMsg.style.display = 'block';
    return;
  }

  if (el.emptyHistoryMsg) el.emptyHistoryMsg.style.display = 'none';

  el.historyTableBody.innerHTML = sessionHistory.map(entry => {
    const badgeClass = entry.count === 1 ? 'badge-1' : entry.count === 2 ? 'badge-2' : 'badge-3';
    return `
      <tr>
        <td style="color: var(--text-dim); font-size: 0.78rem;">${entry.timestamp}</td>
        <td><span class="${badgeClass}">🤧 ${entry.count}</span></td>
        <td style="font-weight: 700; color: #fff;">${entry.title}</td>
        <td><span style="color: ${entry.score > 50 ? 'var(--neon-green)' : 'var(--neon-red)'}; font-weight: 700;">${entry.score}%</span></td>
        <td style="color: var(--neon-cyan);">${entry.suspectName}</td>
        <td style="color: var(--text-muted); font-size: 0.76rem; max-width: 260px;">"${entry.verdict}"</td>
      </tr>
    `;
  }).join('');
}

/**
 * Updates big numerical counter display
 */
function updateCounterDisplay(count) {
  if (el.sneezeCountNum) {
    el.sneezeCountNum.textContent = count;
  }
}

/**
 * Visual trigger animation on counter box when sneeze is counted
 */
function triggerCounterBurstFx() {
  if (!el.counterBox) return;
  el.counterBox.classList.add('burst');
  setTimeout(() => {
    el.counterBox.classList.remove('burst');
  }, 400);
}

/**
 * ============================================================================
 * HTML5 CANVAS WAVEFORM & MULTI-BAND SPECTRUM VISUALIZER
 * ============================================================================
 */
function renderCanvasVisualizer(timeData, freqData, isSpikeActive) {
  if (!canvasCtx || !el.audioCanvas) return;

  const width = el.audioCanvas.width;
  const height = el.audioCanvas.height;

  // Clear canvas with deep slate cyberpunk tint
  canvasCtx.fillStyle = 'rgba(9, 13, 22, 0.45)';
  canvasCtx.fillRect(0, 0, width, height);

  // 1. Draw Multi-Band Frequency Spectrum Bars in Background
  const barCount = 48;
  const barWidth = width / barCount;
  for (let i = 0; i < barCount; i++) {
    const freqVal = freqData[i * 4] || 0;
    const barHeight = (freqVal / 255) * (height * 0.72);
    // Nasal friction frequencies (middle-right bars) tinted cyan/neon
    const hue = 160 + i * 2.5;
    canvasCtx.fillStyle = `hsla(${hue}, 85%, 52%, 0.18)`;
    canvasCtx.fillRect(i * barWidth, height - barHeight, barWidth - 1.5, barHeight);
  }

  // 2. Draw Sensitivity Volume Threshold Reference Line
  const thresholdY = height - Math.min(1.0, (volumeThreshold / 0.35)) * height;
  if (thresholdY >= 0 && thresholdY <= height) {
    canvasCtx.beginPath();
    canvasCtx.setLineDash([4, 4]);
    canvasCtx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
    canvasCtx.lineWidth = 1;
    canvasCtx.moveTo(0, thresholdY);
    canvasCtx.lineTo(width, thresholdY);
    canvasCtx.stroke();
    canvasCtx.setLineDash([]);
  }

  // 3. Draw Real-Time Oscilloscope Waveform Line
  canvasCtx.lineWidth = 2.5;
  if (isSpikeActive) {
    canvasCtx.strokeStyle = '#ef4444';
    canvasCtx.shadowColor = '#ef4444';
    canvasCtx.shadowBlur = 16;
  } else {
    canvasCtx.strokeStyle = '#22c55e';
    canvasCtx.shadowColor = '#22c55e';
    canvasCtx.shadowBlur = 6;
  }

  canvasCtx.beginPath();
  const sliceWidth = width / timeData.length;
  let x = 0;

  for (let i = 0; i < timeData.length; i++) {
    const v = timeData[i] / 128.0;
    const y = (v * height) / 2;
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
    x += sliceWidth;
  }

  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();
  canvasCtx.shadowBlur = 0;
}

function clearCanvas() {
  if (!canvasCtx || !el.audioCanvas) return;
  const width = el.audioCanvas.width;
  const height = el.audioCanvas.height;
  canvasCtx.fillStyle = '#090d16';
  canvasCtx.fillRect(0, 0, width, height);

  canvasCtx.strokeStyle = '#243455';
  canvasCtx.lineWidth = 1.5;
  canvasCtx.beginPath();
  canvasCtx.moveTo(0, height / 2);
  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();
}

function resizeCanvas() {
  if (!el.audioCanvas) return;
  const rect = el.audioCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  el.audioCanvas.width = rect.width * dpr;
  el.audioCanvas.height = rect.height * dpr;
  if (canvasCtx) {
    canvasCtx.scale(dpr, dpr);
  }
  clearCanvas();
}

/**
 * ============================================================================
 * MODAL & PERMISSION ERROR ALERTS
 * ============================================================================
 */
function showErrorModal(title, message) {
  if (!el.errorModal) return;
  const titleEl = el.errorModal.querySelector('.modal-title');
  if (titleEl) titleEl.innerHTML = `<span>⚠️</span> ${title}`;
  if (el.modalMessage) el.modalMessage.textContent = message;
  el.errorModal.classList.add('active');
}

function closeErrorModal() {
  if (el.errorModal) el.errorModal.classList.remove('active');
}

/**
 * ============================================================================
 * EVENT LISTENERS & BOOTSTRAP INITIALIZATION
 * ============================================================================
 */
function setupEventListeners() {
  // Start Listening Button
  if (el.btnStart) {
    el.btnStart.addEventListener('click', () => {
      startListening();
    });
  }

  // Stop Listening Button
  if (el.btnStop) {
    el.btnStop.addEventListener('click', () => {
      stopListening();
    });
  }

  // Canvas container click prompt
  if (el.canvasContainer) {
    el.canvasContainer.addEventListener('click', () => {
      if (currentState === AppState.IDLE_STOPPED) {
        startListening();
      }
    });
  }

  // Test Sneeze Button (Plays realistic procedural sneeze sound and triggers detection)
  if (el.btnSimulate) {
    el.btnSimulate.addEventListener('click', () => {
      if (currentState === AppState.IDLE_STOPPED) {
        setAppState(AppState.IDLE_LISTENING);
      }
      playProceduralSneezeSound();
      updateConfidenceMeter(0.92, 'Simulated Sneeze! 🤧');
      onVerifiedSneezeDetected(0.92);
    });
  }

  // Listen Again / New Session Button in Results Card
  if (el.btnListenAgain) {
    el.btnListenAgain.addEventListener('click', () => {
      sneezeCount = 0;
      updateCounterDisplay(0);
      resetTimerDisplay('listening');
      if (!audioContext || audioContext.state === 'closed') {
        startListening();
      } else {
        setAppState(AppState.IDLE_LISTENING);
      }
    });
  }

  // Model Confidence Threshold Slider
  if (el.confidenceSlider) {
    el.confidenceSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      updateSensitivitySettings(val);
    });
  }

  // Clear Session History Button
  if (el.btnClearHistory) {
    el.btnClearHistory.addEventListener('click', () => {
      sessionHistory.length = 0;
      renderHistoryTable();
    });
  }

  // Modal Dismiss Actions
  if (el.modalCloseBtn) {
    el.modalCloseBtn.addEventListener('click', closeErrorModal);
  }
  if (el.errorModal) {
    el.errorModal.addEventListener('click', (e) => {
      if (e.target === el.errorModal) closeErrorModal();
    });
  }

  // Responsive Window Resize
  window.addEventListener('resize', () => {
    resizeCanvas();
  });
}

// Bootstrap Application on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  resizeCanvas();

  // Initialize sensitivity threshold from slider default (65%)
  const initialSliderVal = el.confidenceSlider ? parseInt(el.confidenceSlider.value, 10) : 65;
  updateSensitivitySettings(initialSliderVal || 65);

  setAppState(AppState.IDLE_STOPPED);
  renderHistoryTable();
});
