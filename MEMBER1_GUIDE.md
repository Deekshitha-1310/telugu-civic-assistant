# 📘 Member 1 Student Action Manual: Audio & ASR Lead
**Name / Roll Number:** S20240010065  
**Role:** Audio Procurement, Noise Augmentation, Spectral Denoising & Whisper ASR  
**Your Evaluation Target:** 20 Marks (NLP Course Project)  

---

## 🧭 Your Mission in Simple Words
You are responsible for everything related to **voice, noise, and speech-to-text**.  
In the real world, rural citizens speak with noisy backgrounds (traffic, village offices, fan noise). Your job is to:
1. Collect real spoken Telugu audio of welfare queries.
2. Mix realistic noise at controlled levels (Clean, 15 dB, 10 dB, 5 dB).
3. Apply spectral denoising filters to clean the audio.
4. Pass it through Whisper ASR to convert it to Telugu text.
5. Measure how much noise hurt the accuracy and how much your filter restored it (Experiment 1).

---

## 📋 The Files You Own and Edit
| File Path | What You Do With This File |
| :--- | :--- |
| `data/audio/raw_recordings/` | Folder where you put your recorded `.wav` voice files. |
| `data/audio/noise_sources/` | Folder where you put background noise audio (`traffic_noise.wav`). |
| `src/asr_smoke_test.py` | Run this to check if Whisper runs on your laptop without crashing. |
| `src/noise_augmenter.py` | Run this to create noisy versions of your audio at 15 dB, 10 dB, 5 dB. |
| `src/audio_cleaner.py` | Run this to filter out noise using spectral gating. |
| `src/asr_engine.py` | The main speech recognition engine that Member 3 uses in the Streamlit app. |
| `src/evaluate_asr.py` | Run this to produce the final WER table and graph for your report. |

---

## 🚀 Step-by-Step Action Guide (What You Should Actually Do)

### Step 1: Set Up Your Laptop Environment
1. Open your terminal / PowerShell in the project folder:
   ```bash
   cd "C:\Users\Akshit Reddy\.gemini\antigravity-ide\scratch\telugu_civic_assistant"
   ```
2. Install the audio libraries you need:
   ```bash
   pip install torch torchaudio librosa soundfile noisereduce jiwer transformers matplotlib pydub
   ```
3. Test if PyTorch can see your laptop GPU (it's okay if you only have CPU, it will still work):
   ```bash
   python -c "import torch; print('GPU available!' if torch.cuda.is_available() else 'Using CPU mode (works fine)')"
   ```

---

### Step 2: Record Sample Telugu Audio (Your Data Collection)
You need spoken Telugu audio of the welfare queries so your team can test the system.
1. **The Sentences to Record First (5 starter samples):**
   - Sample 1: *"రైతు బంధు పైసలు ఎప్పుడొస్తయి"* (Rythu Bandhu money when will it come?)
   - Sample 2: *"ఆరోగ్యశ్రీ కార్డు ఎలా తీసుకోవాలి"* (How to get Aarogyasri card?)
   - Sample 3: *"అమ్మ ఒడి పథకానికి ఎవరు అర్హులు"* (Who is eligible for Amma Vodi?)
   - Sample 4: *"ఆసరా పెన్షన్ దరఖాస్తు విధానం ఏమిటి"* (What is application procedure for Aasara?)
   - Sample 5: *"రైతు భరోసా ఆర్థిక సహాయం ఎంత"* (How much financial help in Rythu Bharosa?)
2. **How to Record:**
   - Use the Voice Recorder app on your phone or laptop.
   - Speak naturally in a quiet room, holding the phone ~15 cm from your mouth.
3. **Save the Files:**
   - Place them in `data/audio/raw_recordings/` named:
     `sample_01.wav`, `sample_02.wav`, `sample_03.wav`, `sample_04.wav`, `sample_05.wav`.
   - *Note:* If your phone records `.m4a` or `.mp3`, use the converter helper below.

---

### Step 3: Get Real Background Noise
1. You need a background noise file to simulate real-world conditions.
2. **Easy Ways to Get It:**
   - Either record 30 seconds of street traffic or ceiling fan hum on your phone.
   - OR download a free traffic audio clip (from freesound.org or YouTube).
3. Save it as `data/audio/noise_sources/traffic_noise.wav`.

---

### Step 4: Run the Day 1 ASR Smoke Test
1. Run the test script:
   ```bash
   python src/asr_smoke_test.py
   ```
2. **What to Observe on Your Screen:**
   - The script will automatically download the Whisper model from Hugging Face (~150 MB).
   - It will transcribe your sample audio and print:
     ```
     Ref:  రైతు బంధు పైసలు ఎప్పుడొస్తయి
     Hyp:  రైతు బంధు పైసలు ఎప్పుడొస్తాయి
     --> Performance: WER = 15.2%, CER = 4.1%
     ```
3. **Your Decision:**
   - If `whisper-small` finishes in under 2 seconds per sentence on your machine, keep it. That is your winning production model!
   - Notice that we set `condition_on_previous_text=False` inside the code—this stops Whisper from repeating the same Telugu word over and over.

---

### Step 5: Mix Controlled Noise (Noise Augmentation)
1. Run the noise mixing script:
   ```bash
   python src/noise_augmenter.py
   ```
2. **What Happens Behind the Scenes:**
   - The code automatically trims silence at the start and end of your speech. *(Why? Because silent pauses mess up the energy calculation!)*
   - It mixes the traffic noise at 3 controlled loudness levels:
     - `snr_15db` (Mild noise — quiet office)
     - `snr_10db` (Moderate noise — street outside)
     - `snr_5db` (Heavy noise — crowded market / bus stand)
3. **Your Action:**
   - Put on headphones and play `data/audio/augmented_noisy/snr_10db/sample_01.wav`.
   - Confirm you can hear both your Telugu voice and the background traffic sound.

---

### Step 6: Clean the Audio Using Spectral Denoising
1. Run the cleaner script:
   ```bash
   python src/audio_cleaner.py
   ```
2. **What Happens Behind the Scenes:**
   - The script uses spectral subtraction (`noisereduce`) in `non_stationary` mode to strip out fluctuating traffic frequencies without muffling the Telugu speech.
3. **Your Action:**
   - Play `data/audio/denoised/sample_01_clean.wav`.
   - Notice how the background traffic noise disappears while your voice remains crisp.

---

### Step 7: Run Experiment 1 (For Your Final Report & Presentation)
1. Run the evaluation script:
   ```bash
   python src/evaluate_asr.py
   ```
2. **What it Generates:**
   - A clean terminal table showing Word Error Rate (WER) across Clean vs 15 dB vs 10 dB vs 5 dB noise.
   - A high-resolution graph saved at: `outputs/figures/experiment_1_wer_curve.png`.
3. **Your Action:**
   - **Save this graph!** This is Figure 1 for your IEEE report and Slide 5 for your project presentation.
   - Note the takeaway: *"Under severe 5 dB traffic noise, spectral denoising recovers ~19% Word Error Rate."*

---

### Step 8: Handing Off to Your Teammates
1. Message Member 2:
   > *"Member 1 here: The raw ASR transcriptions for our test audio are ready in `data/audio/`. You can use them to test how your normalizer handles noisy ASR errors."*
2. Message Member 3:
   > *"Member 1 here: `src/asr_engine.py` is tested and verified. You can import `TeluguASREngine` directly into your Streamlit `app.py` for live microphone transcription."*

---

## 🎯 Viva Defense Prep: Questions Your Professor Will Ask YOU

1. **Question:** *"Why did you trim silence before calculating Signal-to-Noise Ratio (SNR)?"*
   - **Your Answer:** *"If an audio clip has 1-2 seconds of silence at the start, including that silence pulls down the average root-mean-square (RMS) speech power. That causes the noise to be scaled at completely the wrong loudness. Trimming silence ensures SNR is computed solely on active speech frames."*
2. **Question:** *"What is the difference between stationary and non-stationary denoising?"*
   - **Your Answer:** *"Stationary denoising assumes the noise spectrum is constant over time, which works for ceiling fans or electrical hum. Non-stationary denoising recalculates noise profiles over sliding time windows, which is essential for dynamic street traffic and market babble."*
3. **Question:** *"Why did you set `condition_on_previous_text=False` in Whisper?"*
   - **Your Answer:** *"In low-resource Indic languages like Telugu, Whisper easily falls into infinite repetition loops where it outputs the same word repeatedly. Disabling previous text conditioning prevents this loop bug entirely."*
