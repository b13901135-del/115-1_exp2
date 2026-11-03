# 115-1 Biomedical Engineering Lab - Experiment 2

This repository contains the firmware and signal processing code for Experiment 2: Physiological Signal Detection Circuit Design, Real-time Wireless Telemetry, and EEG Digital Filtering.

## Contents
- **rduino_ecg_bluetooth.ino**: Arduino UNO firmware for 200 Hz non-blocking bio-potential acquisition (via pin A0), 10-bit to 8-bit fast right-shift conversion, and single-byte streaming over HC-05 Bluetooth Serial (Section 4).
- **eeg_freq_analysis.py**: Python script for automated BIOPAC raw data parsing, 2nd-order Butterworth IIR bandpass filtering (Delta, Theta, Alpha, Beta), two-sided FFT normalized magnitude spectrum calculation (with ftshift), and slide 48 standard comparative visualization (Section 6).