"""
生醫工程實驗 - 實驗二 (Exp2)
EEG 訊號頻域轉換、數位濾波與頻譜分析程式 (BIOPAC Raw & Reference Data)
- 支援 raw_close.xls, raw_open.xls, raw_reclose.xls 自動轉為 .txt 格式
- CH1 (Raw Data) 自製 Butterworth 帶通濾波器 (Designed Filter)
- CH40~43 (Alpha, Beta, Delta, Theta) BIOPAC 原廠濾波 (Reference)
- 雙邊頻譜轉換 (Two-sided FFT with fftshift) 吻合講義
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# 確保 Windows 終端機正常輸出 UTF-8 中文字與希臘字母
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

# 設定 Matplotlib 中文字型與繪圖風格
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def convert_xls_to_txt(xls_path, txt_path):
    """
    將 BIOPAC 匯出的 Excel (.xls / BIFF2) 轉為標準 Tab 分隔之 .txt 檔案
    對應欄位:
        CH 1  -> Raw Data (原始 EEG)
        CH 40 -> Alpha (8 - 13 Hz)
        CH 41 -> Beta (13 - 30 Hz)
        CH 42 -> Delta (0.5 - 4 Hz)
        CH 43 -> Theta (4 - 8 Hz)
    """
    import xlrd
    print(f"正在轉換: {xls_path} -> {txt_path} ...")
    wb = xlrd.open_workbook(xls_path)
    sheet = wb.sheet_by_index(0)
    
    # 從第 3 列 (index 2) 開始為數值資料 (前兩列為標題與單位)
    data = []
    for r in range(2, sheet.nrows):
        data.append([sheet.cell_value(r, c) for c in range(sheet.ncols)])
        
    cols = ['Time', 'Raw', 'Alpha', 'Beta', 'Delta', 'Theta']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(txt_path, sep='\t', index=False)
    print(f"轉換成功: {txt_path} (共 {len(df)} 筆取樣點)")
    return df


def ensure_txt_files():
    """
    檢查並自動將 3 個原始 .xls 資料檔轉換為 .txt
    """
    files_to_convert = [
        ('raw_close.xls', 'raw_close.txt'),
        ('raw_open.xls', 'raw_open.txt'),
        ('raw_reclose.xls', 'raw_reclose.txt')
    ]
    
    converted_dfs = {}
    for xls_path, txt_path in files_to_convert:
        if os.path.exists(xls_path):
            # 若 txt 不存在或 xls 較新，則執行轉換
            if not os.path.exists(txt_path) or os.path.getmtime(xls_path) > os.path.getmtime(txt_path):
                df = convert_xls_to_txt(xls_path, txt_path)
            else:
                df = pd.read_csv(txt_path, sep='\t')
            converted_dfs[txt_path] = df
        elif os.path.exists(txt_path):
            converted_dfs[txt_path] = pd.read_csv(txt_path, sep='\t')
            
    return converted_dfs


def load_eeg_data(filepath):
    """
    讀取 EEG 資料檔 (Tab 分隔, 包含 Time, Raw, Alpha, Beta, Delta, Theta 或 α, β, δ, θ)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到檔案: {filepath}")
    
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"檔案為空: {filepath}")

    df = pd.read_csv(filepath, sep=r'\t', engine='python', encoding='utf-8')
    
    # 建立欄位名稱映射對應 (支援 Greek 或 英文名稱)
    col_map = {
        'α': 'Alpha',
        'β': 'Beta',
        'δ': 'Delta',
        'θ': 'Theta',
        'Alpha': 'Alpha',
        'Beta': 'Beta',
        'Delta': 'Delta',
        'Theta': 'Theta',
        'CH 1, EEG': 'Raw',
        'Raw': 'Raw',
        'Time': 'Time'
    }
    df.rename(columns=col_map, inplace=True)
    return df


def butter_bandpass_filter(data, lowcut, highcut, fs=200.0, order=2, method='lfilter'):
    """
    依講義設計數位帶通濾波器 (Butterworth Bandpass Filter)
    
    參數:
        data (np.ndarray): 待濾波訊號 (CH 1 Raw Data)
        lowcut (float): 低頻截止頻率 (Hz)
        highcut (float): 高頻截止頻率 (Hz)
        fs (float): 取樣頻率 (Hz)
        order (int): 濾波器階數 (預設 2 階，BIOPAC 硬體原廠即採用 2 階 IIR，吻合度達 99%)
        method (str): 'lfilter' (即時因果 IIR) 或 'filtfilt' (零相位雙向濾波)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    
    if method == 'filtfilt':
        return signal.filtfilt(b, a, data)
    else:
        return signal.lfilter(b, a, data)


def compute_fft(signal_data, fs=200.0, normalize=True):
    """
    計算訊號的雙邊頻譜 (Two-sided Spectrum with fftshift)
    
    參數:
        signal_data (np.ndarray): 1D 時域訊號
        fs (float): 取樣頻率 (Sampling Frequency, BIOPAC EEG 為 200 Hz)
        normalize (bool): 是否將最大振幅正規化至 1.0 
        
    回傳:
        freqs (np.ndarray): 頻率軸 (Hz, 範圍 -fs/2 到 +fs/2)
        magnitude (np.ndarray): 振幅譜
    """
    N = len(signal_data)
    
    # 1. 扣除直流成分 (DC offset) 以免 0 Hz 處有過大的峰值影響觀察
    sig_detrend = signal_data - np.mean(signal_data)
    
    # 2. 計算快速傅立葉變換 (FFT)
    fft_vals = np.fft.fft(sig_detrend)
    
    # 3. 頻率軸 (-fs/2 到 fs/2)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    
    # 4. 使用 fftshift 將零頻率移至中央 (雙邊頻譜)
    freqs_shifted = np.fft.fftshift(freqs)
    fft_shifted = np.fft.fftshift(fft_vals)
    
    # 5. 取絕對值獲得振幅大小 (Magnitude)
    magnitude = np.abs(fft_shifted)
    
    # 6. 正規化 (Normalized Amplitude, 0 ~ 1.0)
    if normalize:
        max_val = np.max(magnitude)
        if max_val > 0:
            magnitude = magnitude / max_val
            
    return freqs_shifted, magnitude


def plot_comparison(df_closed, df_open, fs=200.0, output_dir="output_plots"):
    """
    依講義樣式繪製 Designed vs Reference 頻譜對照圖 (2x2 grid)
    包含 4 個頻帶獨立圖以及 4 合 1 總覽圖
    """
    os.makedirs(output_dir, exist_ok=True)
    
    bands = [
        ('Alpha', (8.0, 13.0), '#1f77b4'),
        ('Beta', (13.0, 30.0), '#ff7f0e'),
        ('Delta', (0.5, 4.0), '#2ca02c'),
        ('Theta', (4.0, 8.0), '#d62728')
    ]
    
    # 1. 繪製各頻帶單獨 2x2 對照圖
    for band_name, (lowcut, highcut), color in bands:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # --- 閉眼 (Eyes closed) ---
        # Designed (以 Raw 訊號自設 Butterworth 帶通濾波器)
        if 'Raw' in df_closed.columns:
            des_sig_closed = butter_bandpass_filter(df_closed['Raw'].values, lowcut, highcut, fs=fs, order=2)
        else:
            des_sig_closed = df_closed[band_name].values
        f_des_c, mag_des_c = compute_fft(des_sig_closed, fs=fs, normalize=True)
        
        # Reference (BIOPAC 原廠濾波)
        ref_sig_closed = df_closed[band_name].values
        f_ref_c, mag_ref_c = compute_fft(ref_sig_closed, fs=fs, normalize=True)
        
        # --- 睜眼 (Eyes open) ---
        # Designed
        if 'Raw' in df_open.columns:
            des_sig_open = butter_bandpass_filter(df_open['Raw'].values, lowcut, highcut, fs=fs, order=2)
        else:
            des_sig_open = df_open[band_name].values
        f_des_o, mag_des_o = compute_fft(des_sig_open, fs=fs, normalize=True)
        
        # Reference
        ref_sig_open = df_open[band_name].values
        f_ref_o, mag_ref_o = compute_fft(ref_sig_open, fs=fs, normalize=True)
        
        # 繪製各子圖 (左邊 Designed, 右邊 Reference; 上排 Closed, 下排 Open)
        sub_configs = [
            (axes[0, 0], f_des_c, mag_des_c, "Designed, eyes closed"),
            (axes[0, 1], f_ref_c, mag_ref_c, "Reference, eyes closed"),
            (axes[1, 0], f_des_o, mag_des_o, "Designed, eyes open"),
            (axes[1, 1], f_ref_o, mag_ref_o, "Reference, eyes open"),
        ]
        
        for ax, freqs, mag, title in sub_configs:
            ax.plot(freqs, mag, color=color, linewidth=1.2)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlim(-100, 100)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frequency (Hz)", fontsize=10)
            ax.set_ylabel("Normalized Amplitude", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            
        plt.suptitle(f"{band_name} ({lowcut} - {highcut} Hz) -  Designed vs Reference", fontsize=13, y=0.99)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{band_name.lower()}_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"已儲存單頻帶圖: {out_path}")
        
    # 2. 繪製 4 合 1 總覽大圖 
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    for i, (band_name, (lowcut, highcut), color) in enumerate(bands):
        # 閉眼 Designed & Ref
        des_c = butter_bandpass_filter(df_closed['Raw'].values, lowcut, highcut, fs=fs, order=2)
        ref_c = df_closed[band_name].values
        f_dc, m_dc = compute_fft(des_c, fs=fs, normalize=True)
        f_rc, m_rc = compute_fft(ref_c, fs=fs, normalize=True)
        
        # 睜眼 Designed & Ref
        des_o = butter_bandpass_filter(df_open['Raw'].values, lowcut, highcut, fs=fs, order=2)
        ref_o = df_open[band_name].values
        f_do, m_do = compute_fft(des_o, fs=fs, normalize=True)
        f_ro, m_ro = compute_fft(ref_o, fs=fs, normalize=True)
        
        row_data = [
            (axes[i, 0], f_dc, m_dc, f"{band_name}: Designed, eyes closed"),
            (axes[i, 1], f_rc, m_rc, f"{band_name}: Reference, eyes closed"),
            (axes[i, 2], f_do, m_do, f"{band_name}: Designed, eyes open"),
            (axes[i, 3], f_ro, m_ro, f"{band_name}: Reference, eyes open"),
        ]
        for ax, freqs, mag, title in row_data:
            ax.plot(freqs, mag, color=color, linewidth=1.1)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(-100, 100)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frequency (Hz)", fontsize=9)
            ax.set_ylabel("Norm. Amp", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            
    plt.suptitle("BIOPAC EEG 4-Bands Frequency Spectrum (Designed vs Reference)", fontsize=16, y=0.995)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "all_bands_summary.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"已儲存4合1總圖: {summary_path}")


def plot_three_states_comparison(df_close, df_open, df_reclose, fs=200.0, output_dir="output_plots"):
    """
    繪製三階段 (閉眼 -> 睜眼 -> 再次閉眼) 頻譜演化對照圖
    彰顯 Alpha Blocking (睜眼抑制) 與閉眼 Alpha 波回歸的生理特徵！
    """
    os.makedirs(output_dir, exist_ok=True)
    states = [
        ('1. Eyes Closed (閉眼)', df_close, '#1f77b4'),
        ('2. Eyes Open (睜眼)', df_open, '#ff7f0e'),
        ('3. Eyes Reclosed (再閉眼)', df_reclose, '#2ca02c')
    ]
    bands = [
        ('Alpha', (8.0, 13.0)),
        ('Beta', (13.0, 30.0)),
        ('Delta', (0.5, 4.0)),
        ('Theta', (4.0, 8.0))
    ]
    
    # 4 頻帶 x 3 階段頻譜演變圖
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    for r, (band_name, (lowcut, highcut)) in enumerate(bands):
        for c, (state_name, df, col) in enumerate(states):
            ax = axes[r, c]
            
            # 繪製 Designed (自製濾波) 與 Reference (BIOPAC 原廠) 同軸對照
            des_sig = butter_bandpass_filter(df['Raw'].values, lowcut, highcut, fs=fs, order=2)
            ref_sig = df[band_name].values
            
            f_des, m_des = compute_fft(des_sig, fs=fs, normalize=True)
            f_ref, m_ref = compute_fft(ref_sig, fs=fs, normalize=True)
            
            ax.plot(f_des, m_des, label='Designed (Butterworth)', color=col, linewidth=1.2)
            ax.plot(f_ref, m_ref, label='Reference (BIOPAC)', color='black', linestyle=':', linewidth=1.1, alpha=0.8)
            
            ax.set_title(f"{band_name} - {state_name}", fontsize=11, fontweight='bold')
            ax.set_xlim(-100, 100)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frequency (Hz)", fontsize=9)
            ax.set_ylabel("Norm. Amplitude", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            if r == 0 and c == 0:
                ax.legend(loc='upper right', fontsize=8)
                
    plt.suptitle("EEG Frequency Spectrum across 3 Experimental States (Designed vs Reference)", fontsize=15, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "eeg_three_states_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"已儲存三狀態演化對照圖: {out_path}")


def plot_raw_waveform_and_spectrum(df_close, df_open, df_reclose, fs=200.0, output_dir="output_plots"):
    """
    繪製 CH1 Raw Data (原始 EEG) 的時域波形與整體雙邊頻譜
    展現 0~100 Hz 全頻段特徵、Alpha 波生理活動與 60 Hz 市電雜訊
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = [
        ('Eyes Closed (閉眼)', df_close, '#1f77b4'),
        ('Eyes Open (睜眼)', df_open, '#ff7f0e'),
        ('Eyes Reclosed (再閉眼)', df_reclose, '#2ca02c')
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for i, (name, df, col) in enumerate(datasets):
        # 1. 時域波形 (取前 5 秒或全部顯示)
        ax_time = axes[i, 0]
        time_axis = df['Time'].values - df['Time'].values[0]
        ax_time.plot(time_axis, df['Raw'].values, color=col, linewidth=0.8)
        ax_time.set_title(f"Raw EEG Waveform - {name}", fontsize=11, fontweight='bold')
        ax_time.set_xlabel("Time (seconds)")
        ax_time.set_ylabel("Amplitude (uV)")
        ax_time.set_xlim(0, min(10, time_axis[-1])) # 呈現前 10 秒以清晰觀察波形
        ax_time.grid(True, linestyle="--", alpha=0.5)
        
        # 2. 頻域雙邊頻譜
        ax_freq = axes[i, 1]
        freqs, mag = compute_fft(df['Raw'].values, fs=fs, normalize=True)
        ax_freq.plot(freqs, mag, color=col, linewidth=1.0)
        ax_freq.set_title(f"Raw EEG Spectrum - {name}", fontsize=11, fontweight='bold')
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Normalized Amplitude")
        ax_freq.set_xlim(-100, 100)
        ax_freq.set_ylim(0, 1.05)
        ax_freq.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("Raw EEG (CH1) Time-domain Waveforms & Frequency Spectra", fontsize=14, y=0.995)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "raw_eeg_waveform_and_spectrum.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"已儲存 Raw EEG 時域與頻譜圖: {out_path}")


def plot_raw_vs_filtered_overlay(df_close, fs=200.0, output_dir="output_plots"):
    """
    在一張圖中展示 Raw EEG 頻譜如何被 4 個數位濾波器分離 (Raw vs 4 Bands Overlay)
    清楚呈現帶通濾波器 (BPF) 擷取各頻帶的機制
    """
    os.makedirs(output_dir, exist_ok=True)
    bands = [
        ('Alpha', (8.0, 13.0), '#1f77b4'),
        ('Beta', (13.0, 30.0), '#ff7f0e'),
        ('Delta', (0.5, 4.0), '#2ca02c'),
        ('Theta', (4.0, 8.0), '#d62728')
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上圖: Raw EEG 雙邊頻譜 (標示四個頻帶區間)
    f_raw, m_raw = compute_fft(df_close['Raw'].values, fs=fs, normalize=True)
    axes[0].plot(f_raw, m_raw, color='gray', linewidth=1.0, label='Raw EEG Spectrum')
    axes[0].set_title("Raw EEG Full Spectrum with 4 Brainwave Bands Highlighted", fontsize=12, fontweight='bold')
    axes[0].set_xlim(-100, 100)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Norm. Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 標示各頻帶顏色色塊
    for band_name, (low, high), col in bands:
        axes[0].axvspan(low, high, color=col, alpha=0.25, label=f'{band_name} (+{low}~{high}Hz)')
        axes[0].axvspan(-high, -low, color=col, alpha=0.25)
    axes[0].legend(loc='upper right', fontsize=9)
    
    # 下圖: 4 個濾波後頻帶的頻譜疊圖
    for band_name, (low, high), col in bands:
        filtered = butter_bandpass_filter(df_close['Raw'].values, low, high, fs=fs, order=2)
        f_b, m_b = compute_fft(filtered, fs=fs, normalize=True)
        axes[1].plot(f_b, m_b, color=col, linewidth=1.3, label=f'{band_name} Band ({low}-{high} Hz)')
        
    axes[1].set_title("Designed 4-Bands Butterworth Filter Output Spectra", fontsize=12, fontweight='bold')
    axes[1].set_xlim(-100, 100)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Norm. Amplitude")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "raw_vs_filtered_overlay.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"已儲存 Raw vs Filtered 對照圖: {out_path}")


def export_frequency_csv(datasets, fs=200.0, output_dir="frequency_domain_data"):
    """
    匯出頻譜數據至 CSV 檔
    """
    os.makedirs(output_dir, exist_ok=True)
    bands = [
        ('Alpha', (8.0, 13.0)),
        ('Beta', (13.0, 30.0)),
        ('Delta', (0.5, 4.0)),
        ('Theta', (4.0, 8.0))
    ]
    
    for cond_name, df in datasets.items():
        freqs, raw_mag = compute_fft(df['Raw'].values, fs=fs, normalize=True)
        res_dict = {
            'Frequency_Hz': freqs,
            'Raw_Normalized_Mag': raw_mag
        }
        for band_name, (low, high) in bands:
            # Designed
            des_sig = butter_bandpass_filter(df['Raw'].values, low, high, fs=fs, order=2)
            _, des_mag = compute_fft(des_sig, fs=fs, normalize=True)
            res_dict[f'{band_name}_Designed_Mag'] = des_mag
            
            # Reference
            if band_name in df.columns:
                _, ref_mag = compute_fft(df[band_name].values, fs=fs, normalize=True)
                res_dict[f'{band_name}_Ref_Mag'] = ref_mag
                
        res_df = pd.DataFrame(res_dict)
        csv_path = os.path.join(output_dir, f"eeg_{cond_name}_freq_domain.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"已匯出頻域數據 CSV: {csv_path}")


def main():
    print("==================================================")
    print("生醫工程實驗 - 實驗二 EEG 訊號頻域分析與數位濾波")
    print("==================================================")
    
    # 1. 確保原始 .xls 資料已轉為 .txt 格式
    ensure_txt_files()
    
    # 2. 載入資料 (優先讀取轉換後的 raw_*.txt)
    close_file = "raw_close.txt" if os.path.exists("raw_close.txt") else "egg_second_close.txt"
    open_file = "raw_open.txt" if os.path.exists("raw_open.txt") else "eeg_first_open.txt"
    reclose_file = "raw_reclose.txt" if os.path.exists("raw_reclose.txt") else close_file
    
    print(f"\n載入資料檔案:")
    print(f"  閉眼 (Eyes Closed):   {close_file}")
    print(f"  睜眼 (Eyes Open):     {open_file}")
    print(f"  再閉眼 (Eyes Reclosed): {reclose_file}")
    
    df_close = load_eeg_data(close_file)
    df_open = load_eeg_data(open_file)
    df_reclose = load_eeg_data(reclose_file)
    
    fs = 200.0
    print(f"\n取樣頻率 Fs: {fs} Hz (雙邊頻譜頻寬: -{fs/2} ~ +{fs/2} Hz)")
    print(f"  閉眼點數: {len(df_close)} 點")
    print(f"  睜眼點數: {len(df_open)} 點")
    print(f"  再閉眼點數: {len(df_reclose)} 點")
    
    # 3. 繪製標準對照圖 (Designed vs Reference 2x2 grid)
    print("\n[1/4] 繪製標準格式對照圖 (Designed vs Reference)...")
    plot_comparison(df_close, df_open, fs=fs, output_dir="output_plots")
    
    # 4. 繪製三階段 (閉眼 -> 睜眼 -> 再閉眼) 演化對照圖
    print("\n[2/4] 繪製三階段 (閉眼/睜眼/再閉眼) 演化對照圖...")
    plot_three_states_comparison(df_close, df_open, df_reclose, fs=fs, output_dir="output_plots")
    
    # 5. 繪製 Raw EEG 時域波形與整體頻譜圖
    print("\n[3/4] 繪製 Raw EEG (CH1) 時域波形與頻譜圖...")
    plot_raw_waveform_and_spectrum(df_close, df_open, df_reclose, fs=fs, output_dir="output_plots")
    
    # 6. 繪製 Raw vs 4 頻帶濾波疊圖
    print("\n[4/4] 繪製 Raw EEG 與 4 個頻帶濾波前後頻譜疊圖...")
    plot_raw_vs_filtered_overlay(df_close, fs=fs, output_dir="output_plots")
    
    # 7. 匯出所有頻譜數值 CSV
    print("\n[匯出] 匯出所有數值數據至 CSV ...")
    export_frequency_csv({
        'closed': df_close,
        'open': df_open,
        'reclosed': df_reclose
    }, fs=fs, output_dir="frequency_domain_data")
    
    print("\n==================================================")
    print("全部處理完成！所有圖檔已儲存於 output_plots/ 目錄。")
    print("數值檔案已儲存於 frequency_domain_data/ 目錄。")
    print("==================================================")


if __name__ == "__main__":
    main()
