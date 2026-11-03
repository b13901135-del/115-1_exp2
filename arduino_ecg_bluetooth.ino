#include <SoftwareSerial.h>
#include <math.h>

SoftwareSerial BTSerial(2, 3); // RX = Pin 2, TX = Pin 3

const long baud_rate = 9600;
const int ecg_pin = A0;

bool start_recv = false;       // 控制是否開始傳送的開關
unsigned long last_sample_time = 0;

// 設定取樣率：5000 微秒 (us) = 5 毫秒 (ms) = 200 Hz 取樣率
const unsigned long sample_interval_us = 5000; 

// 測試模式開關：true 為傳送假的正弦波測試，false 為讀取真實 A0 心電圖
bool test_mode = false; 
float angle = 0.0;

void setup() {
  Serial.begin(baud_rate);
  BTSerial.begin(baud_rate);
  pinMode(ecg_pin, INPUT);
}

void loop() {
  // 1. 檢查手機有沒有傳來控制指令
  if (BTSerial.available()) {
    char received_char = BTSerial.read();
    Serial.print("Command: ");
    Serial.println(received_char);
    
    // 切換開始/停止狀態
    start_recv = !start_recv; 
  }

  // 2. 如果開關開啟，且時間達到取樣間隔（每 5ms 執行一次 = 200 Hz）
  if (start_recv && (micros() - last_sample_time >= sample_interval_us)) {
    last_sample_time = micros(); // 更新時間紀錄

    byte value = 0;

    if (test_mode) {
      // 【測試模式】產生 Sin 波 (0~255)
      value = (byte)((sin(angle) + 1.0) * 127.5);
      angle += 0.1;
      if (angle >= 2 * PI) angle -= 2 * PI;
    } else {
      // 【真實模式】讀取 A0 並將 0~1023 轉為 0~255 (1 Byte)
      // 使用 >> 2 (右移兩位元，等同除以 4) 比 map() 快非常多，適合高頻取樣
      value = analogRead(ecg_pin) >> 2; 
    }

    // 3. 傳送單一 Byte 給手機 App
    BTSerial.write(value);
  }
}

