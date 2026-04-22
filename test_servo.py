#!/usr/bin/env python3
"""
debug_servo.py
Skrip sederhana untuk menguji pergerakan 1 motor servo via PCA9685.
Tanpa ROS2, murni menggunakan library Adafruit.
"""

import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

print("Mempersiapkan I2C dan modul PCA9685...")
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Frekuensi standar servo RC

# Inisialisasi 1 motor servo HANYA di Channel 0
# Menggunakan min_pulse=600 dan max_pulse=2400 (standar aman MG996R)
print("Menginisialisasi servo di Channel 0...")
servo_uji = servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400)

def set_servo(sudut):
    """Fungsi pembantu untuk memutar servo ke sudut tertentu"""
    try:
        servo_uji.angle = sudut
    except Exception as e:
        print(f"Error memutar servo: {e}")

try:
    print("=====================================================")
    print("Memulai tes pergerakan 1 servo (Looping).")
    print("Tekan Ctrl + C di keyboard untuk menghentikan program.")
    print("=====================================================")
    
    # 1. Posisikan ke tengah (90 derajat) sebagai pemanasan
    print("Posisi awal (90 derajat)...")
    set_servo(90)
    time.sleep(2)

    # 2. Looping pergerakan bolak-balik
    while True:
        print("Menyapu ke 135 derajat...")
        for sudut in range(90, 136, 2):
            set_servo(sudut)
            time.sleep(0.02) # Delay kecil agar pergerakan mulus
        time.sleep(0.5)

        print("Menyapu ke 45 derajat...")
        for sudut in range(135, 44, -2):
            set_servo(sudut)
            time.sleep(0.02)
        time.sleep(0.5)

        print("Kembali ke 90 derajat...")
        for sudut in range(45, 91, 2):
            set_servo(sudut)
            time.sleep(0.02)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nProgram dihentikan oleh pengguna (Ctrl+C).")

finally:
    # 3. Sangat Penting: Matikan aliran listrik ke motor agar tidak panas
    print("Mematikan torsi servo (Release)...")
    servo_uji.angle = None
    pca.deinit()
    print("Selesai. Hardware aman ditutup.")