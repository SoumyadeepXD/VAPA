import time
import math
try:
    import smbus2
except ImportError:
    try:
        import smbus as smbus2
    except ImportError:
        smbus2 = None

class VAPAActuatorController:
    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    # Channel Map
    CHANNELS = {
        # Fingers: 5x MG996R Servos (CH 0 to CH 4)
        "finger_thumb": 0,
        "finger_index": 1,
        "finger_middle": 2,
        "finger_ring": 3,
        "finger_pinky": 4,
        # Heavy Joint Arm: 3x DS3225 Servos (CH 5 to CH 7)
        "arm_base_yaw": 5,
        "arm_shoulder_pitch": 6,
        "arm_elbow_pitch": 7,
        # Wrist Pitch: 1x DS3218 Servo (CH 8)
        "wrist_pitch": 8
    }

    def __init__(self, bus_num=1, address=0x40, freq=50):
        if smbus2 is None:
            raise RuntimeError("smbus2 is not installed. Install via 'pip install smbus2'.")
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        self._reset()
        self.set_pwm_freq(freq)

    def _reset(self):
        self.bus.write_byte_data(self.address, self.MODE1, 0x00)
        time.sleep(0.01)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0 / 4096.0 / float(freq_hz) - 1.0
        prescale = int(math.floor(prescaleval + 0.5))
        oldmode = self.bus.read_byte_data(self.address, self.MODE1)
        newmode = (oldmode & 0x7F) | 0x10
        self.bus.write_byte_data(self.address, self.MODE1, newmode)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.MODE1, oldmode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.MODE1, oldmode | 0x80)

    def set_servo_angle(self, joint_name, angle_deg, min_us=500, max_us=2500):
        if joint_name not in self.CHANNELS:
            raise KeyError(f"Joint '{joint_name}' not found in channel mapping.")
        
        channel = self.CHANNELS[joint_name]
        angle_deg = max(0.0, min(180.0, angle_deg))
        pulse_us = min_us + (angle_deg / 180.0) * (max_us - min_us)
        ticks = int((pulse_us / 20000.0) * 4096)
        
        base_reg = self.LED0_ON_L + 4 * channel
        self.bus.write_i2c_block_data(
            self.address, 
            base_reg, 
            [0, 0, ticks & 0xFF, ticks >> 8]
        )

if __name__ == "__main__":
    actuators = VAPAActuatorController()
    print("[INFO] Testing MG996R Finger Servo (CH 0)...")
    actuators.set_servo_angle("finger_thumb", 90)
    
    print("[INFO] Testing DS3225 Arm Base Servo (CH 5)...")
    actuators.set_servo_angle("arm_base_yaw", 45)
    
    print("[INFO] Testing DS3218 Wrist Servo (CH 8)...")
    actuators.set_servo_angle("wrist_pitch", 30)
