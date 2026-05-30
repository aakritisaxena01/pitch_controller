import time
import numpy as np
from ultralytics import YOLO
from pymavlink import mavutil
import pyrealsense2 as rs

TARGET_CLASS_ID = 0         
THRESHOLD_PX    = 25.0      
YAW_PGAIN       = float(input("enter YAW_PGAIN: "))
YAW_IGAIN       = float(input("enter YAW_IGAIN: "))
YAW_DGAIN       = float(input("enter YAW_DGAIN: "))      
PITCH_PGAIN     = 0.1
PITCH_IGAIN     = 0.1
PITCH_DGAIN     = 0.1
SERVO_RANGE_DEG = 180
SERVO_NEUTRAL   = 90

model = YOLO("/home/aakriti/Downloads/best.pt")  #for hand detection
master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
master.wait_heartbeat()
print("Connected")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

for i in range(30):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    print(f"Frame {i}: color valid = {bool(color)}")  
print("Camera ready")
msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
if not msg:
    print("No heartbeat received — aborting")
    exit()
else:
    mode = mavutil.mode_string_v10(msg)
    armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    print(f"Current mode: {mode}, Armed: {bool(armed)}")

    mode_id = master.mode_mapping()['GUIDED']
    master.set_mode(mode_id)
    time.sleep(2)

    if not armed:
        master.arducopter_arm()
        master.motors_armed_wait()
        print("Armed")
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, 5
        )
        time.sleep(8)
    else:
        print("Already armed, starting tracking")

def set_yaw(angle_deg):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        abs(angle_deg),
        10,
        1 if angle_deg > 0 else -1,
        1,
        0, 0, 0
    )

def angle_to_pwm(angle_deg):
    angle_deg = max(0, min(SERVO_RANGE_DEG, angle_deg))
    return int(500 + (angle_deg / SERVO_RANGE_DEG) * 2000)

def set_servo_pwm(channel: int, pwm_us: int):
    pwm_us = max(500, min(2500, pwm_us))   # clamp to safe range
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,   # command id 183
        0,          # confirmation
        channel,    # param1: servo channel number
        pwm_us,     # param2: PWM value (µs)
        0, 0, 0, 0, 0   # unused params
    )
    print(f"  → Channel {channel} set to {pwm_us} us")

integral = 0.0
yintegral = 0.0
prev_y = 90.0
pitch_angle = 90.0
yaw_change = 0.0
prev_x = 0.0
INTEGRAL_CLAMP = 1000.0  

def pidp(error_y, pitch_angle, dt):
    global integral, prev_y
    p = error_y
    if dt == 0.0:
        dt = 0.034
    d = (pitch_angle - prev_y) / dt
    integral += error_y * dt
    integral = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral))  
    prev_y = pitch_angle
    print("integral term: ", PITCH_IGAIN * integral, "derivative term: ", PITCH_DGAIN * d, "proportional term: ", PITCH_PGAIN * p)
    return PITCH_PGAIN * p - PITCH_DGAIN * d + PITCH_IGAIN * integral

def pidy(error_x, yaw_change, dty):
    global yintegral, prev_x
    p = error_x
    if dty == 0.0:
        dty = 0.034
    d = (yaw_change - prev_x) / dty
    yintegral += error_x * dty
    yintegral = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, yintegral))  
    prev_x = yaw_change
    print("integral term: ", YAW_IGAIN * yintegral, "derivative term: ", YAW_DGAIN * d, "proportional term: ", YAW_PGAIN * p)
    return YAW_PGAIN * p - YAW_DGAIN * d + YAW_IGAIN * yintegral

set_servo_pwm(9, 1500)
i=0  
start_time = time.time()
p_dt = time.time()
y_dt = time.time()
time.sleep(2)
pitch_angle = 90.0
yaw_change = 0.0
fps = 0
no_target_frames = 0
try:
    while True:
        best_frame = None
        list1=[]
        list2=[]
        frame = None
        for k in range(2):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000) 
            except RuntimeError as e:
                print(f"Frame timeout, skipping: {e}")
                continue  
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            fps = fps+1
            h, w, _ = frame.shape
            frame_cx = w // 2
            frame_cy = h // 2
            results = model(frame, verbose=False)
            best_box  = None
            best_area = 0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) != TARGET_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_box  = (x1, y1, x2, y2)
                list1.append(best_area)
                list2.append(best_box)
        if frame is None:
            master.recv_match(blocking=False)
            continue
        if not list1 or max(list1) == 0:
            best_frame = None
        else:
            maxindex=list1.index(max(list1))
            best_frame=list2[maxindex]  
        error_x = 0
        error_y = 0
        
        if best_frame is not None:        
            x1, y1, x2, y2 = best_frame
            obj_cx = (x1 + x2) // 2
            obj_cy = (y1 + y2) // 2
            error_x = obj_cx - frame_cx   
            error_y = obj_cy - frame_cy   
            no_target_frames = 0                
            MAX_ANGLE_CHANGE = 1.5 
            MAX_YANGLE_CHANGE = 2 
            if abs(error_x) > THRESHOLD_PX:
                dty = time.time() - y_dt
                y_dt = time.time()
                yaw_change = pidy(error_x, yaw_change, dty)
                yaw_change = max(-MAX_YANGLE_CHANGE, min(MAX_YANGLE_CHANGE, yaw_change))
                set_yaw(yaw_change)
            if abs(error_y) > THRESHOLD_PX:
                dt   = time.time() - p_dt
                p_dt = time.time()
                pidc = pidp(error_y, pitch_angle, dt)            
                target_pitch = SERVO_NEUTRAL + pidc
                target_pitch = max(0, min(180, target_pitch))
                delta = target_pitch - pitch_angle
                delta = max(-MAX_ANGLE_CHANGE, min(MAX_ANGLE_CHANGE, delta))
                pitch_angle += delta                 
                pwm = angle_to_pwm(pitch_angle)
                set_servo_pwm(9, pwm)
                i=i+1
                elapsed = time.time() - start_time         
                if elapsed >= 1.0:
                    print("number of command sent per sec", i)
                    print("fps:", fps)
                    fps = 0
                    i=0
                    start_time = time.time()

        else:
            integral = 0.0
            yintegral = 0.0
            no_target_frames += 1
            if no_target_frames > 2:
                set_yaw(0)
finally:
    print("Switching to LOITER...")
    mode_id = master.mode_mapping()['LOITER']
    master.set_mode(mode_id)
    pipeline.stop()
