#!/usr/bin/env python3
import smbus
import time

# I2C bus
bus = smbus.SMBus(2)  # For Jetson boards, /dev/i2c-1 is typically used

# PCF8591 address
PCF8591_ADDRESS = 0x57

def setup():
    # No specific setup required for PCF8591, it's ready to read after initialization
    pass

def read_analog_input(channel):
    if channel < 0 or channel > 3:
        raise ValueError("Channel must be between 0 and 3")
    # Command to select the channel (0x40 is the base command, add channel number)
    bus.write_byte(PCF8591_ADDRESS, 0x50 | channel)
    time.sleep(0.1)  # Wait for the conversion to complete
    return bus.read_byte(PCF8591_ADDRESS)  # Read the ADC value

def direction():
    # Read analog values from channels
    x_value = read_analog_input(0)
    y_value = read_analog_input(2)
    button_value = read_analog_input(1)
    
    # Determine if the button is pressed
    button_pressed = button_value <= 30 	
    
    # Determine the joystick direction
    if x_value <= 30:
        x = -1  # Left
    elif x_value >= 225:
        x = 1   # Right
    else:
        x = 0

    if y_value <= 30:
        y = -1  # Up
    elif y_value >= 225:
        y = 1   # Down
    else:
        y = 0
    
    # Return (x, y) coordinates and button pressed state
    return (x_value, y_value, button_value)

def loop():
    status = ''
    while True:
        x, y, button_status = direction()
        # Form a status message
        #if button_pressed:
        #    button_status = "Button Pressed"
        #else:
        #    button_status = "Button Not Pressed"
        status = f"X: {x}, Y: {y}, {button_status}"
        print(status)
        time.sleep(0.1)  # Avoid too rapid polling

def destroy():
    pass

if __name__ == '__main__':  # Program starts here
    setup()
    try:
        loop()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed
        destroy()

