from periphery import GPIO
import simpleaudio.functionchecks as fc

gpio6 = GPIO(6, "out")
gpio73 = GPIO(73, "out")

def main():
    fc.LeftRightCheck.run()

if __name__ == '__main__':
    main()
