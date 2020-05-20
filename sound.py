from periphery import GPIO
from playsound import playsound

gpio6 = GPIO(6, "out")
gpio73 = GPIO(73, "out")


def main():
    gpio6.write(True)
    gpio73.write(False)
    playsound(entry.wav)
    gpio73.write(True)
    gpio6.write(False)


if __name__ == '__main__':
    main()
