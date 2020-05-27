from periphery import GPIO
from playsound import playsound

gpio6 = GPIO(6, "out")
gpio73 = GPIO(73, "out")


def main():
    gpio73.write(True)
    gpio6.write(False)
    playsound('~/ValtWAV/welcome.wav')
    gpio6.write(True)
    gpio73.write(False)
    playsound('~/ValtWAV/package.wav')
    gpio73.write(True)
    gpio6.write(False)
    playsound('~/ValtWAV/goodday.wav')
    gpio6.write(True)
    gpio73.write(False)
    playsound('~/ValtWAV/key.wav')
    gpio73.write(True)
    gpio6.write(False)
    playsound('~/ValtWAV/denied.wav')
    gpio6.write(True)
    gpio73.write(False)
    playsound('~/ValtWAV/stay.wav')
    gpio73.write(True)
    gpio6.write(False)
    playsound('~/ValtWAV/parcel.wav')
    gpio6.write(True)
    gpio73.write(False)
    

if __name__ == '__main__':
    main()
