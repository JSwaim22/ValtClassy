import gstreamer
from periphery import GPIO
from playsound import playsound

gpio6 = GPIO(6, "out")
gpio73 = GPIO(73, "out")


def main():
    gpio6.write(True)
    gpio73.write(False)
    playsound("welcomeT.wav")
    gpio73.write(True)
    gpio6.write(False)
    #playsound("entry.wav")
    #gpio6.write(True)
    #gpio73.write(False)
    #playsound("package.wav")
    #gpio73.write(True)
    #gpio6.write(False)
    #playsound("goodday.wav")
    #gpio6.write(True)
    #gpio73.write(False)
    #playsound("key.wav")
    #gpio73.write(True)
    #gpio6.write(False)
    #playsound("denied.wav")
    #gpio6.write(True)
    #gpio73.write(False)
    #playsound("stay.wav")
    #gpio73.write(True)
    #gpio6.write(False)
    #playsound("parcel.wav")
    #gpio6.write(True)
    #gpio73.write(False)
    

if __name__ == '__main__':
    main()
