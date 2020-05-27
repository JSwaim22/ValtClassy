from periphery import GPIO
import simpleaudio as sa

def main():
    wave_obj = sa.WaveObject.from_wave_file("welcome.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

if __name__ == '__main__':
    main()
