import pygame

pygame.mixer.init()
pygame.mixer.music.load("dataset/maestro-v3.0.0-midi/2008/MIDI-Unprocessed_10_R3_2008_01-05_ORIG_MID--AUDIO_10_R3_2008_wav--4.midi")
pygame.mixer.music.play()

# Attendre la fin de la lecture
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)