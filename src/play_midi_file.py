import pygame

pygame.mixer.init()
pygame.mixer.music.load("assets/audio/transcription.mid")
pygame.mixer.music.play()

# Attendre la fin de la lecture
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)