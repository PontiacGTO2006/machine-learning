'''
Filename: drawpad.py - assemebled using Python 3.12.2
This is the main 'runner' for the QuickDraw model. To run this
file, all you need is a mouse, trackpad, or (ideally) stylus that you can
draw on your screen with. I've already trained a model that's able
to recognize drawings of fruit and of shapes, they can be found in the 'models' folder
in this directory. Simply change the path on line 15 if you need to use a different
model.

To use the QuickDraw webcam model, simply run this code, use your mouse to draw the
shape (or the fruit) on the screen, and the model will recognize the category of the 
shape or fruit. There are three possible classes in this model version. If you seem to
be getting poor results, draw larger. Press Escape to close down the window.
'''
import pygame
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

model = keras.models.load_model("quickdraw_model/models/quickdraw_model_shapes_v2.keras")
CLASS_NAMES = ['circle', 'square', 'triangle'] # update class names if you change the dataset categories

pygame.init()
window_size = (800, 800)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("QuickDraw Model")

canvas = pygame.Surface(window_size) # create the game window canvas
canvas.fill((0, 0, 0))

drawing = False
pen_radius = 4
last_pos = None
prediction_text = ""

clock = pygame.time.Clock()

def predict_image():
    pygame.image.save(canvas, "temp.png")
    img = Image.open("temp.png").convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array, verbose=0)[0] # feed the drawn image through the model to be predicted upon
    label = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return f"{label} ({confidence:.2f})"
    
def draw(surface, start, end, width):
    if start and end:
        pygame.draw.line(surface, (255, 255, 255), start, end, width)


running = True
while running:
    clock.tick(30)

    for event in pygame.event.get(): # check for user keystrokes to clear or end the game
        if event.type == pygame.QUIT:
            running = False

      
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                canvas.fill((0, 0, 0))
                prediction_text = ""
            elif event.key == pygame.K_ESCAPE:
                running = False

    if drawing:
        current_pos = pygame.mouse.get_pos()
        draw(canvas, last_pos, current_pos, pen_radius * 2)
        last_pos = current_pos

    prediction_text = predict_image()

    screen.blit(canvas, (0, 0))

    # set up the label text to appear on screen
    font = pygame.font.SysFont(None, 36)
    pred_surface = font.render("Guess: " + prediction_text, True, (255, 255, 255))
    screen.blit(pred_surface, (10, 10))

    help_font = pygame.font.SysFont(None, 20)
    help_text = help_font.render("Space: Clear  |  Esc: Quit", True, (180, 180, 180))
    screen.blit(help_text, (10, 50))

    note_font = pygame.font.SysFont(None, 18)
    note_text = note_font.render("Note: Draw shapes of a larger size if you are getting poor results.", True, (180, 180, 180))
    screen.blit(note_text, (10, 70))

    pygame.display.flip()

pygame.quit()