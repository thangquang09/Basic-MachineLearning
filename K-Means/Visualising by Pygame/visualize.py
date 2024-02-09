import pygame
from random import randint
import math
from sklearn.cluster import KMeans
import numpy as np

def distance(P1, P2):
    return math.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2)

# initial pygame

pygame.init()

# set up size of window

screen = pygame.display.set_mode((1280, 700))

pygame.display.set_caption('Kmeans Visualization')

running = True

clock = pygame.time.Clock()
# colors
BACKGROUND = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (128, 128, 128)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)
NAVY = (0, 0, 128)
BACKGROUND_PANEL = (249, 255, 230) # yellow
colors = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, GRAY, MAROON, OLIVE, PURPLE, TEAL, NAVY]
# text
font = pygame.font.SysFont('sans', 40) # font sans, size 40
font_small = pygame.font.SysFont('sans', 20) # font sans, size 20
def makefont(text, color, size = 40):
    font = pygame.font.SysFont('sans', size)
    return font.render(text, True, color)


text_plus = makefont('+', WHITE) # True is AA
text_minus = makefont('-', WHITE)
text_run = makefont('RUN', WHITE)
text_random = makefont('RANDOM', WHITE, 30)
text_algorithm = makefont('Algorithm', WHITE)
text__reset = makefont('RESET', WHITE)
k_point = 0
point_in_panel = [] # Store points when click in panel
clusters = [] # Store points which are clusters
labels = []

while running:
    clock.tick(60)
    screen.fill(BACKGROUND)
    mouse_x, mouse_y = pygame.mouse.get_pos() # location of mouse
    # Draw interface
    # Draw Panel
    full_screen = pygame.draw.rect(screen, BLACK, (50, 50, 740, 500)) # (x, y, width, height)
    panel = pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55, 730, 490))

    # K + button
    Kplus_button = pygame.draw.rect(screen, BLACK, (850, 50, 50, 50)) # draw button
    screen.blit(text_plus, (Kplus_button.x + 15, Kplus_button.y)) # draw text

    # K - button
    Kminus_button = pygame.draw.rect(screen, BLACK, (950, 50, 50, 50))
    screen.blit(text_minus, (Kminus_button.x + 20, Kminus_button.y))

    # Run button
    run_button = pygame.draw.rect(screen, BLACK, (850, 150, 150, 50))
    screen.blit(text_run, (run_button.x + 40, run_button.y + 5))

    # Random button
    random_button = pygame.draw.rect(screen, BLACK, (850, 250, 150, 50))
    screen.blit(text_random, (random_button.x + 17, random_button.y + 5))

    # Algorithm button
    algorithm_button = pygame.draw.rect(screen, BLACK, (850, 400, 150, 50))
    screen.blit(text_algorithm, (algorithm_button.x + 5, algorithm_button.y + 5))

    # Reset button
    reset_button = pygame.draw.rect(screen, BLACK, (850, 500, 150, 50))
    screen.blit(text__reset, (reset_button.x + 15, reset_button.y + 5))


    # K noti
    text_k = font.render('K = ' + str(k_point) + ' (max 12)', True, BLACK)
    screen.blit(text_k, (1025, 50))

    # draw mouse position when mouse is in panel
    if panel.x <= mouse_x <= panel.x + panel.width and panel.y <= mouse_y <= panel.y + panel.height:
        text_mouse = font_small.render('(' + str(mouse_x - panel.x) + ', ' + str(mouse_y - panel.y) + ')', True, BLACK)
        screen.blit(text_mouse, (mouse_x + 15, mouse_y))

    # end draw interface

    # for loop using for process button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN: # when mouse click
            # change K button +
            if 850 < mouse_x < 900 and 50 < mouse_y < 100:
                # print('press K +')
                if k_point < 12:
                    k_point += 1
            # change K button -
            if 950 < mouse_x < 1000 and 50 < mouse_y < 100:
                if k_point > 0:
                    k_point -= 1
            # run button
            if 850 < mouse_x < 1000 and 150 < mouse_y < 200:
                if clusters == []:
                    continue
                labels = []
                # assign points to closest clusters
                for p in point_in_panel:
                    dis_min = 1e9
                    label = -1 # index of cluster
                    for c in clusters:
                        if distance(p, c) < dis_min:
                            dis_min = distance(p, c)
                            label = clusters.index(c)
                    labels.append(label)
                # print('press run')

                # update clusters
                for i in range(k_point):
                    sum_x, sum_y = 0, 0
                    count = 0
                    for j in range(len(point_in_panel)):
                        if labels[j] == i:
                            sum_x += point_in_panel[j][0]
                            sum_y += point_in_panel[j][1]
                            count += 1
                    if count != 0:
                        clusters[i] = [sum_x / count, sum_y / count]

            # random button
            if 850 < mouse_x < 1000 and 250 < mouse_y < 300:
                clusters = []
                labels = []
                for i in range(k_point):
                    random_point = [randint(0, panel.width), randint(0, panel.height)]
                    clusters.append(random_point)
                    # print(clusters)
            # algorithm button
            if 850 < mouse_x < 1000 and 400 < mouse_y < 450:
                if k_point == 0 or clusters == [] or point_in_panel == []:
                    continue
                kmeans = KMeans(n_clusters=k_point, init='k-means++', n_init=10).fit(point_in_panel)
                labels = list(kmeans.predict(point_in_panel))
                clusters = list(kmeans.cluster_centers_)
                clusters = [item.tolist() for item in clusters]
                # print('press Algorithm')
            # reset button
            if 850 < mouse_x < 1000 and 500 < mouse_y < 550:
                labels = []
                clusters = []
                point_in_panel = []
                k_point = 0
                wcss_point = 0
                # print('press reset')
            # click in panel
            if panel.x <= mouse_x <= panel.x + panel.width and panel.y <= mouse_y <= panel.y + panel.height:
                labels = []
                point = [mouse_x - panel.x, mouse_y - panel.y]
                point_in_panel.append(point)
                # print(point_in_panel)
    # draw points
    for i in range(len(point_in_panel)):
        pygame.draw.circle(screen, BLACK, (point_in_panel[i][0] + panel.x, point_in_panel[i][1] + panel.y), 6)
        if labels == []:
            pygame.draw.circle(screen, WHITE, (point_in_panel[i][0] + panel.x, point_in_panel[i][1] + panel.y), 4)
        else:
            pygame.draw.circle(screen, colors[labels[i]], (point_in_panel[i][0] + panel.x, point_in_panel[i][1] + panel.y), 4)
    # draw clusters
    for i in range(len(clusters)):
        pygame.draw.circle(screen, colors[i], (int(clusters[i][0]) + panel.x, int(clusters[i][1]) + panel.y), 9)

    # WCSS noti
    wcss_point = 0
    if clusters != [] and labels != []:
        for i in range(len(point_in_panel)):
            wcss_point += distance(point_in_panel[i], clusters[labels[i]])

    text_wcss = font.render('WCSS = ' + str(int(wcss_point)), True, RED)
    screen.blit(text_wcss, (865, 325))

    pygame.display.flip()

pygame.quit()