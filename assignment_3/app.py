import cv2
import numpy as np

def sobel_edge_detection (image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (3, 3), 0)
    sobel = cv2.Sobel(blur, cv2.CV_64F, dx = 1, dy = 1, ksize=1)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite("photos/sobel_edges.png", sobel)
    return sobel


def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(grayscale, (3, 3), 0)
    edges = cv2.Canny(blur, threshold_1, threshold_2)
    cv2.imwrite("photos/canny_edges.png", edges)
    return edges

def template_match(image, template):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = gray_template.shape[::-1]
    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("photos/template_match.png", image)
    return image


def resize(image, scale_factor, up_or_down):
    if up_or_down == "up":
        resized = cv2.pyrUp(image, dstsize=(image.shape[1] * scale_factor, image.shape[0] * scale_factor))
    elif up_or_down == "down":
        resized = cv2.pyrDown(image, dstsize=(image.shape[1] // scale_factor, image.shape[0] // scale_factor))
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    cv2.imwrite(f"resized_{up_or_down}.png", resized)
    return resized

if __name__ == "__main__":
    lambo = cv2.imread("lambo.png")
    sobel_result = sobel_edge_detection(lambo)
    canny_result = canny_edge_detection(lambo, 50, 50)

    shapes = cv2.imread("shapes.png")
    template = cv2.imread("shapes_template.jpg")
    template_result = template_match(shapes, template)

    resized_up = resize(lambo, 2, "up")
    resized_down = resize(lambo, 2, "down")