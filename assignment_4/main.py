import cv2
import numpy as np



def harris(reference_image):
    image = cv2.imread(reference_image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow("harris", image)
    cv2.imwrite("harris.jpg", image)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()

def align_sift(img_path, ref_path, max_features=10, good_match_precent=0.7):
    img, ref = cv2.imread(img_path), cv2.imread(ref_path)
    g1, g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)

    if d1 is None or d2 is None: return

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    good = [m for m,n in flann.knnMatch(d1, d2, k=2) if m.distance < good_match_precent*n.distance]

    if len(good) >= max_features:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if M is not None:
            cv2.imwrite("aligned.jpg", cv2.warpPerspective(img, M, (ref.shape[1], ref.shape[0])))
            cv2.imwrite("matches.jpg", cv2.drawMatches(img,kp1,ref,kp2,good,None,
                            matchesMask=mask.ravel().tolist(), flags=2))
            return
    cv2.imwrite("matches.jpg", cv2.drawMatches(img,kp1,ref,kp2,good,None,flags=2))

def main():
    harris("reference_img.png")
    align_sift("align_this.jpg", "reference_img.png", 10, 0.7)
if __name__ == '__main__':
    main()