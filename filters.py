import cv2
import face_recognition
import numpy as np
from matplotlib import pyplot as plt


class Filters:

    @staticmethod
    def draw_landmarks(img, type='points', thickness=5, color=(255, 0, 0)):
        # Load the image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find the face landmarks using face_recognition.face_landmarks1111
        face_landmarks_list = face_recognition.face_landmarks(image)

        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

        # Draw the landmarks on the image
        if type == 'points':
            for face_landmarks in face_landmarks_list:
                for name, list_of_points in face_landmarks.items():
                    for point in list_of_points:
                        cv2.circle(image, point, thickness, color, -1)

        elif type == 'lines':
            for face_landmarks in face_landmarks_list:

                # Print the location of each facial feature in this image
                for facial_feature in face_landmarks.keys():
                    print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    cv2.line(image, face_landmarks[facial_feature][0], face_landmarks[facial_feature][-1], color, thickness)

        # Convert the image back to BGR color (which OpenCV uses)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image


    @staticmethod
    def draw_rectangle(img):
        # Draw a rectangle around the detected face
        locs = face_recognition.face_locations(img)

        for (top, right, bottom, left) in locs:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        return img


    @staticmethod
    def resize_img(img, width, height, interpolation=cv2.INTER_LINEAR):
        return cv2.resize(img, (width, height), interpolation=interpolation)


    @staticmethod
    def translate_img(img, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


    @staticmethod
    def rotate_img(img, angle=0, center=None, scale=1.0):
        (h, w) = img.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        dst = cv2.warpAffine(img, M, (w, h))

        return dst


    @staticmethod
    def affine_transform_img(img, src_points=None, dst_points=None):
        if src_points is None:
            src_points = np.float32([[50, 50], [200, 50], [50, 200]])

        if dst_points is None:
            dst_points = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(src_points, dst_points)
        dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return dst


    @staticmethod
    def perspective_transform_img(img, src_points=None, dst_points=None):
        if src_points is None:
            src_points = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

        if dst_points is None:
            dst_points = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        return dst


    @staticmethod
    def image_inpainting(img, mask_img, method=cv2.INPAINT_TELEA):
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        return cv2.inpaint(img, mask_img, 3, method)


    @staticmethod
    def fastNlMeansDenoisingColored(img, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21):
        # noice is expected to be gaussian noise
        return cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)


    @staticmethod
    def gaussian_pyramids(image, levels=4, max_display_width=300):
        pyramid = [image]

        for i in range(levels - 1):
            # Apply Gaussian smoothing
            blurred = cv2.GaussianBlur(pyramid[i], (5, 5), 0)

            # Downsample the blurred image
            downsampled = cv2.pyrDown(blurred)

            # Add the downsampled image to the pyramid
            pyramid.append(downsampled)


        return Filters.create_hierarchy(pyramid, max_display_width)


    @staticmethod
    def create_hierarchy(pyramid, max_width):
        total_height = sum(img.shape[0] for img in pyramid)

        scale_factor = max_width / pyramid[0].shape[1]  # Calculate the scaling factor
        resized_height = int(total_height * scale_factor)  # Calculate the new height

        hierarchy_img = np.ones((resized_height, max_width, 3), dtype=np.uint8) * 255
        y_offset = 0

        for img in pyramid:
            height, width = img.shape[:2]
            resized_width = int(width * scale_factor)  # Calculate the new width

            # Resize the pyramid image
            resized_img = cv2.resize(img, (resized_width, int(height * scale_factor)))

            hierarchy_img[y_offset:y_offset + resized_img.shape[0], :resized_img.shape[1]] = resized_img
            y_offset += resized_img.shape[0]

        return hierarchy_img


    # Interactive Foreground Extraction using GrabCut Algorithm
    @staticmethod
    def grab_cut(img, rect=None, mask=None, iter_count=5):
        if rect is None:
            rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        return img


    @staticmethod
    def grab_cut2(img, rect=None, mask=None, iter_count=5):
        if rect is None:
            rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Create a temporary mask for sure background and sure foreground regions
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # Convert the RGB image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Update mask values based on pixel intensities
        mask[np.logical_or((img_gray == 0), (img_gray == 255))] = cv2.GC_PR_BGD
        mask[np.logical_and((img_gray != 0), (img_gray != 255))] = cv2.GC_PR_FGD

        # Run GrabCut algorithm
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)

        # Create a binary mask where sure background and likely background are set to 0, and the rest to 1
        mask2 = np.where(np.logical_or(mask == cv2.GC_BGD, mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')

        # Apply the mask to the original image
        img_with_mask = img * mask2[:, :, np.newaxis]

        return img_with_mask


    @staticmethod
    def find_histogram(clt):
        num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=num_labels)

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist


    @staticmethod
    def plot_histogram(img):
        color = ('b', 'g', 'r')

        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])

        plt.show()


    @staticmethod
    def generate_histogram_image(img):
        color = ('b', 'g', 'r')
        hist_height = 100
        hist_width = 256
        bin_width = int(hist_width / 256)

        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
            hist = np.int32(np.around(hist))

            for x, h in enumerate(hist):
                cv2.rectangle(hist_image, (x * bin_width, hist_height - 1),
                              ((x + 1) * bin_width - 1, hist_height - h.item()), (0, 0, 255), -1)

        return hist_image


    @staticmethod
    def apply_LBP(img, num_points=8, radius=1):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = np.zeros_like(gray, dtype=np.uint8)

        for y in range(radius, gray.shape[0] - radius):
            for x in range(radius, gray.shape[1] - radius):
                center_pixel = gray[y, x]
                binary_code = 0

                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    x_i = int(x + radius * np.cos(angle))
                    y_i = int(y - radius * np.sin(angle))

                    if gray[y_i, x_i] >= center_pixel:
                        binary_code |= 1 << (num_points - 1 - i)

                lbp[y, x] = binary_code

        return lbp


    @staticmethod
    def histogram_equalization(img):
        # Split the image into channels
        b, g, r = cv2.split(img)

        # Apply histogram equalization to each channel
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        # Merge the equalized channels back into a BGR image
        equalized_img = cv2.merge((b_eq, g_eq, r_eq))

        return equalized_img


    @staticmethod
    def adaptive_histogram_equalization(img):
        # Convert the image to LAB color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split the LAB image into channels
        l, a, b = cv2.split(lab_img)

        # Apply Adaptive Histogram Equalization to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        # Merge the equalized L channel with the original A and B channels
        lab_eq_img = cv2.merge((l_eq, a, b))

        # Convert the LAB equalized image back to BGR color space
        equalized_img = cv2.cvtColor(lab_eq_img, cv2.COLOR_LAB2BGR)

        return equalized_img


    @staticmethod
    def thresholding(img, threshold=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        ret, thresh = cv2.threshold(gray_img, threshold, max_value, threshold_type)

        return thresh


    @staticmethod
    def dilation(img, kernel_size=3, iterations=1):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply dilation
        dilated_img = cv2.dilate(img, kernel, iterations=iterations)

        return dilated_img


    @staticmethod
    def erosion(img, kernel_size=3, iterations=1):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply erosion
        eroded_img = cv2.erode(img, kernel, iterations=iterations)

        return eroded_img


    @staticmethod
    def morphological_opening(img, kernel_size=5):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply opening
        opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        return opened_img


    @staticmethod
    def morphological_closing(img, kernel_size=5):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply closing
        closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        return closed_img


    @staticmethod
    def morphological_gradient(img, kernel_size=5):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply gradient
        gradient_img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

        return gradient_img


    @staticmethod
    def morphological_tophat(img, kernel_size=5):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply top hat
        tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

        return tophat_img


    @staticmethod
    def morphological_blackhat(img, kernel_size=5):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply black hat
        blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

        return blackhat_img


    @staticmethod
    def smoothing_filter(img, kernel_size=5):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)

        # Apply smoothing filter
        smoothed_img = cv2.filter2D(img, -1, kernel)

        return smoothed_img


    @staticmethod
    def averaging_filter(img, kernel_size=5):
        # Apply averaging filter
        averaged_img = cv2.blur(img, (kernel_size, kernel_size))

        return averaged_img


    @staticmethod
    def gaussian_filter(img, kernel_size=5):
        # Apply Gaussian filter
        gaussian_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        return gaussian_img


    @staticmethod
    def bilateral_filter(img, kernel_size=9, sigma_color=75, sigma_space=75):
        # Apply bilateral filter
        bilateral_img = cv2.bilateralFilter(img, kernel_size, sigma_color, sigma_space)

        return bilateral_img


    @staticmethod
    def median_filter(img, kernel_size=5):
        # Apply median filter
        median_img = cv2.medianBlur(img, kernel_size)

        return median_img


    @staticmethod
    def max_filter(img, kernel_size=5, iterations=1):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply max filter
        max_img = cv2.dilate(img, kernel, iterations=iterations)

        return max_img


    @staticmethod
    def min_filter(img, kernel_size=5, iterations=1):
        # Create the kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply min filter
        min_img = cv2.erode(img, kernel, iterations=iterations)

        return min_img


    @staticmethod
    def gabor_filter(img, kernel_size=5, sigma=5, theta=10, lambd=10, gamma=1, psi=1):
        # Create the kernel
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, cv2.CV_32F)

        # Apply the filter
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)

        return filtered_img


    @staticmethod
    def high_pass_filter(img):
        # Create the kernel
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernel)

        return filtered_img


    @staticmethod
    def sharpening_filter(img):
        # Create the kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernel)

        return filtered_img


    @staticmethod
    def unsharp_masking_filter(img):
        # Create the kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernel)

        return filtered_img


    @staticmethod
    def laplacian_filter(img):
        # Create the kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernel)

        return filtered_img


    @staticmethod
    def sobel_filter(img):
        # kernels
        kernelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=int)
        kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernelx)
        filtered_img = cv2.filter2D(filtered_img, -1, kernely)

        return filtered_img


    @staticmethod
    def roberts_filter(img):
        # kernels
        kernelx = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=int)
        kernely = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernelx)
        filtered_img = cv2.filter2D(filtered_img, -1, kernely)

        return filtered_img


    @staticmethod
    def prewitt_filter(img):
        # kernels
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernelx)
        filtered_img = cv2.filter2D(filtered_img, -1, kernely)

        return filtered_img


    @staticmethod
    def kirsch_filter(img):
        # kernels
        kernelx = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int)
        kernely = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=int)
        kernel45 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=int)
        kernel135 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=int)
        kernel90 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=int)
        kernel225 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=int)
        kernel180 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=int)
        kernel315 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernelx)
        filtered_img = cv2.filter2D(filtered_img, -1, kernely)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel45)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel135)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel90)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel225)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel180)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel315)

        return filtered_img


    @staticmethod
    def canny_filter(img, threshold_1=100, threshold_2=200):
        # Apply the filter
        filtered_img = cv2.Canny(img, threshold_1, threshold_2)

        return filtered_img


    @staticmethod
    def scharr_filter(img):
        # Apply the Scharr filter
        gradient_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        gradient_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)

        # Compute the gradient magnitude
        gradient_mag = cv2.magnitude(gradient_x, gradient_y)

        # Convert the gradient magnitude to uint8 for visualization
        filtered_img = cv2.convertScaleAbs(gradient_mag)

        return filtered_img


    @staticmethod
    def robinson_filter(img):
        # kernels
        kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
        kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernelx)
        filtered_img = cv2.filter2D(filtered_img, -1, kernely)

        return filtered_img


    @staticmethod
    def nevatia_babu_filter(img):
        # kernels
        kernelx = np.array([[100, 100, 100, 100, 100],[100, 100, 100, 100, 100],
                            [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100],
                            [-100, -100, -100, -100, -100]], dtype=int)

        kernely = np.array([[-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100],
                            [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100],
                            [-100, -100, 0, 100, 100]], dtype=int)

        kernel45 = np.array([[-100, 0, 100, 100, 100], [-100, 0, 100, 100, 100],
                            [-100, 0, 0, 100, 100], [-100, -100, -100, 0, 100],
                            [-100, -100, -100, -100, -100]], dtype=int)

        kernel135 = np.array([[-100, -100, -100, -100, -100], [-100, -100, -100, 0, 100],
                              [-100, -100, 0, 100, 100], [-100, 0, 100, 100, 100],
                              [100, 100, 100, 100, 100]], dtype=int)

        kernel90 = np.array([[-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100],
                             [0, 0, 0, 0, 0], [100, 100, 100, 100, 100],
                             [100, 100, 100, 100, 100]], dtype=int)

        kernel225 = np.array([[100, 100, 100, 100, 100], [100, 100, 100, 0, -100],
                              [100, 100, 0, -100, -100], [100, 0, -100, -100, -100],
                              [-100, -100, -100, -100, -100]], dtype=int)

        kernel180 = np.array([[100, 100, 100, 100, 100], [100, 100, 100, 100, 100],
                              [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100],
                              [-100, -100, -100, -100, -100]], dtype=int)

        kernel315 = np.array([[-100, -100, -100, -100, -100], [100, 0, -100, -100, -100],
                              [100, 100, 0, -100, -100], [100, 100, 100, 0, -100],
                              [100, 100, 100, 100, -100]], dtype=int)

        # Apply the filter
        filtered_img = cv2.filter2D(img, -1, kernelx)
        filtered_img = cv2.filter2D(filtered_img, -1, kernely)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel45)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel135)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel90)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel225)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel180)
        filtered_img = cv2.filter2D(filtered_img, -1, kernel315)

        return filtered_img


    @staticmethod
    def fuzzy_filter(img):
        # Apply the filter
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)
        filtered_img = cv2.filter2D(img, -1, kernel)

        return filtered_img


    @staticmethod
    def embossing_filter(img):
        # Apply the filter
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=int)
        filtered_img = cv2.filter2D(img, -1, kernel)

        return filtered_img


    @staticmethod
    def harris_corner_filter(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform Harris corner detection
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # Dilate the corner points to enhance visibility
        dst = cv2.dilate(dst, None)

        # Mark the corners on the original image
        img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners as red

        return img


    @staticmethod
    def shi_tomasi_filter(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform Shi-Tomasi corner detection
        gray = np.float32(gray)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)

        # Mark the corners on the original image
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        return img


    @staticmethod
    def SIFT_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        # Detect key points
        kp, des = sift.detectAndCompute(gray, None)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)

        return img


    @staticmethod
    def SURF_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create SURF object
        surf = cv2.xfeatures2d.SURF_create()

        # Detect key points
        kp, des = surf.detectAndCompute(gray, None)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)

        return img


    @staticmethod
    def ORB_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create ORB object
        orb = cv2.ORB_create()

        # Detect key points
        kp = orb.detect(gray, None)

        # Compute the descriptors
        kp, des = orb.compute(gray, kp)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)

        return img


    @staticmethod
    def FAST_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create FAST object
        fast = cv2.FastFeatureDetector_create()

        # Detect key points
        kp = fast.detect(gray, None)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)

        return img


    @staticmethod
    def BRIEF_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create star detector object
        star = cv2.xfeatures2d.StarDetector_create()

        # Create BRIEF extractor object
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # Detect key points
        kp = star.detect(gray, None)

        # Compute the descriptors
        kp, des = brief.compute(gray, kp)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)


        return img


    @staticmethod
    def BRISK_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create BRISK object
        brisk = cv2.BRISK_create()

        # Detect key points
        kp = brisk.detect(gray, None)

        # Compute the descriptors
        kp, des = brisk.compute(gray, kp)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)


        return img


    @staticmethod
    def AKAZE_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create AKAZE object
        akaze = cv2.AKAZE_create()

        # Detect key points
        kp, des = akaze.detectAndCompute(gray, None)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)

        return img


    @staticmethod
    def KAZE_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create KAZE object
        kaze = cv2.KAZE_create()

        # Detect key points
        kp, des = kaze.detectAndCompute(gray, None)

        # Draw key points
        img = cv2.drawKeypoints(img, kp, img)

        return img


    @staticmethod
    def GFTT_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform GFTT corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

        # Draw the detected corners on the image
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        return img


    @staticmethod
    def MSER_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create MSER object
        mser = cv2.MSER_create()

        # Detect key points
        regions, bboxes = mser.detectRegions(gray)

        # Draw key points
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(img, hulls, 1, (0, 255, 0))

        return img


    @staticmethod
    def distance_transform_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Perform distance transform
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        # Normalize the distance image for range [0, 1] so we can visualize and threshold it
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

        return dist_transform


    @staticmethod
    def hough_lines_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform Canny edge detection
        gray = cv2.Canny(gray, 50, 200)

        # Perform Hough lines probabilistic transform
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        # Draw lines on the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img


    @staticmethod
    def hough_lines_filter2(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform Canny edge detection
        gray = cv2.Canny(gray, 50, 200)

        # Perform Hough lines transform
        lines = cv2.HoughLines(gray, 1, np.pi / 180, 200)

        # Draw lines on the image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))

            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img


    @staticmethod
    def hough_circles_filter(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform Canny edge detection
        gray = cv2.Canny(gray, 50, 200)

        # Perform Hough circles transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

        # Draw detected circles on the image
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Draw outer circle
                cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                # Draw center of circle
                cv2.circle(img, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        return img

    @staticmethod
    def apply_filter(filter_index, image):
        filter_mapping = {
            0: Filters.fastNlMeansDenoisingColored,
            1: Filters.smoothing_filter,
            2: Filters.averaging_filter,
            3: Filters.median_filter,
            4: Filters.apply_LBP,
            5: Filters.bilateral_filter,
            6: Filters.histogram_equalization,
            7: Filters.adaptive_histogram_equalization,
            8: Filters.thresholding,
            9: Filters.grab_cut,
            10: Filters.gaussian_filter,
            11: Filters.dilation,
            12: Filters.erosion,
            13: Filters.morphological_opening,
            14: Filters.morphological_closing,
            15: Filters.morphological_gradient,
            16: Filters.morphological_tophat,
            17: Filters.morphological_blackhat,
            18: Filters.max_filter,
            19: Filters.min_filter,
            20: Filters.gabor_filter,
            21: Filters.high_pass_filter,
            22: Filters.laplacian_filter,
            23: Filters.sharpening_filter,
            24: Filters.unsharp_masking_filter,
            25: Filters.scharr_filter,
            26: Filters.robinson_filter,
            27: Filters.embossing_filter,
            28: Filters.sobel_filter,
            29: Filters.roberts_filter,
            30: Filters.prewitt_filter,
            31: Filters.kirsch_filter,
            32: Filters.canny_filter,
            33: Filters.nevatia_babu_filter,
            34: Filters.fuzzy_filter,
            35: Filters.harris_corner_filter,
            36: Filters.shi_tomasi_filter,
            37: Filters.SIFT_filter,
            38: Filters.SURF_filter,
            39: Filters.ORB_filter,
            40: Filters.FAST_filter,
            41: Filters.BRIEF_filter,
            42: Filters.BRISK_filter,
            43: Filters.GFTT_filter,
            44: Filters.MSER_filter,
            45: Filters.AKAZE_filter,
            46: Filters.KAZE_filter,
            47: Filters.distance_transform_filter,
            48: Filters.hough_lines_filter,
            49: Filters.hough_circles_filter
        }

        # Check if the selected index exists in the mapping
        if filter_index in filter_mapping:
            # Retrieve the filter method based on the index
            filter_method = filter_mapping[filter_index]

            # Apply the filter method to the image
            filtered_image = filter_method(image)
            return filtered_image
        else:
            # Handle the case when an invalid index is selected
            print("Invalid filter index!")
            return None
