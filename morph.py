import cv2
from util import read_and_scale
from PIL import Image
import numpy as np
from util import get_face_analyser
from insightface.utils import face_align
import uuid
import itertools
from new_faces import faces_from_image_gen
from util import preprocess_image, process_image, create_new_directory
import os

def calculate_delaunay_triangles(rect, points):
    """Calculate the Delaunay triangulation for a set of points"""
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))

    triangleList = subdiv.getTriangleList()
    delaunayTri = []

    # Use a smaller threshold for floating-point comparison
    threshold = 0.1

    for t in triangleList:
        ind = []
        for i in [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]:
            for j, p in enumerate(points):
                if abs(i[0] - p[0]) < threshold and abs(i[1] - p[1]) < threshold:
                    ind.append(j)
                    break
        if len(ind) == 3:
            delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def rect_contains(rect, point):
    """Check if a point is inside a rectangle"""
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def apply_affine_transform(src, srcTri, dstTri, size):
    """Apply affine transform calculated using srcTri and dstTri."""
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """Warps and alpha blends triangular regions."""
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask

    #debug_img = np.uint8(img.copy())
    #Image.fromarray(cv2.cvtColor(debug_img,cv2.COLOR_RGBA2BGR)).save('/app/faces/debug_morph.jpg')

# Main function
def morph_faces(img1, img2, points1, points2, alpha=0.5):
    """ Morphs two faces given two sets of facial landmarks. """
    
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Allocate space for final output
    imgMorph = img1.copy()

    # Rectangle to be used with Subdiv2D
    size = img1.shape
    rect = (0, 0, size[1], size[0])

    # Calculate Delaunay triangles
    dt = calculate_delaunay_triangles(rect, points1)

    #print(dt)

    # Morph all triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        t = []

        # Get points for img1, img2 and imgMorph
        for j in range(0, 3):
            t1.append(points1[dt[i][j]])
            t2.append(points2[dt[i][j]])
            t.append((points1[dt[i][j]][0] * (1 - alpha) + points2[dt[i][j]][0] * alpha,
                      points1[dt[i][j]][1] * (1 - alpha) + points2[dt[i][j]][1] * alpha))
        
        # Morph one triangle at a time.
        morph_triangle(img1, img2, imgMorph, t1, t2, t, alpha)

    # Display Result
    face_analyser = get_face_analyser()
    img = np.uint8(imgMorph)
    face = face_analyser.get(img)[0]
    return (img, face)

def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

def create_sliding_window_groups(collection, window_size):
    return [collection[i:i + window_size] for i in range(len(collection) - window_size + 1)]

def process_group(group):
    a, b = group
    return morph_faces(a[0], b[0], a[1].landmark_2d_106, b[1].landmark_2d_106)

def reduce_recursively(collection, window_size):
    while len(collection) > 1:
        groups = create_sliding_window_groups(collection, window_size)
        collection = [process_group(group) for group in groups]

    return collection[0]

def morph_images(loops=1, randomize=True, faces_directory='/app/faces/orig', output_path='/app/faces/new/', scale=False):
    
    output_path = create_new_directory(output_path)
    face_analyser = get_face_analyser()

    for _ in range(0, loops):
        run_id = str(uuid.uuid4())
        
        images=None

        if randomize:
            images = faces_from_image_gen(["a beautiful 35yo woman"])
        else:
            files = os.listdir(faces_directory)
            image_files = [os.path.join(faces_directory, file) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            images = []
            for f in image_files:
                if scale:
                    _, img = read_and_scale(f)
                else:
                    img = cv2.imread(f)
                for x in process_image(Image.fromarray(img)):
                    images.append(preprocess_image(x))

        aligned = []

        for i, image in enumerate(images):
            #face = face_analyser.get(image)[0]
            face = image["first_face"]
            M = face_align.estimate_norm(face.kps, 256)
            M[0, 2] += 50
            M[1, 2] += 50
            aligned_img = cv2.warpAffine(np.array(image["data"]), M, (256 + 100, 256 + 100), borderValue=0.0) 
            #Image.fromarray(cv2.cvtColor(aligned_img,cv2.COLOR_RGBA2BGR)).save(f'{output_path}/{run_id}_{i}-aligned.png')

            face_aligned = face_analyser.get(aligned_img)[0]

            aligned.append((aligned_img, face_aligned))

        #points1 = aligned[0][1].landmark_2d_106
        #img = aligned[0][0].copy()
        #rect = (0, 0, img.shape[1], img.shape[0])
        #subdiv = cv2.Subdiv2D(rect)
        #for p in points1:
        #    subdiv.insert((p[0], p[1]))
        #draw_delaunay(img, subdiv, (255, 255, 255))
        #Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)).save('{output_path}/{run_id}_debug.jpg')

        array = aligned
        pairs = list(itertools.permutations(array, 2))
        print(f'Pairs: {len(pairs)}')
        new_array = []
        for i, (a, b) in enumerate(pairs):
            x, y = morph_faces(a[0], b[0], a[1].landmark_2d_106, b[1].landmark_2d_106)
            new_array.append((x, y))
        
        result = reduce_recursively(new_array, 2)
        Image.fromarray(cv2.cvtColor(result[0],cv2.COLOR_RGBA2BGR)).save(f'{output_path}/morph_{run_id}.jpg')        
            
        ####
        #### OLD WAY
        ####
        ####aligned_with_reversed = [aligned, aligned[::-1]]        
        ####for ordering, aligned in enumerate(aligned_with_reversed):
        ####    average = []
        ####    averaged_image = aligned[0][0]
        ####    for i, current in enumerate(aligned[:-1]):
        ####        next = aligned[i+1]
        ####        if len(average) > 0:
        ####            average = (average + next[1].landmark_2d_106) / 2
        ####        else:
        ####            average = (current[1].landmark_2d_106 + next[1].landmark_2d_106) / 2
        ####        averaged_image = morph_faces(averaged_image, next[0], current[1].landmark_2d_106, average)
        ####        Image.fromarray(cv2.cvtColor(averaged_image,cv2.COLOR_RGBA2BGR)).save(f'{output_path}/morph_{run_id}_{ordering}_{i}.jpg')
        ####
        #### OLD WAY
        ####

def create_batches(files, batch_size):
    batches = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        batches.append(batch)
    return batches