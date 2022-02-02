from scipy.stats import entropy
import numpy as np
import cv2

def macroBlocksMaker(row_size, column_size, array): #returns array with image divided on to macroblocks
    blocks=[]
    for r in range(0,array.shape[0] - row_size+1, row_size):
        for c in range(0,array.shape[1] - column_size+1, column_size):
            macroBlock = array[r:r+row_size,c:c+column_size].astype('int16')
            blocks.append(macroBlock)
    blocks=np.array(blocks)
    return blocks

def roundNumber(size, i):
    return ((i - 1) // size + 1) * size

def fillWithBlackLines(image):
    filled_image = []
    height = image.shape[0]
    width = image.shape[1]
    black_pixels = np.array(black_pixel() * (roundNumber(16, width) - width))
    for row in image:
        filled_row = np.append(row, black_pixels, axis = 0)
        filled_image.append(filled_row)
    black_line = np.array(black_pixel() * roundNumber(16, width))
    for j in range(roundNumber(16, height) - height):
        filled_image.append(black_line)
    return np.array(filled_image)

def black_pixel():
    return [0]

def level_analysis(source, target):
    source=np.array(source)
    target=np.array(target)
    global hierimg1
    global hierimg2
    for i in range(2):
        hierimg1 = hierimg1 + [hierarchicalDiv(source)]
        source =hierarchicalDiv(source)
        hierimg2 = hierimg2 + [hierarchicalDiv(target)]
        target =hierarchicalDiv(target)

def hierarchicalDiv(array):
    array=np.array(array)
    x, y = array.shape
    hierarchicalImage=[]
    for i in range(0, x, 2):
        for j in range(0, y, 2):
            try:
                hierarchicalImage.append(array[i][j])
            except:
                continue
    hierarchicalImage=np.array(hierarchicalImage)
    hierarchicalImage=np.reshape(hierarchicalImage, (int(x/2), int(y/2)))
    return(hierarchicalImage)

def improveDeeperLevels(movement_blocks,hierimg1,hierimg2):
    l = [8,16]
    for k in range(2):
        image1 = macroBlocksMaker(l[k],l[k],hierimg1[1-k])
        image2 = macroBlocksMaker(l[k],l[k],hierimg2[1-k])
        to_be_checked = []
        for i in range(len(movement_blocks)):#we will only check the blocks we saw movement in the previous hierarchical step
            if estimateMotion(image1[movement_blocks[i]]-image2[movement_blocks[i]]):
                continue #if it is still movement checks next block
            else:
                to_be_checked = to_be_checked + [i]#if there is no movement then save the index of the block that will later be poppes from the list
        movement_blocks = [x for x in movement_blocks if x not in to_be_checked]#pops unwanted blocks
    return(image1, image2 ,movement_blocks)

def estimateMotion(array):
    array=np.array(array)
    x, y = array.shape
    num_of_zeros = x*y - np.count_nonzero(array) #count the number of zeroes in the given image
    if (num_of_zeros >= 0.8 * x * y): #if the given array is at least 80% of zeroes then no motion is detected
        return(0)
    else:
        return(1)

def ImageReconstruction(x, y,image2):
    r=1
    for i in range(x):
        #initialisation of getting dimensions
        output = np.array(image2[i*(y)])
        for j in range(y-1):
            output = np.concatenate((output,image2[r]), axis=1)
            r= r +1
        r = r+1
        #initialisation of getting dimensions
        if(i==0):
            showim = output
        else:
            showim = np.concatenate((showim,output),axis=0)
    return(showim)#returns our reconstructed image

cap = cv2.VideoCapture('../auxiliary2021/original_movies/golf.avi')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = []
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = 30
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('../auxiliary2021/final_movies/exercise2/golf2.avi', fourcc , fps , size ,False)
fc = 0
ret, r_frame = cap.read()
r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)
global hierimg1
global hierimg2
while cap.isOpened():
    ret, c_frame = cap.read()
    if not ret:
        break
    if (c_frame.shape[0] % 16 != 0 or c_frame.shape[1] % 16 != 0):
            c_frame = fillWithBlackLines(c_frame)
    if (r_frame.shape[0] % 16 != 0 or r_frame.shape[1] % 16 != 0):
            r_frame = fillWithBlackLines(r_frame)
    c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
    hierimg1 = [r_frame]
    hierimg2 = [c_frame]
    level_analysis(r_frame,c_frame)
    image1 = macroBlocksMaker(4, 4, hierimg1[2])
    image2 = macroBlocksMaker(4, 4, hierimg2[2])
    movement_blocks = []
    for i in range(len(image1)):
        if estimateMotion(image1[i] - image2[i]):
            movement_blocks = movement_blocks + [i]  # if movement then append the index of the block
    image1, image2, movement_blocks = improveDeeperLevels(movement_blocks, hierimg1, hierimg2)
    image3 = image2
    for i in range(len(movement_blocks)):

        image2[movement_blocks[i]] = image1[movement_blocks[i]]




    y = int(c_frame.shape[1] / 16)
    x = int(c_frame.shape[0] / 16)


    image4 = ImageReconstruction(x,y,np.uint8(image2))

    print("Frame " + str(fc + 1) + " complete")
    try:
        image = image4

        buf.append(image.astype('int16'))  # check "continue" at line 16
        fc = fc + 1
    except:
        continue
    out.write(image4)


print('Video is read!')


cap.release()
out.release()
cv2.destroyAllWindows()