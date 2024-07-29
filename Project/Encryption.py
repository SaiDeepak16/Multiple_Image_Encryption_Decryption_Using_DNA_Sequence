from __future__ import annotations
import hashlib
import math
from PIL import Image

"""
The get_matrix function extract 2D matrix representation of image from 
a 3D numpy array
(The 3D array likely represents an RGB image with color planes)
"""


def get_matrix(im, plane):
    image = []
    width = im.shape[0]
    height = im.shape[1]
    for i in range(width):  # loops through image pixel grid
        row = []
        for j in range(height):
            try:
               # Here im.item(i,j,plane)
                # Extract pixel value from specified color plane i.e. 3D im array
                # i,j represent pixel position;
                # Plane specifies color channel
                row.append(im.item(i, j, plane))
                """
            appends the pixel value to current row
            Each row of matrix contains pixel value of im 
            Here it is converted from 3D to 2D
        """
            except:
                row = [im.item(i, j, plane)]
        try:
            image.append(row)
            # Appends row to image list creating 2D array here
        except:
            image = [row]
    return image


"""
The assign_matrix iterates over the pixels of img  for each pixel i,j 
And tries to assign a tuple of RGB values(in range of 256) to the image 
Form corresponding color matrices (mat,mat1,mat2)
"""


def assign_matrix_image(img, mat2, mat1, mat):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            try:
                img[j, i] = (int(mat[i][j]) % 256, int(mat1[i][j]) %
                             256, int(mat2[i][j]) % 256)
            except:
                img[j, i] = (int(mat[i][j]) % 256, int(mat1[i][j]) %
                             256, int(mat2[i][j]) % 256)
    return img


def toDecimal(binary):  # Converts the binary value to corresponding Decimal value
    decimal = 0
    for bit in binary:
        decimal = decimal * 2 + bit
    return decimal


def toBinary(n):  # COnverts n into corresponding Binary values
    counter = 8
    bitseq = []
    while (True):
        q = n/2
        r = n % 2
        n = q
        counter = counter-1
        try:
            bitseq.append(int(r))
        except:
            bitseq = [r]
        if (counter <= 0):
            break
    try:
        bitseq.reverse()
    except:
        bitseq = []
    return bitseq


"""
The renyi_seq generates sequence of numbers using Renyi map, 
Renyi map is nonlinear chaotic map commonly used in chaos-based cryptography
x:Initial value of seq
a,b: Coefficients that influence map
alpha: Scaling factor
n: number of iterations to generatge the sequence 
arr[] = List to store generated sequence
"""


def renyi_seq(x, a, b, alpha, n):
    arr = []
    xn = x
    for i in range(n):
        # Renyi map formula
        f = math.tan(x)*(1+x)*b + (1-x)*x*a
        # Scales fractional part by alpha; converts to integer
        xn, temp = math.modf(f)
        xn = int(xn * alpha)
        x = xn
        try:
            # Generated value f (fractional part) is appended to arr
            arr.append(f)
        except:
            arr = [f]
    return arr


"""
  The renyi_key_img iterate over each element in image matrix 
  Generate key Matrix Elements using Renyi sequence into key matrix (kmat)  
  kmat is likely used in subsequent encryption steps 
  rs:Renyi Sequence
  mat: Image Matrix
"""


def renyi_key_img(rs, mat):
    kmat = []
    x = 0
    for i in range(len(mat)):
        row = []
        for j in range(len(mat[i])):
            try:
                row.append(int(rs[x] % 256))
            except:
                row = [int(rs[x] % 256)]
            x = x + 1
        try:
            kmat.append(row)
        except:
            kmat = [row]
    return kmat


def tent_seq(x, r, n):
    xn = x
    arr = []
    for i in range(n):
        xn = min(x, 1-x) * r
        x = xn
        try:
            arr.append(xn)
        except:
            arr = [xn]
    return arr


Rule = [
    ["A", "G", "C", "T"],
    ["A", "C", "G", "T"],
    ["G", "A", "T", "C"],
    ["C", "A", "T", "G"],
    ["G", "T", "A", "C"],
    ["C", "T", "A", "G"],
    ["T", "G", "C", "A"],
    ["T", "C", "G", "A"]
]
Rule_reverse = [
    {"A": 0, "G": 1, "C": 2, "T": 3},
    {"A": 0, "C": 1, "G": 2, "T": 3},
    {"G": 0, "A": 1, "T": 2, "C": 3},
    {"C": 0, "A": 1, "T": 2, "G": 3},
    {"G": 0, "T": 1, "A": 2, "C": 3},
    {"C": 0, "T": 1, "A": 2, "G": 3},
    {"T": 0, "G": 1, "C": 2, "A": 3},
    {"T": 0, "C": 1, "G": 2, "A": 3}
]

"""
This dna_code_to_map will convert the dna sequece based on the specified 'rule'

For example
The dna_code_to_map function will convert the DNA sequence "AGCT" based on the specified rule

For the first character 'A' in the DNA sequence:
Rule[1][0] corresponds to 'A', so the numerical value 0 is appended to the num list.

For the second character 'G' in the DNA sequence:
Rule[1][1] corresponds to 'G', so the numerical value 1 is appended to the num list.

For the third character 'T' in the DNA sequence:
Rule[1][3] corresponds to 'T', so the numerical value 3 is appended to the num list.

For the fourth character 'C' in the DNA sequence:
Rule[1][2] corresponds to 'C', so the numerical value 2 is appended to the num list.

After processing all characters in the DNA sequence, the num list will be [0, 1, 3, 2].
"""


def dna_code_to_map(st, rule):
    num = []
    for i in range(4):
        for j in range(4):
            if (Rule[rule][j] == st[i]):
                try:
                    num.append(j)
                except:
                    num = [j]
    return num


"""
The map_to_dna_code is the opposite of above function

For exampla:
# For the first numerical value 0:

Rule[1][0] corresponds to 'A', so 'A' is added to the resulting DNA sequence.
For the second numerical value 1:

Rule[1][1] corresponds to 'G', so 'G' is added to the resulting DNA sequence.
For the third numerical value 3:

Rule[1][3] corresponds to 'T', so 'T' is added to the resulting DNA sequence.
For the fourth numerical value 2:

Rule[1][2] corresponds to 'C', so 'C' is added to the resulting DNA sequence.
After processing all numerical values, the resulting DNA sequence will be: 'AGTC'
"""


def map_to_dna_code(num, rule):
    st = ""
    for i in range(4):
        st = st + Rule[rule][num[i]]
    return st


def dna_oper_encode(s1, s2, oper, rule):
    n1 = dna_code_to_map(s1, rule)
    n2 = dna_code_to_map(s2, rule)
    n3 = []
    for i in range(4):
        if (oper == 0):
            a = (n1[i]+n2[i]) % 4
        elif (oper == 1):
            a = (n1[i]-n2[i]) % 4
        else:
            a = n1[i] ^ n2[i]
        try:
            n3.append(a)
        except:
            n3 = [a]
    s3 = map_to_dna_code(n3, rule)
    return s3


def dna_oper_decode(s1, s2, oper, rule):
    n1 = dna_code_to_map(s1, rule)
    n2 = dna_code_to_map(s2, rule)
    n3 = []
    for i in range(4):
        if (oper == 0):
            a = (n1[i]-n2[i]) % 4
        elif (oper == 1):
            a = (n1[i]+n2[i]) % 4
        else:
            a = n1[i] ^ n2[i]
        try:
            n3.append(a)
        except:
            n3 = [a]
    s3 = map_to_dna_code(n3, rule)
    return s3


"""
The int_to_dna_code funtion calls another function named toBinary to conver the integer l into it's binary representation  
The binary representation is stored in the variable 'a'
Iterates over the binary representation (a) in pairs of bits (2 bits at a time, starting from the least significant bits)
The fucntion checks the combination of bits and maps it to a corresponding DNA base
THis mapping is done using the Rule list

The resultion DNA bases are concatenated to the string st.(repeated for all pairs of bits in binary representation)

The func returns the final DNA sequence (st)

Parameters:
l - The integer value that needs to be converted into a DNA sequence
rule - The rule used to determins the mappin from 
binary to DNA bases Likely determined from tent sequence or another source
"""


def int_to_dna_code(l, rule):
    a = toBinary(l)
    st = ""
    for i in range(0, 8, 2):
        if (a[i] == 0 and a[i+1] == 0):
            st = st + Rule[rule][0]
        elif (a[i] == 0 and a[i+1] == 1):
            st = st + Rule[rule][1]
        elif (a[i] == 1 and a[i+1] == 0):
            st = st + Rule[rule][2]
        else:
            st = st + Rule[rule][3]
    return st


"""
This function does the opposite to the above function 

Parameters:
s - 
"""


def dna_code_to_int(s, rule):
    b = []
    for i in range(4):
        a = Rule_reverse[rule][s[i]]
        if (a == 0):
            try:
                b.append(0)
                b.append(0)
            except:
                b = [0]
                b = [0]
        elif (a == 1):
            try:
                b.append(0)
                b.append(1)
            except:
                b = [0]
                b = [1]
        elif (a == 2):
            try:
                b.append(1)
                b.append(0)
            except:
                b = [1]
                b = [0]
        else:
            try:
                b.append(1)
                b.append(1)
            except:
                b = [1]
                b = [1]
    a = toDecimal(b)
    return a


"""
The encode_dna_mat function iterate over each elemeent of the input matrix mat  
For each element in the matrix a rule is calaculated based on the tent sequence (tr) and 
the current position x.
The fuction the calls the int_to_dna_code function to encode the numerical value of the 
matrix element into a DNA
The encoded DNA sequence (st) is appended to the current row The row is then appended 
to the matrix emat.

"""


def encode_dna_mat(mat, tr):
    emat = []
    x = 0
    for i in range(len(mat)):
        row = []
        for j in range(len(mat[0])):
            rule = int(tr[x]*x) % 8
            st = int_to_dna_code(mat[i][j], rule)
            try:
                row.append(st)
            except:
                row = []
            x = x + 1
        try:
            emat.append(row)
        except:
            emat = [row]
    return emat


"""
The function decode_dna_mat is the opposite of the previouse function 
Here the matrix that is encoded is obtained and decoded using 
the function dna_code_to_int and stores the solution in emat matrix
"""


def decode_dna_mat(mat, tr):
    emat = []
    x = 0
    for i in range(len(mat)):
        row = []
        for j in range(len(mat[0])):
            rule = int(tr[x]*x) % 8
            n = dna_code_to_int(mat[i][j], rule)
            try:
                row.append(n)
            except:
                row = []
            x = x + 1
        try:
            emat.append(row)
        except:
            emat = [row]
    return emat


"""
mat: The first matrix containing DNA sequences 
mat1: The second matrix containing DNA sequences
ts: The parameter containing rules or parameters for the DNA operations

The function dna_opers_encode This function likely performs a DNA operation on the
corresponding elements of mat and mat1 based on the determined rule and operation.

 Loop over all rows and based on the values of ts and x calculate value which
is used to determina how the DNA operation to be performed  

oper: Calculated based on the values of ts and x It likely determins the type
of DNA operation to be performed
"""


def dna_opers_encode(mat, mat1, ts):
    omat = []
    x = 0
    for i in range(len(mat)):
        row = []
        rule = int(ts[x]*x) % 8
        oper = int(ts[x]*x) % 3
        for j in range(len(mat[i])):
            val = dna_oper_encode(mat[i][j], mat1[i][j], oper, rule)
            try:
                row.append(val)
            except:
                row = [val]
        x = x + 1
        try:
            omat.append(row)
        except:
            row = [omat]
    return omat


def dna_opers_decode(mat, mat1, ts):
    omat = []
    x = 0
    for i in range(len(mat)):
        row = []
        rule = int(ts[x]*x) % 8
        oper = int(ts[x]*x) % 3
        for j in range(len(mat[i])):
            val = dna_oper_decode(mat[i][j], mat1[i][j], oper, rule)
            try:
                row.append(val)
            except:
                row = [val]
        x = x + 1
        try:
            omat.append(row)
        except:
            row = [omat]
    return omat


"""
The funciton get_inital_values performs bitwise XOR operations between the current XOR value 
and the product o f the ASCII value of the character and its position in the chuck
Meaning the XOR operation introducres a level of randomness and non-linearity,
ensuring that different characters in different positions contribute diffenrently to the final result
This helps in creating a more diverse set of initial values

 _sum = _sum + ord(s[j]):
# Purpose: This step accumulates the ASCII values of the characters in the chunk.
The sum provides a measure of the total "weight" of hte characters in theis part of calculation
By summing up ASCII values characters with higher ASCII values contribute more to this part of the calculation
It adds another layer of complexity and diversity to the generated values

val = (xor + _sum) / 8192:
# Purpose: This step combines the XOR and sum values and scales the result by dividing by 8192.
The combination of XOR and sum values range providing a normaized and consistent output
The specific choice of 8192 might be arbitary and could be related to the desired range of values
"""


def get_initial_values(hash):
    strings = [(hash[i:i+8]) for i in range(0, len(hash), 8)]
    values = []
    for i in range(len(strings)):
        s = strings[i]
        xor = 0
        sum = 0
        for j in range(len(s)):
            xor = xor ^ (ord(s[j]) * j)
            sum = sum + ord(s[j])
        val = (xor + sum) / 8192
        try:
            values.append(val)
        except:
            values = [val]
    values.append((values[0] + values[4] + values[7])/3)
    # print(values)
    return values


"""
The encrypt_image function obtain a list of inital values, like based on hash

The generation of initial values from the hash is a step in the encryption process that
involves deriving numerical values from the hash of the encryption key.

This hash is a crucial step in creating unique and deterministic values based on the encryption key. 
These values likely contribute to the randomness or uniqueness of the encryption process. 

In image processing, images are often represented as 3D arrays, where the third dimension corresponds
to the color channels. Each element in the array represents the intensity of a pixel in a specific color channel
(e.g., red, green, or blue). Separating color channels can be useful for various purposes, 
including image manipulation, analysis, and encryption.

The values passed to the renyi_seq function are derived from the initial values obtained based 
on the hash of the encryption key (v), and they are used as parameters to generate a Rényi sequence.

The Rényi sequence is likely employed in the encryption process to introduce chaos and randomness.
The specific values and their combinations contribute to the uniqueness and variability of the generated sequence.

The renyi_key_img function is responsible for generating the key matrix (key_img) based on the Rényi sequence 
and image matrix. The specifics of how the Rényi sequence influences key generation are 
determined by the implementation of this function.

Purpose of Key Matrix:
The key matrix is likely used in subsequent encryption steps It may be combined withother matrices,
undergoe further transformation or contribute to the generation of pseudorandomvalues used in the encryption process

The encode_dna_mat function is responsibel for encoding the key matrix into a DNA sequence.
The result of this encoding process is stored in the variable encoded_key_image. 
This variable likely holds a representation of the key matrix in the form of a DNA sequence.

The ts (tent sequence) is used as part of the encoding process
The encoded_key_image variable is likely used in subsequent steps possibly in combination with other 
encoded data or as a component in the encryption process

rs: The Rényi sequence generated earlier using the renyi_seq function. 
This sequence is likely used as a source of randomness for key generation.
mat: The image matrix, which represents the pixel values of the image. 
This matrix might be processed to derive key-related information.
key_img: The key matrix generated using the Rényi sequence and the image matrix.
ts: The tent sequence, likely used for encoding.
encoded_img: The matrix of DNA sequences obtained by encoding the initial image matrix.
encoded_key_image: The matrix of DNA sequences obtained by encoding the key matrix.


hashlib.sha256(key.encode()): Creates a SHA-256 hash object from the UTF-8 encoded version of the encryption key.
hash.hexdigest(): Converts the hash object into its hexadecimal representation.

"""


def encrypt_image(im, target_file_name, key):
    print("Encryption started.....")
    hash = hashlib.sha256(key.encode())
    hash = hash.hexdigest()
    print(hash)
    v = get_initial_values(hash)
    print("hash values", v)
    mat = get_matrix(im, 0)
    rs = renyi_seq(v[0], (v[2]+v[4])*10, (v[5]+v[6])*10,
                   (v[1]+v[7])*12345, len(mat)*len(mat[0]))

# The function renyi_seq is being called with specific arguments.
# Let's break down why these particular values are passed:

# v[0]: This is the first value from the list v obtained from the get_initial_values function.
# The values in v are generated based on the hash of the encryption key.
# This initial value likely serves as the starting point for the Rényi sequence.

# (v[2]+v[4])*10: This is a combination of the third and fifth values in v, multiplied by 10.
# The specific multiplication factor and the choice of these particular values contribute
# to the coefficients influencing the Rényi map formula. These coefficients introduce a level of variability
# and complexity to the generated sequence.

# (v[5]+v[6])*10: Similar to the previous case, this is a combination of the sixth and seventh values in v,
# multiplied by 10. These values contribute to the coefficients in the Rényi map formula,
# introducing additional variability.

# (v[1]+v[7])*12345: This combination involves the second and eighth values in v, multiplied by 12345.
# Again, this introduces more variability and complexity to the coefficients used in the Rényi map formula.
# The choice of 12345 as a multiplier is arbitrary and is likely chosen for its prime factorization properties.

# len(mat)*len(mat[0]): The length of the matrix mat (presumably representing an image) is used as the n parameter
# for the Rényi sequence function. This determines how many iterations of the Rényi map will be computed,
# influencing the length of the generated sequence.

    ts_r = (v[6] + 1.4)

# ts_r is calculated by adding 1.4 to the seventh value (v[6]) from the initial values obtained earlier.
# This value is used as a parameter for the subsequent tent_seq function. The addition of 1.4 introduces a
# constant factor to the calculation.

    if (ts_r - 2 > 0):
        ts_r = ts_r - 2
# There is a conditional check: if ts_r - 2 > 0:. If the result of subtracting 2 from ts_r is greater than 0,
# then ts_r is adjusted by subtracting 2. This conditional check ensures that ts_r remains

    ts = tent_seq(v[3], ts_r, len(mat)*len(mat[0]))

# The adjusted ts_r and the fourth value (v[3]) from the initial values are used as parameters
# for the tent_seq function. The result, ts, is a sequence generated using the tent map.
# The length of the sequence is determined by the product of the width and height of the image m

# Here's an overview of what the tent map and its parameters are doing in this context:

# The tent map is a chaotic map that generates a sequence of values based on an initial value and a parameter.
# It is commonly used in chaos-based encryption algorithms.

# The parameters (v[6] + 1.4) and v[3] are used to introduce variability and randomness into the tent map.
# The seventh initial value (v[6]) influences the starting point, and
# the fourth initial value (v[3]) affects the behavior of the tent map.

# The adjustment of ts_r and the subsequent calculation of ts contribute to the creation of a dynamic sequence
# that is likely used in subsequent encryption steps.

# These steps showcase the use of different chaotic maps (Rényi map and tent map) and their parameters to generate sequences
# that add randomness and complexity to the encryption process. The resulting sequences (rs and ts) are then utilized
# in the encryption operations, which could involve further manipulation of matrices or image data.

    key_img = renyi_key_img(rs, mat)
    encoded_key_image = encode_dna_mat(key_img, ts)

# The ts (tent sequence) is used as part of the encoding process
# The encoded_key_image variable is likely used in subsequent steps possibly in combination with other
# encoded data or as a component in the encryption process

    encoded_img = encode_dna_mat(mat, ts)
    output_img = dna_opers_encode(encoded_img, encoded_key_image, ts)

# encoded_img: The matrix of DNA sequences obtained by encoding the initial image matrix.
# encoded_key_image: The matrix of DNA sequences obtained by encoding the key matrix.
# ts: The parameter that likely contains rules or parameters for the DNA operations. It is used in
# the dna_opers_encode function.

    decoded_output_img = decode_dna_mat(output_img, ts)
    mat = get_matrix(im, 1)
    rs = renyi_seq(v[1], (v[0]+v[6])*10, (v[2]+v[4])*10,
                   (v[3]+v[5])*12345, len(mat)*len(mat[0]))
    ts_r = (v[5] + 1.4)
    if (ts_r - 2 > 0):
        ts_r = ts_r - 2
    ts = tent_seq(v[7], ts_r, len(mat)*len(mat[0]))
    key_img = renyi_key_img(rs, mat)
    encoded_key_image = encode_dna_mat(key_img, ts)
    encoded_img = encode_dna_mat(mat, ts)
    output_img = dna_opers_encode(encoded_img, encoded_key_image, ts)
    decoded_output_img1 = decode_dna_mat(output_img, ts)
    mat = get_matrix(im, 2)
    rs = renyi_seq(v[2], (v[3]+v[5])*10, (v[1]+v[7])*10,
                   (v[0]+v[4])*12345, len(mat)*len(mat[0]))
    ts_r = (v[4] + 1.4)
    if (ts_r - 2 > 0):
        ts_r = ts_r - 2
    ts = tent_seq(v[8], ts_r, len(mat)*len(mat[0]))
    key_img = renyi_key_img(rs, mat)
    encoded_key_image = encode_dna_mat(key_img, ts)
    encoded_img = encode_dna_mat(mat, ts)
    output_img = dna_opers_encode(encoded_img, encoded_key_image, ts)
    decoded_output_img2 = decode_dna_mat(output_img, ts)
    img = Image.new("RGB", (im.shape[1], im.shape[0]))

# The line img = Image.new("RGB", (im.shape[1], im.shape[0])) creates a new RGB (Red, Green, Blue) image
# using the Python Imaging Library (PIL) or its successor, the Pillow library.

# The Image.new function is a part of the PIL or Pillow library, and it is used to create a new image
# with a specified mode and size.

# "RGB": This specifies the mode of the image. In this case, "RGB" indicates that the image will have
# three color channels: Red, Green, and Blue. Each channel can have values ranging from 0 to 255,
# representing different intensities of the respective colors.
# (im.shape[1], im.shape[0]): This specifies the size of the image in pixels. width and height respectively
# So, the line creates a new RGB image with dimensions matching the width and height of the original image (im).

    pix = img.load()

#   The line pix = img.load() is used to obtain a pixel access object for the image. In Pillow
#   (a fork of the Python Imaging Library or PIL), the load() method is used to create a pixel access
#   object that allows direct access to the pixel values of the image

# img: This is the image object that you have created using Image.new or loaded from a file.

# .load(): This method returns a pixel access object that you can use to read or modify individual pixel values.

    pix = assign_matrix_image(pix, decoded_output_img,
                              decoded_output_img1, decoded_output_img2)
# here we assign the manipulated matrix into the pix as pixels creating the image
    img.save(target_file_name, "BMP")


def decrypt_image(im, target_file_name, key):
    print("Decryption started.....")
    hash = hashlib.sha256(key.encode()).hexdigest()
    print(hash)
    v = get_initial_values(hash)
    print(v)
    mat = get_matrix(im, 0)
    rs = renyi_seq(v[0], (v[2]+v[4])*10, (v[5]+v[6])*10,
                   (v[1]+v[7])*12345, len(mat)*len(mat[0]))
    ts_r = (v[6] + 1.4)
    if ts_r - 2 > 0:
        ts_r = ts_r - 2
    ts = tent_seq(v[3], ts_r, len(mat)*len(mat[0]))
    key_img = renyi_key_img(rs, mat)
    encode_img = encode_dna_mat(mat, ts)
    encoded_key_image = encode_dna_mat(key_img, ts)
    output_img = dna_opers_decode(encode_img, encoded_key_image, ts)
    decoded_output_img = decode_dna_mat(output_img, ts)
    mat = get_matrix(im, 1)
    rs = renyi_seq(v[1], (v[0]+v[6])*10, (v[2]+v[4])*10,
                   (v[3]+v[5])*12345, len(mat)*len(mat[0]))
    ts_r = (v[5] + 1.4)
    if ts_r - 2 > 0:
        ts_r = ts_r - 2
    ts = tent_seq(v[7], ts_r, len(mat)*len(mat[0]))
    key_img = renyi_key_img(rs, mat)
    encode_img = encode_dna_mat(mat, ts)
    encoded_key_image = encode_dna_mat(key_img, ts)
    output_img = dna_opers_decode(encode_img, encoded_key_image, ts)
    decoded_output_img1 = decode_dna_mat(output_img, ts)
    mat = get_matrix(im, 2)
    rs = renyi_seq(v[2], (v[3]+v[5])*10, (v[1]+v[7])*10,
                   (v[0]+v[4])*12345, len(mat)*len(mat[0]))
    ts_r = (v[4] + 1.4)
    if ts_r - 2 > 0:
        ts_r = ts_r - 2
    ts = tent_seq(v[8], ts_r, len(mat)*len(mat[0]))
    key_img = renyi_key_img(rs, mat)
    encode_img = encode_dna_mat(mat, ts)
    encoded_key_image = encode_dna_mat(key_img, ts)
    output_img = dna_opers_decode(encode_img, encoded_key_image, ts)
    decoded_output_img2 = decode_dna_mat(output_img, ts)
    img = Image.new("RGB", (im.shape[1], im.shape[0]))
    pix = img.load()
    pix = assign_matrix_image(pix, decoded_output_img,
                              decoded_output_img1, decoded_output_img2)
    img.save(target_file_name, "BMP")
