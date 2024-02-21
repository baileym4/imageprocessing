"""
Image Processing
"""

#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
import math
from PIL import Image
import random


def get_pixel(image, row, col, edge=None):
    """
    Gets the pixel value given a specific row and column in an image

    Args:
        image (dict): contains keys width and height that store values for
        width and heightand key pixels which is a list of numbers to
        represent a greyscale image
        row (int): int that describes row positon in picture grid
        col (int): int that describes col position in picture grid
        edge (None, optional): describes what type of wrapping the picture
        will be doing. Defaults to None.

    Returns:
        The pixel value at the specified row, col (float or int)
    """
    # define index in list based on given row and col
    index = (image["width"] * row) + col

    if edge == "zero":
        if row < 0 or col >= image["width"] or row >= image["height"] or col < 0:
            return 0  # return zero if outside

    if edge == "wrap":
        if row < 0 or row >= image["height"]:
            row = row % image["height"]
        if col < 0 or col >= image["width"]:
            col = col % image["width"]
        index = (image["width"] * row) + col  # reassign index

    if edge == "extend":
        if col < 0:
            col = 0
        if col >= image["width"]:
            col = image["width"] - 1
        if row < 0:
            row = 0
        if row >= image["height"]:
            row = image["height"] - 1

        index = (image["width"] * row) + col  # reassign index

    return image["pixels"][index]  # return pixel at correct index


def set_pixel(image, row, col, color):
    """
    Sets a color value of a pixel at a specifc row and col of an image

    Args:
        image (dict): contains keys width and height that store values for
        width and heightand key pixels which is a list of numbers to
        represent a greyscale image
        row (int): int that describes row positon in picture grid
        col (int): int that describes col position in picture grid
        color (int or float): color value pixel should be assigned
    """
    index = (image["width"] * row) + col

    image["pixels"][index] = color  # reset color


def apply_per_pixel(image, func):
    """_summary_
    Applies a given function to every pixel of an image and
    returns the new image created

    Args:
        image (dict): contains keys width and height that store values
        for width and heightand key pixels which is a list of numbers
        to represent a greyscale image
        func (func): Any function that changes the value of a pixel

    Returns:
        A new image (dict) with pixels changed based on the given function
    """
    # create new image
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image["pixels"][:],
    }

    # nested loop allows us to apply func for every pixel
    for col in range(image["width"]):
        for row in range(image["height"]):
            pixel = get_pixel(image, row, col)  # changed variable name for
            # clarity
            new_pixel_color = func(pixel)  # apply function
            # set pixel value
            set_pixel(result, row, col, new_pixel_color)

    return result


def inverted(image):
    """
    Inverts a grey scale image by inverting each pixel
    For example 255 inverts to 0 and 113 inverts to 142

    Args:
        image (dict): contains keys width and height that store values
        for width and heightand key pixels which is a list of numbers
        to represent a greyscale image

    Returns:
        dict: A new inverted version of the image
    """
    return apply_per_pixel(image, lambda color: 255 - color)  # invert each pixel


# HELPER FUNCTIONS


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    Kernel is a a dictionary with keys width and height whose
    values are ints describing the width and height in pixels,
    respectively of a grey scale image. The third key is pixels whose
    value is a list of scaling values. The kernel has the same
    structure and keys of an image so that it can be treated the
    same way in my code and because it is easy to think about
    everything in the same structure.
    """

    if boundary_behavior not in ["zero", "extend", "wrap"]:
        return None

    # create correlated image
    correlated_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image["pixels"][:],
    }

    # generate a list of tuples of all pixel coordinates (row, col)
    pixel_coords = []
    for i in range(len(image["pixels"])):
        col = i % image["width"]
        row = i // image["width"]
        pixel_coords.append((row, col))

    correlated_pixels = []

    # generate row, col for each pixel in the kernel
    # based on the pixel
    # from the image that is being used
    for pix in pixel_coords:
        dist_from_center = int(
            (kernel["width"] - 1) / 2
        )  # calculate how far up, down, left, and right kernel will extend
        up, down = (
            pix[0] - dist_from_center,
            pix[0] + dist_from_center,
        )  # determine up and down dist
        left, right = (
            pix[1] - dist_from_center,
            pix[1] + dist_from_center,
        )  # determine left and right dist
        kernel_coords = []
        for x in range(left, right + 1):
            for y in range(up, down + 1):
                kernel_coords.append((y, x))  # create list kernel coords
                # flip x and y

        # apply the kernel to each pixels
        new_pix = 0
        for coord in kernel_coords:
            kernel_val = get_pixel(
                kernel,
                coord[0] - pix[0] + dist_from_center,
                coord[1] - pix[1] + dist_from_center,
                boundary_behavior,
            )  # calculate correct position and make sure it is in kernel
            image_val = get_pixel(
                image, coord[0], coord[1], boundary_behavior
            )  # get image val pixel based on shifted kernel coords
            new_pix += kernel_val * image_val

        correlated_pixels.append(new_pix)

    correlated_image["pixels"] = correlated_pixels

    return correlated_image


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    # go through each pixel
    for i in range(len(image["pixels"])):
        if image["pixels"][i] > 255:
            image["pixels"][i] = 255  # if val greater than 255 make it 255
        if image["pixels"][i] < 0:
            image["pixels"][i] = 0  # if val less than 0 make it 0
        image["pixels"][i] = round(
            image["pixels"][i]
        )  # round value to ensure it is an int and not a float


# FILTERS


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    """

    box_blur = create_box_blur(kernel_size)
    blurred_image = correlate(
        image, box_blur, "extend"
    )  # correlate the image based on the box blur generated
    round_and_clip_image(blurred_image)  # make sure image has valid pixels

    return blurred_image


def sharpened(image, n):
    """

    Applies a sharpening filter on the image by scaling the original
    image by 2 and subtracting a blurred version of the image using a
    box blur of size n x n

    Args:
        image (dict): contains keys width and height that store values
        for width and height and key pixels which is a list of numbers
        to represent a greyscale image
        n (int): The dimension of a box blur; an odd number that creates
        a blur kernel

    Returns:
        image (dict): A new dict that represents the sharpened version of
        the original greyscale image
    """
    # scale pixels by 2
    scaled_pixels = []
    for pixel in image["pixels"]:
        scaled_pixels.append(pixel * 2)

    blur_version = blurred(image, n)

    sharp_pixels = []

    for i in range(len(image["pixels"])):
        sharp_pixels.append(
            scaled_pixels[i] - blur_version["pixels"][i]
        )  # apply sharpen equation to each pixel
    sharpened_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": sharp_pixels,
    }
    round_and_clip_image(
        sharpened_image
    )  # make sure all pixels in sharpened image are valid

    return sharpened_image


def create_box_blur(n):
    """
    Helper function that generates a box blur kernel

    Args:
        n (int) = dimension of box, an odd number

    Returns:
        dict: with keys height and width that both have thevalue n
        and a key of pixels which is a list containing the scaling
        value of the box blur for each pixel which is 1/n**2
    """
    total_len = n * n  # determine total len of pixel list
    pixels = [1 / (n**2)] * total_len
    box_blur = {
        "width": n,
        "height": n,
        "pixels": pixels,
    }  # create dict form of box blur kernel
    return box_blur


def edges(image):
    """
    A filter that emphasizes the edges of an image and returns a new
    image with emphasized edges. To emphasize the edges two predetermined
    kernels are applied to the image

    Args:
        image (dict): contains keys width and height that store values
        for width and height and key pixels which is a list of numbers
        to represent a greyscale image

    Returns:
        dict: The sharpened verion of the grey scale image
        with keys width, height, and pixels. Width and height
        have int values representing the dimensions of the image.
        pixels containsa list of the sharpened pixel
        values for the new image
    """

    # create K1 and K2
    k_1 = {"width": 3, "height": 3, "pixels": [-1, -2, -1, 0, 0, 0, 1, 2, 1]}
    k_2 = {"width": 3, "height": 3, "pixels": [-1, 0, 1, -2, 0, 2, -1, 0, 1]}

    # correlate image using K_1 and K_2
    output_1 = correlate(image, k_1, "extend")
    output_2 = correlate(image, k_2, "extend")

    final_output_pixels = []
    # apply formula to each pixel using output lists
    for i in range(len(image["pixels"])):
        output_pix = round(
            math.sqrt((output_1["pixels"][i] ** 2) + (output_2["pixels"][i] ** 2))
        )
        final_output_pixels.append(output_pix)

    # create new edge image
    edge_image = {
        "width": image["width"],
        "height": image["height"],
        "pixels": final_output_pixels,
    }
    round_and_clip_image(edge_image)  # make sure pixels in edge_image are valid

    return edge_image


# VARIOUS FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def color_handler(image):
        image_copy = {
            "width": image["width"],
            "height": image["height"],
            "pixels": image["pixels"][:],
        }
        grey_scale_images = split_into_greyscale_images(image_copy)
        filtered_images = [
            filt(grey_scale_images[0]),
            filt(grey_scale_images[1]),
            filt(grey_scale_images[2]),
        ]
        color_image = combine_greyscale_images_to_rgb(filtered_images)
        return color_image

    return color_handler


def split_into_greyscale_images(image):
    """_summary_
    Splits a color image into three grey scale images
    representing red, green, and blue pixels

    Args:
        image (dict): contains keys width and height that
        store values for width and height and
        key pixels which is a list of tuples representing
        red, green, and blue pixel values
    """

    red_pixels, green_pixels, blue_pixels = [], [], []
    # seperate pixel values
    for rgb_pixel in image["pixels"]:
        red_pixels.append(rgb_pixel[0])
        green_pixels.append(rgb_pixel[1])
        blue_pixels.append(rgb_pixel[2])

    red_im = {"width": image["width"], "height": image["height"], "pixels": red_pixels}
    green_im = {
        "width": image["width"],
        "height": image["height"],
        "pixels": green_pixels,
    }
    blue_im = {
        "width": image["width"],
        "height": image["height"],
        "pixels": blue_pixels,
    }
    split_images = [red_im, green_im, blue_im]

    return split_images


def combine_greyscale_images_to_rgb(split_images):
    """
    Creates a color image given grey scale images
    representing red, green, and blue values

    Args:
        split_images (list): A list of 3 grey scale
        image dictionaries representing the red, green,
        and blue parts of a color image

    Returns:
        dict: A color image whose pixels are the combined
        red, green, and blue components of a color image
    """

    summed_pixels = []
    red_pic = split_images[0]
    width = red_pic["width"]
    height = red_pic["height"]
    red_im, green_im, blue_im = split_images[0], split_images[1], split_images[2]
    red_vals, green_vals, blue_vals = (
        red_im["pixels"],
        green_im["pixels"],
        blue_im["pixels"],
    )
    for i in range(len(split_images[0]["pixels"])):
        # combine pixels
        rgb_new = (red_vals[i], green_vals[i], blue_vals[i])
        summed_pixels.append(rgb_new)

    combined_image = {"width": width, "height": height, "pixels": summed_pixels}
    return combined_image


def make_blur_filter(kernel_size):
    """
    returns a blur filter so that the helper function
    color_filter_from_greyscale_filter can be
    utilized

    Args:
        kernel_size (int): size of blur kernel

    Returns:
        A blur filter
    """

    def blur_image(image):
        return blurred(image, kernel_size)

    return blur_image


def make_sharpen_filter(kernel_size):
    """
    returns a sharp filter so that the helper function
    color_filter_from_greyscale_filter can be
    utilized

    Args:
        kernel_size (int): size of sharp kernel

    Returns:
        A sharp filter
    """

    def sharp_image(image):
        return sharpened(image, kernel_size)

    return sharp_image


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def apply_outer_filter(image):
        current_photo = image
        for i in range(len(filters)):
            current_filt = filters[i]
            # apply filter to current_photo
            current_photo = current_filt(current_photo)
        return current_photo

    return apply_outer_filter


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """

    final_image = image
    # remove seams
    for num in range(ncols):
        grey_scale_version = greyscale_image_from_color_image(final_image)
        energy_im = compute_energy(grey_scale_version)
        cumulative_map = cumulative_energy_map(energy_im)
        min_seam = minimum_energy_seam(cumulative_map)
        final_image = image_without_seam(final_image, min_seam)

    return final_image


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    new_pixels = []
    for pix in image["pixels"]:
        new_pix = round((pix[0] * 0.299) + (pix[1] * 0.587) + (pix[2] * 0.114))
        new_pixels.append(new_pix)

    final_grey_im = {
        "width": image["width"],
        "height": image["height"],
        "pixels": new_pixels,
    }
    return final_grey_im


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """

    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """

    energy_pixels = []  # pixels values
    cumulative_map = {
        "width": energy["width"],
        "height": energy["height"],
        "pixels": energy_pixels,
    }
    # search 3 adjacent above pixels for least enegry
    for row in range(energy["height"]):
        for col in range(energy["width"]):
            current_energy = get_pixel(energy, row, col, "extend")
            if row == 0:
                energy_pixels.append(current_energy)
            else:
                adj_pix = [
                    get_pixel(cumulative_map, row - 1, col - 1, "extend"),
                    get_pixel(cumulative_map, row - 1, col, "extend"),
                    get_pixel(cumulative_map, row - 1, col + 1, "extend"),
                ]
                lowest_energy = min(adj_pix)
                final_energy = current_energy + lowest_energy
                energy_pixels.append(final_energy)

    return cumulative_map


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    # find lowest energy pixel
    last_row = cem["pixels"][((cem["height"] - 1) * cem["width"]) :]
    seam = []
    start_value = min(last_row)
    # get index of that value then search adjacent pixels
    current_col = last_row.index(start_value)
    seam.append(current_col + ((cem["height"] - 1) * cem["width"]))
    # check next row for lowest pixel
    for row in reversed(range(cem["height"] - 1)):
        if current_col == 0:
            adj_pix = [
                float("inf"),
                get_pixel(cem, row, current_col, "extend"),
                get_pixel(cem, row, current_col + 1),
            ]
        else:
            adj_pix = [
                get_pixel(cem, row, current_col - 1, "extend"),
                get_pixel(cem, row, current_col, "extend"),
                get_pixel(cem, row, current_col + 1, "extend"),
            ]

        # find lowest energy pixel
        lowest_energy = min(adj_pix)
        ind = adj_pix.index(lowest_energy)

        index_final = (cem["width"] * row) + current_col + ind - 1

        current_col = current_col + ind - 1  # update col

        seam.append(index_final)

    return seam


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """

    seam_indicies = seam[:]  # SORT in descending order
    seam_indicies.sort(reverse=True)
    seamed_image = {
        "width": image["width"] - 1,
        "height": image["height"],
        "pixels": image["pixels"][:],
    }

    for ind in seam_indicies:
        # remove pixel
        seamed_image["pixels"].pop(ind)

    return seamed_image


def custom_feature(image):
    """
    Takes in a color image and randomly adds
    white sparkles onto the image
    Uses the random import to randomly generate
    when to start a sparkle

    Args:
        image (dict): contains keys width and height that
        store values for width and height and key pixels
        which is a list of numbers to represent a greyscale
        image

    Returns:
        A color image (dict)
    """
    # center of sparkle coords
    center_sparkles = []
    sparkle_pix = image["pixels"][:]
    sparkled_image = {
        "width": image["width"],
        "height": image["height"],
        "pixels": sparkle_pix,
    }

    # randomly choose center pixels
    for row in range(3, image["height"] - 4):
        for col in range(3, image["width"] - 4):
            current_pix = (row, col)
            if random.randint(0, 250) == 1:
                center_sparkles.append(current_pix)  # append the tuple for the image

    # set to 255 for pixels surrounding
    adj_pixels = []
    for coord in center_sparkles:
        row, col = coord[0], coord[1]
        # find near pixels in sparkle
        surrounding_pixels = [
            (row, col),
            (row - 1, col),
            (row - 2, col),
            (row - 3, col),
            (row - 1, col - 1),
            (row - 2, col - 1),
            (row - 3, col - 1),
            (row - 1, col + 1),
            (row - 2, col + 1),
            (row - 3, col + 1),
            (row, col - 1),
            (row, col - 2),
            (row, col - 3),
            (row - 1, col - 1),
            (row - 2, col - 2),
            (row - 3, col - 3),
            (row + 1, col - 1),
            (row + 2, col - 2),
            (row + 3, col - 3),
            (row, col + 1),
            (row, col + 2),
            (row, col + 3),
            (row - 1, col + 1),
            (row - 2, col + 2),
            (row - 3, col + 3),
            (row + 1, col + 1),
            (row + 2, col + 2),
            (row + 3, col + 3),
            (row + 1, col),
            (row + 2, col),
            (row + 3, col),
            (row + 1, col - 1),
            (row + 2, col - 2),
            (row + 3, col - 3),
            (row + 1, col + 1),
            (row + 2, col + 2),
            (row + 3, col + 3),
        ]
        adj_pixels.extend(surrounding_pixels)

    # make sparkle pixels white
    for coord in adj_pixels:
        set_pixel(sparkled_image, coord[0], coord[1], (255, 255, 255))

    return sparkled_image


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    # cat_pic = load_color_image("test_images/cat.png")
    # color_version = color_filter_from_greyscale_filter(inverted)
    # color_final = color_version(cat_pic)
    # color_cat = save_color_image(color_final,
    # "test_results/colorcat.png", mode="PNG")

    # python_pic = load_color_image("test_images/python.png")
    # color_py = color_filter_from_greyscale_filter(make_blur_filter(9))
    # color_py_f = color_py(python_pic)
    # color_p_f = save_color_image(color_py_f,
    # "test_results/colorpython.png", mode="PNG")

    # sparrow_pic = load_color_image("test_images/sparrowchick.png")
    # color_spar = color_filter_from_greyscale_filter(make_sharpen_filter(7))
    # color_spar_f = color_spar(sparrow_pic)
    # color_sp_f = save_color_image(
    #     color_spar_f, "test_results/sparrowsharp.png", mode="PNG"
    # )
    # pattern = load_color_image("test_images/pattern.png")
    # pattern_carve = seam_carving(pattern, 1)
    # pattern_final = save_color_image(
    #     pattern_carve, "test_results/patternfinal.png", mode="PNG"
    # )
    # print("pattern_image", pattern_carve)
    # pattern_correct = load_color_image("test_results/pattern_1seam.png")
    # print("pattern_1_seam", pattern_correct)
    # if pattern_carve == pattern_correct:
    #     print("yay")

    # cat = load_color_image("test_images/cat.png")
    # sparkle = sparkles(cat)
    # final_sparkle = save_color_image(sparkle, 
    # "test_results/sparklecat_new.png", mode="PNG")
    # if cat != sparkle:
    #     print("not the same")

    # blue_fish = load_color_image("test_images/bluegill.png")
    # sparkle_fish = sparkles(blue_fish)
    # fish_sparkle = save_color_image(sparkle_fish, 
    # "test_results/sparklefish__.png", mode="PNG")

    # two_cat = load_color_image("test_images/twocats.png")
    # cat_carved = seam_carving(two_cat, 100)
    # final_cats = save_color_image(
    #     cat_carved, "test_results/twocatscarved.png", mode="PNG"
    # )
    filter1 = color_filter_from_greyscale_filter(edges)
    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    filt = filter_cascade([filter1, filter1, filter2, filter1])
    frog_pic = load_color_image("test_images/frog.png")
    frog_final = filt(frog_pic)
    final_frog = save_color_image(frog_final, "test_results/frogfilt.png", mode="PNG")
