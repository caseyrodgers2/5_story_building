# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:44:54 2021

@author: Casey Rodgers

Creates realistic Reinforced Concrete textures for blender.
Creates these images:
    - RGB image (diffuse (aka base colors))
    - Bump map (small details)
    - Displacement map (large details)
    - Metallic image (shows shininess (0=rough, 1=shiny))
    
Reference: GenConcreteDamageTexture.py by Yasutaka Narazaki
    - https://answers.opencv.org/question/163561/looking-for-a-thinningskeletonizing-algorithm-with-opencv-in-python/
    
References: 
    -Adobe Photoshop by Adobe. https://www.adobe.com/products/photoshop.html 
    
    // Finding Damage
    -Real Python. “Image Segmentation Using Color Spaces in Opencv + Python.” Real Python, Real Python, 7 Nov. 2020, https://realpython.com/python-opencv-color-spaces/. 
    -Hacklavya, et al. “OpenCV Tutorial: A Guide to Learn Opencv.” PyImageSearch, 17 Apr. 2021, https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/. 
    
    // Generating Cracks
    -Zhang, T. Y., Suen, C. Y.: A Fast Parallel Algorithm for Thinning Digital Patterns. Communications of ACM. 27(3), 236–239 (1984) 
    -“Looking for a Thinning/Skeletonizing Algorithm with Opencv in Python. Edit.” Looking for a Thinning/Skeletonizing Algorithm with Opencv in Python. - OpenCV Q&A Forum, https://answers.opencv.org/question/163561/looking-for-a-thinningskeletonizing-algorithm-with-opencv-inpython/. 
    -“Concrete 024 on Ambientcg.” AmbientCG, https://ambientcg.com/view?id=Concrete024. 
    
    // Generating Spalling
    -Travall. “Procedural 2d Island Generation - Noise Functions.” Medium, Medium, 5 Aug. 2020, https://medium.com/@travall/procedural-2d-island-generation-noise-functions-13976bddeaf9. 
    -Textures for 3D, Graphic Design and Photoshop! https://www.textures.com/download/MetalVarious0041/52051.  
    -“Paste Another Image into an Image with Python, Pillow.” Paste Another Image into an Image with Python, Pillow, https://note.nkmk.me/en/python-pillow-paste/. 
    -“How Do I Increase the Contrast of an Image in Python Opencv.” Stack Overflow, 1 Oct. 1964, https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv. 
    
    // Blender
    -“Setup a Skybox Using the Sky Texture in Blender.” Artisticrender.com, 27 May 2021, https://artisticrender.com/setup-a-skybox-using-the-sky-texture-in-blender/. 
    -Burley, B.; Disney, W.; Studios, A. Physically Based Shading at Disney 
    
    
"""

# Import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import imutils
from PIL import Image, ImageDraw, ImageChops
import noise

from undamaged_textures import UndamagedTex


class PhysicsDamage():
    
    # init method or constructor
    def __init__(self):
        self.ut = UndamagedTex()    # Create an UndamagedTex object
    
    
    
    ## Crop Photo to get rid of the background
    def cropBorder(self, img, imgRef, borderColor, scale):
        """ 
        Crop out the border of an image. (Based on first pixel to be less than borderColor in upper left corner 
        and bottom right corner)
        
        img = image we want cropped (abaqus screenshot) array
        imgRef = reference image to find cropping coords array (if == 0, then imgRef = img)
        borderColor = color of border we want to crop to as 3-tuple (1, 2, 3)
        scale = how much larger do you want texture than original image
        
        Returns cropped image array and top left corner coords
        """
        if (type(imgRef) is int):
            imgRef = img
            
        # Find the top left coords
        (h, w, d) = imgRef.shape  # Image shape
        #print("width={}, height={}, depth={}".format(w, h, d))
        
        topLeftX = 0  # Initialize top left x coord
        topLeftY = 0  # Initialize top left y coord
        stop = False  # When to stop?
        
        # Go through image and find first pixel that is less than the border color
        for y in range(h):
            for x in range(w):
                #print("x = {}".format(x))
                #print("y = {}".format(y))
                (B, G, R) = imgRef[y, x]
                if (np.all((B, G, R) <= borderColor)):
                    stop = True
                    topLeftX = x
                    topLeftY = y
                    break
            if stop:
                break
        
        # Find the bottom right coords
        stop = False   # When to stop?
        botRightX = 0  # Initialize bottom right x coord
        botRightY = 0  # Initialize bottom right y coord
        
        # Go through image and find first pixel that is less than the border color
        for y in range(h):
            for x in range(w):
                (B, G, R) = imgRef[h - 1 - y, w - 1 - x]
                #print(w - 1 - x)
                #print(h - 1 - y)
                #print("R={}, G={}, B={}".format(R, G, B))
                if (np.all((B, G, R) <= borderColor)):
                    stop = True
                    botRightX = w - 1 - x
                    botRightY = h - 1 - y
                    break
            if stop:
                break     
            
        # Crop photo
        #print("topLeft x = {}".format(topLeftX))
        #print("topLeft y = {}".format(topLeftY))
        #print("botRight x = {}".format(botRightX))
        #print("botRight y = {}".format(botRightY))
        image2 = img[topLeftY:botRightY, topLeftX:botRightX]
        #print(image2)
        
        # Resize photo to get more detail later
        (h2, w2, d2) = image2.shape
        resized = cv2.resize(image2, (w2*scale, h2*scale))
        #print("width={}, height={}, depth={}".format(w2, h2, d2))
        #plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        #plt.title("cropped")
        #plt.show()
    
        return resized, topLeftX, topLeftY
    
    
    
    ## Identify Damaged Areas
    def findDamage(self, img):
        """
        Find damage using contours
        
        img = input img of cropped abaqus screenshot
        
        Returns contours
        """
        
        # Convert the image to HSV to isolate the red color.
        # Find lower and upper bounds. 
        # Isolate red color and make everything else black.
        image4 = img.copy()
        hsv = cv2.cvtColor(image4, cv2.COLOR_BGR2HSV)
        lower = (0, 255, 50)
        upper = (20, 255, 226)
        mask = cv2.inRange(hsv, lower, upper)
        # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        # plt.title("damage")
        # plt.show()
        
        # Get rid of the grid lines. First invert to black and white
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)  # Invert black and white
        
        # Get opening morphology to get rid of it
        ksize1 = int(max(image4.shape) / 300)
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize1, ksize1))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k1)
        
        # Invert back
        ret2, thresh2 = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY_INV)
        #plt.imshow(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB))
        #plt.title("no grid")
        #plt.show()
        
        # Get contours
        cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Print out contours
        output = img.copy()
        for c in cnts:
            # Draw each contour on the output image with a 3px thick purple outline, then display 
            # the output contours one at a time
            cv2.drawContours(output, [c], -1, (240, 0, 159), 15)
            #plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            #plt.show()
        
        plt.figure(figsize=(8,8))
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Damage Contours')
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.show()
        
        return cnts
    
    
    ## Create a Perlin Noise Texture
    def createPerlin(self, shape, scale, octaves, lac, pers):
        """
        Create a Perlin Noise Texture
        
        shape = shape of texture (height, width)
        scale = at what distance you're viewing the texture
        octaves = number of levels of detail 
        lac = lacunarity (aka frequency) = determines how much detail is added / removed at each octave 
        pers = persistence = determines how much each octave contributes to overall
        
        Returns perlin texture (image)
        """
        
        perlin_tex = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                perlin_tex[i][j] = noise.snoise2(i/scale, j/scale, octaves=octaves,
                                           persistence=pers, lacunarity=lac)
        
        # Convert Perlin noise from [-1, 1] to [0, 1]
        perlin_tex = perlin_tex * 0.5 + 0.5
        perlin_tex = Image.fromarray(perlin_tex * 255)
        
        return perlin_tex
    
    
    
    ## Create thin cracks
    def createThinCracks(self, cracks_mask):
        """
        Generate cracks when there's a thinner mesh and the cracks are more defined
        
        cracks_mask = mask where cracks are drawn
        
        Returns crack displacement map
        """
        
        #print('Making cracks from thin mesh')
        
        cracks_mask2 = cv2.cvtColor(cracks_mask.copy(), cv2.COLOR_BGR2GRAY)
        
        max_dim = max(cracks_mask2.shape)
        
        # plt.imshow(cv2.cvtColor(cracks_mask2, cv2.COLOR_GRAY2RGB))
        # plt.title("cracks mask")
        # plt.show()
        
        # Try to close some gaps so the cracks don't look weird
        ksize1 = int(max_dim / 300)
        #print(ksize1)
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize1, ksize1))
        cracks_mask2 = cv2.dilate(cracks_mask2, k1)
        
        plt.imshow(cv2.cvtColor(cracks_mask2, cv2.COLOR_GRAY2RGB))
        plt.title("cracks mask 2")
        plt.show()
        
        thinned = cv2.ximgproc.thinning(cracks_mask2)  # Main thin line
        thinned[thinned > 0] = 255
        
        # Make line a little thicker
        thinned2 = cv2.dilate(thinned, None, iterations = 2)
        
        crack_lines = thinned2.copy()
        
        #plt.imshow(cv2.cvtColor(thinned, cv2.COLOR_GRAY2RGB))
        #plt.title("thinned")
        #plt.show()
        
        # Create Perlin Noise Texture
        (h2, w2, d) = cracks_mask.shape
        shape = (h2, w2)
        scale = ksize1
        #print(scale)
        octaves = 8
        pers = 0.5
        lac = 2.0
        perlin = self.createPerlin(shape, scale, octaves, lac, pers)
        perlin = perlin.convert("RGB")
        #perlin.show()
        perlin_arr = np.array(perlin)
        perlin_arr = cv2.cvtColor(perlin_arr, cv2.COLOR_RGB2GRAY)
        
        # Blur cracks, add thin line, then repeat (if needed)
        output = thinned2.copy()
        sigma = int(max_dim / 600)
        
        for i in range(4):
            
            output[thinned2 > 0] = 255
    
            output = cv2.GaussianBlur(output, (0, 0), sigma, sigma)
            
            # plt.imshow(cv2.cvtColor(output, cv2.COLOR_GRAY2RGB))
            # plt.title("blurred")
            # plt.show()
            
        #print(perlin_arr.max())
        #print(output.max())
            
        # Blend perlin noise in with crack map
        #plt.imshow(cv2.cvtColor(perlin_arr, cv2.COLOR_GRAY2RGB))
        #plt.title("perlin")
        #plt.show()
        
        #plt.imshow(cv2.cvtColor(output, cv2.COLOR_GRAY2RGB))
        #plt.title("output")
        #plt.show()
        
        output_fl = np.float64(output) / 255.0          # Convert to float for mult
        perlin_arr_fl = np.float64(perlin_arr) / 255.0  # Convert to float for mult
        
        output2_fl = cv2.multiply(output_fl, perlin_arr_fl)
        output2_fl = np.clip(output2_fl, 0, 1)
        output2 = (output2_fl * 255).astype(np.uint8)
        #print(output2.max())
        #print(output2.shape)
        
        #plt.imshow(cv2.cvtColor(output2, cv2.COLOR_GRAY2RGB))
        #plt.title("combined")
        #plt.show()
        
        # Add main crack line back
        output2[thinned > 0] = 255
        
        # Cut off the mask
        output2 = cv2.bitwise_and(output2, output2, mask=cracks_mask2)
        
        # plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        # plt.title("final")
        # plt.show()
        
        return output2, crack_lines
    
    
    
    ## Draw Cracks
    def drawCracks(self, img, material, cnts, scale, yesCracks):
        """
        Go through each contour and determine whether it's a crack or spalling
        
        img = input image of abaqus screenshot
        material = material image
        cnts = contours of damage
        scale = scale of output texture you want
        yesCracks = 1: yes we want cracks. 0: all spalling
        
        Returns cracks image array, cracks mask array, spalling mask array
        """
        
        outline = img.copy()
        #cracks = material.copy()
        (h, w, d) = img.shape
        spall_mask = np.zeros(outline.shape,np.uint8)
        cracks = np.zeros(outline.shape,np.uint8)
        cracks_line = np.zeros(outline.shape,np.uint8)
        curr_index = 0
        
        for c in cnts:
            area = cv2.contourArea(c)
            #print("area = {}".format(area))
            perimeter = cv2.arcLength(c, True)
            #print("perimeter = {}".format(perimeter))
            if (perimeter <= 0 or area <= 0):
                continue
            ratio = area / perimeter
            #print("ratio = {}".format(ratio))
            
            # Crack
            
            if (ratio < scale*10.0 and area > scale*20.0 and yesCracks == 1):
                #print("area = {}".format(area))
                #print("perimeter = {}".format(perimeter))
                
                cv2.drawContours(cracks, [c], -1, (255, 255, 255), -1)
                cv2.drawContours(outline, [c], -1, (240, 0, 159), 3)
                
            # Save indices for spalling for later
            elif (area > scale*20.0 or yesCracks == 0):
                cv2.drawContours(spall_mask,[c],-1,(255, 255, 255),-1)
                cv2.drawContours(outline, [c], -1, (240, 0, 159), 3)
            
            curr_index = curr_index + 1
        
        # For coarse mesh, blur cracks
        if (yesCracks == 1):
            # For fine mesh, add cracks
            cracks2, cracks_line = self.createThinCracks(cracks)
                
        else:
            
            #cracks2 = cracks.copy()
            
            cracks2, cracks_line = self.createThinCracks(spall_mask)  # Generate thin cracks
            
            plt.imshow(cv2.cvtColor(cracks2, cv2.COLOR_BGR2RGB))
            #plt.title("Contour Outlines")
            plt.show()
    
            # Make thin cracks a bit bigger
            cracks2 = cv2.GaussianBlur(cracks2, (0, 0), 3*scale)
            cracks2 = cv2.dilate(cracks2, None, iterations=scale)  # Create spalling shadow outline
            
            cracks2 = cracks2 + (255 - np.amax(cracks2))  # Make it lighter, so it will appear darker later
                
            
        #plt.imshow(cv2.cvtColor(outline, cv2.COLOR_BGR2RGB))
        #plt.title("Contour Outlines")
        #plt.show()
        
        return spall_mask, cracks2, cracks_line
    
    
    
    ## Create rebar
    def createRebar(self, w, h, actW, rebar_type, long_rebar_fn, stir_rebar_fn):
        """
        Creates a rebar profile used to draw rebar to the image later.
        
        w = width of image needed (same size as undamaged textures)
        h = height of image needed (same size as undamaged textures)
        actW = actual structure width (m)
        rebar_type = 0: longitudinal rebar. 1: stirrup rebar
        long_rebar_fn: longitudinal rebar file name
        stir_rebar_fn: Stirrup rebar file name
        
        Returns crack displacement map
        """
        
        if (rebar_type == 0):
            
            ## Longitudinal Bars
            rebar_tex = Image.open(long_rebar_fn)
            #rebar_tex.show()
            rebar_w, rebar_h = rebar_tex.size  # texture size
            
            # Figure out dimensions we want. There are 1 rebar per photo
            ratio_pix = w / actW        # ratio of pix / m
            #print("ratio_pix={}".format(ratio_pix))
            
            longi_space = ratio_pix * 0.15    # Longitudinal Spacing (150 mm)
            #print("longi_space={}".format(longi_space))
            
            clear_cover = int(ratio_pix * 0.0118)  # Clear cover 0.0118 m (3 in)
            #print(clear_cover)
            
            rebar_act_width = 0.0286512       # Num 9 rebar diameter (1.128 in)
            new_width = int(ratio_pix * rebar_act_width) 
            #print("new_width={}".format(new_width))
            
            new_height = int(rebar_h / rebar_w * new_width)  # New height for resized rebar
            
            #print("rebar_h={}".format(rebar_h))
            #print("rebar_w={}".format(rebar_w))
            #print("new_height={}".format(new_height))
            
            rebar_tex_resized = rebar_tex.resize((new_width, new_height))  # Resize rebar
            #rebar_tex_resized.show()
            rebar_tex_re2 = rebar_tex_resized.copy()
            
            
            # Paste enough rebar textures to cover cropped image
            # Convert rebar profile to Image type
            rebar_profile = np.zeros((h, w),np.uint8)
            rebar_profile = Image.fromarray(rebar_profile)
            rebar_profile = rebar_profile.convert("RGB")
            pasteX = clear_cover - 1
            pasteY = 0
            
            while (pasteX < w - clear_cover):
                #print("pasteX={}".format(pasteX))
                
                while (pasteY < h):
                    #print("pasteY={}".format(pasteY))
                    rebar_profile.paste(rebar_tex_re2, (pasteX, pasteY))
                    pasteY = pasteY + new_height
                
                longi_space2 = int(longi_space * random.uniform(0.8, 1.2))    # Longitudinal Spacing Add random
                pasteX = pasteX + longi_space2
                pasteY = 0
        
        
        elif (rebar_type == 1):
            
            ## Stirrups
            rebar_tex = Image.open(stir_rebar_fn)
            rebar_w, rebar_h = rebar_tex.size             # texture size
            
            # Figure out dimensions we want. There are 1 rebar per photo
            ratio_pix = w / actW        # ratio of pix / m
            #print("ratio_pix={}".format(ratio_pix))
            
            stir_space = int(ratio_pix * 0.2)      # Stirrup spacing (200 mm)
            
            clear_cover = int(ratio_pix * 0.0118)     # 0.0118 m (3 in)
            #print(clear_cover)
            
            rebar_act_width = 0.0127           # Num 4 rebar diameter (0.5 in)
            new_height = int(ratio_pix * rebar_act_width)  # New height for rebar
            #print("new_width={}".format(new_width))
            
            new_width = int(rebar_w / rebar_h* new_height) # New width for rebar
            #print("rebar_h={}".format(rebar_h))
            #print("rebar_w={}".format(rebar_w))
            #print("new_height={}".format(new_height))
            
            rebar_tex_resized = rebar_tex.resize((new_width, new_height))  # Resize rebar
            #rebar_tex_resized.show()
            rebar_tex_re2 = rebar_tex_resized.copy()
            
            
            # Paste enough rebar textures to cover cropped image
            # Convert rebar profile to Image type
            rebar_profile = np.zeros((h, w),np.uint8)
            rebar_profile = Image.fromarray(rebar_profile)
            rebar_profile = rebar_profile.convert("RGB")
            pasteX = clear_cover - 1
            pasteY = 0
            
            while (pasteY < h):
                #print("pasteX={}".format(pasteX))
                
                while (pasteX < w - clear_cover):
                    #print("pasteY={}".format(pasteY))
                    rebar_profile.paste(rebar_tex_re2, (pasteX, pasteY))
                    pasteX = pasteX + new_width
                
                stir_space2 = int(stir_space * random.uniform(0.8, 1.2))      # Stirrup spacing
                pasteY = pasteY + stir_space2
                pasteX = clear_cover - 1
    
        # Fix clear cover on right side
        #rebar_profile.show()
        d = ImageDraw.Draw(rebar_profile)
        coords = (w - clear_cover - 1, 0, w - 1, h - 1)
        d.rectangle(coords, fill="black")
        
        # Close open image and return rebar profile
        rebar_tex.close()
        #rebar_profile.show()
        return rebar_profile
    
    
    
    ## Draw spalling
    def drawSpalling(self, img, spall_mask, actW, rebar, scale, yesCracks, long_rebar_fn, stir_rebar_fn):
        """
        Draw out spalling with shadows and rebar
        
        img = image we want to add the spalling (rgb image)
        spall_mask = spalling mask from drawCracks
        actW = actual structure width (m)
        rebar = 0 for no rebar, 1 for rebar
        scale = scale you want the output texture to be
        yesCracks = 1: yes we want cracks. 0: all spalling
        long_rebar_fn: longitudinal rebar file name
        stir_rebar_fn: Stirrup rebar file name
        
        Return rgb image, displacement map, bump map, and metallic map
        """
        
        (h2, w2, d) = img.shape
        shape = (h2, w2)
                
        ## Create rgb spalling texture 
        spall_conc = np.clip(img.copy() + 25, 0, 255)       # Want spalling conc to be darker in color        
        spall_img = Image.fromarray(np.uint8(spall_conc))   # Convert img to Image type
        spall_img = spall_img.convert("L")
        #spall_img.show()
        #print(spall_img2.shape)
        #print(perlin_arr.shape)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbImage = Image.fromarray(img2)              # Convert img to Image type
        rgbImage = rgbImage.convert("RGB")
            
        # Combine spall_mask
        mask_img = Image.fromarray(np.uint8(spall_mask))  # Convert img to Image type
        mask_img = mask_img.convert("L")
        mask_img2 = mask_img.copy()
        rgbImage.paste(spall_img, mask_img2)  # Paste spalling conc to other conc with mask
        
        rgbImage.save("temp/spall_color.jpg", quality=95)
        
        ## Create bump texture
        # Make spall_mask dilated a bit
        spall_mask12 = spall_mask.copy()              
        ksize1 = int(max(spall_mask12.shape) / 1000)  # Kernel size
        kernel = np.ones((ksize1, ksize1),np.uint8)   # Kernel for dilating
        spall_mask12 = cv2.dilate(spall_mask12,kernel,iterations = 2)  # Dilate spalling mask
        mask_img3 = Image.fromarray(np.uint8(spall_mask12))  # Convert to Image type
        mask_img3 = mask_img.convert("L")
        
        bump_img = Image.new('L', (w2, h2))  # Create new black bump img
        
        # Create perlin noise texture
        scale = 50
        octaves = 8
        pers = 0.5
        lac = 2.0
        perlin = self.createPerlin(shape, scale, octaves, lac, pers)
        #perlin_arr = np.array(perlin)
        #perlin.show()
        
        bump_img.paste(perlin, (0, 0), mask_img3)  # Paste perlin noise texture to bump img
        
        bump_img.save("temp/spall_bump.jpg", quality=95)
        
        ## Create displacement texture
        spall_mask2 = spall_mask.copy()                           # Create copy of spalling mask
        spall_mask2 = cv2.erode(spall_mask2, None, iterations=2)  # Create spalling shadow outline
        
        #shadow_img2 = Image.fromarray(spall_mask2)
        
        
        # Spalling is large and don't want huge dips
        shadow_img = Image.fromarray(spall_mask2)
        #displ_img2.show()
        shadow_img2 = shadow_img.copy()
        shadow_img2.paste(shadow_img2, (0, 0), mask_img2)
        displ_img = shadow_img2.copy()
        
        
        
        ## Draw out rebar
        metal_img = Image.new('L', (w2, h2))  # Create new black rebar img
        
        if (rebar == 1):
            
            ## Longitudinal bars
            
            plt.imshow(cv2.cvtColor(spall_mask, cv2.COLOR_BGR2RGB))
            plt.title("Spall Mask")
            plt.show()
            
            # Erode spall_mask for rebar mask
            rebar_mask = spall_mask.copy()
            rebar_mask = cv2.erode(rebar_mask, None, iterations=int(scale))
            rebar_mask = cv2.cvtColor(rebar_mask, cv2.COLOR_BGR2GRAY)
            plt.imshow(cv2.cvtColor(rebar_mask, cv2.COLOR_BGR2RGB))
            plt.title("Eroded size")
            plt.show()
        
            # Create Rebar profile. Convert to opencv from pil
            w3, h3 = rgbImage.size
            rebar = self.createRebar(w3, h3, actW, 0, long_rebar_fn, stir_rebar_fn)
            #rebar.show()
            rebar_cv = np.array(rebar)
            rebar_cv = rebar_cv[:, :, ::-1].copy()  # Convert to BGR
            
            # Use rebar mask on rebar
            rebar_profile = cv2.bitwise_and(rebar_cv, rebar_cv, mask=rebar_mask)
            cv2.imwrite("temp/rebar_profile.jpg", rebar_profile)
            #plt.imshow(cv2.cvtColor(rebar_profile, cv2.COLOR_BGR2RGB))
            #plt.title("rebar")
            #plt.show()
            
            # Get gray rebar for pasting mask
            rebar_gray = rebar_profile.copy()
            rebar_gray = cv2.cvtColor(rebar_gray, cv2.COLOR_BGR2GRAY)
            rebar_bump = rebar_gray.copy()
            rebar_gray[rebar_gray != 0] = 255
            cv2.imwrite("temp/rebar_gray.jpg", rebar_gray)
            
            # Find rebar max and min and make contrast larger
            rebar_bump = cv2.addWeighted(rebar_bump, 4, rebar_bump, 0, -127)
            #print(np.amin(rebar_bump[rebar_bump>0]), np.amax(rebar_bump))
            #cv2.imwrite("temp/rebar_bump3.jpg", rebar_bump)
            
            # Make rebar bump darker
            rebar_bump = cv2.bitwise_not(rebar_bump)
            rebar_bump = cv2.cvtColor(rebar_bump, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("temp/rebar_bump.jpg", rebar_bump)
            
            rebar_bump2 = np.clip(rebar_bump - 50, 0, 255)
            rebar_bump2 = cv2.cvtColor(rebar_bump2, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("temp/rebar_bump2.jpg", rebar_bump2)
            
            # Add rebar bump map to bump and displ map
            rebar_grayImg = Image.open("temp/rebar_gray.jpg")
            rebar_bump_img = Image.open("temp/rebar_bump.jpg")
            rebar_bump_img2 = Image.open("temp/rebar_bump2.jpg")
            
            bump_img.paste(rebar_bump_img2, (0, 0), rebar_grayImg)            
            displ_img.paste(rebar_bump_img, (0, 0), rebar_grayImg)
            
            # Add rebar to spalling jpg
            rebar_final = Image.open("temp/rebar_profile.jpg")
            #rebar_final.show()
            rgbImage.paste(rebar_final, (0, 0), rebar_grayImg)
            
            rgbImage.save("temp/rebar1.jpg", quality=95)
                
            # Metal image and close open images
            metal_img = rebar_grayImg.copy()
            rebar_final.close()
            rebar_grayImg.close()
            rebar_bump_img.close()
            
            
            ## Stirrups bars
            
            # Erode spall_mask for rebar mask (want to make it look like rebar are going into sides)
            rebar_mask = spall_mask.copy()
            rebar_mask = cv2.erode(rebar_mask, None, iterations=int(scale/2))
            rebar_mask = cv2.cvtColor(rebar_mask, cv2.COLOR_BGR2GRAY)
            
            # Create Rebar profile. Convert to opencv from pil
            rebar = self.createRebar(w3, h3, actW, 1, long_rebar_fn, stir_rebar_fn)
            #rebar.show()
            rebar_cv = np.array(rebar)
            rebar_cv = rebar_cv[:, :, ::-1].copy()  # Convert to BGR
            
            # Use rebar mask on rebar
            rebar_profile = cv2.bitwise_and(rebar_cv, rebar_cv, mask=rebar_mask)
            cv2.imwrite("temp/rebar_profile2.jpg", rebar_profile)
            #plt.imshow(cv2.cvtColor(rebar_profile, cv2.COLOR_BGR2RGB))
            #plt.title("rebar")
            #plt.show()
            
            # Get gray rebar for pasting mask
            rebar_gray = rebar_profile.copy()
            rebar_gray = cv2.cvtColor(rebar_gray, cv2.COLOR_BGR2GRAY)
            rebar_bump = rebar_gray.copy()
            rebar_gray[rebar_gray != 0] = 255
            cv2.imwrite("temp/rebar_gray2.jpg", rebar_gray)
            
            # Find rebar max and min and make contrast larger
            rebar_bump = cv2.addWeighted(rebar_bump, 2, rebar_bump, 0, -127)
            #print(np.amin(rebar_bump[rebar_bump>0]), np.amax(rebar_bump))
            
            # Make rebar bump darker
            rebar_bump = cv2.bitwise_not(rebar_bump)
            rebar_bump = cv2.cvtColor(rebar_bump, cv2.COLOR_GRAY2BGR)
            rebar_bump = np.clip(rebar_bump - 100, 0, 255)
            rebar_bump = cv2.cvtColor(rebar_bump, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("temp/rebar_bump6.jpg", rebar_bump)
            
            # Add rebar bump map to bump and displ map
            rebar_grayImg = Image.open("temp/rebar_gray2.jpg")
            rebar_bump_img = Image.open("temp/rebar_bump6.jpg")
            
            bump_img.paste(rebar_bump_img, (0, 0), rebar_grayImg)
            displ_img.paste(rebar_bump_img, (0, 0), rebar_grayImg)
            
            # Add rebar to spalling jpg
            rebar_final = Image.open("temp/rebar_profile2.jpg")
            #rebar_final.show()
            rgbImage.paste(rebar_final, (0, 0), rebar_grayImg)
            
            rgbImage.save("temp/rebar2.jpg", quality=95)
            
            # Add
            metal_img.paste(rebar_grayImg, (0, 0), rebar_grayImg)
            rebar_final.close()
            rebar_grayImg.close()
            rebar_bump_img.close()
        
        # Close open images and return
        return rgbImage, displ_img, bump_img, metal_img
    
    
    
    ## Save the Images
    def saveImages(self, cracks_img, rgbImage, displ_img, bump_img, metal_img, mat_normal_im, mat_rough_im, mat_displ_im, spall_mask):
        """
        Save images to textures folder
        
        cracks =         Image. Crack map
        rgbImage =       Image. RGB image
        displ_img =      Image. Displacement image
        bump_img =       Image. Bump image
        metal_img =      Image. Metal image
        mat_normal_im =  np.ndarray. Material normal map
        mat_rough_im =   np.ndarray. Material roughness map
        mat_displ_im =   np.ndarray. Material displacement map
        spall_mask =     np.ndarray. Spalling mask
        
        Does not return anything. Saves textures to folder.
        """
        
        # Material Normal im
        mat_norm_img = Image.fromarray(np.uint8(mat_normal_im))
        mat_norm_img = mat_norm_img.convert("RGB")
        mat_norm_img.save("textures/mat_norm.jpg", quality=95)
        
        # Material Rough im
        mat_rough_img = Image.fromarray(np.uint8(mat_rough_im))
        mat_rough_img = mat_rough_img.convert("RGB")
        mat_rough_img.save("textures/mat_rough.jpg", quality=95)
        
        # Material Displacement im
        mat_displ_img = Image.fromarray(np.uint8(mat_displ_im))
        mat_displ_img = mat_displ_img.convert("RGB")
        mat_displ_img.save("textures/mat_displ.jpg", quality=95)
    
        # Cracks displacmenet image
        # cracks_img.paste(orig_img, (0, 0), mask)
    
        # Save cracks displ image
        cracks_img.save("textures/cracks_displ.jpg", quality=95)
    
        # Save rgb image
        rgbImage.save("textures/rgbImage.jpg", quality=95)
    
        # Displacment image
        displ_img.save("textures/displacement.jpg", quality=95)
    
        # Spalling Bump image
        #bump_img.paste(cracks_img, (0, 0), cracks_img)
        bump_img.save("textures/spall_bump.jpg", quality=95)
    
        # Metallic map image
        metal_img.save("textures/metallic.jpg", quality=95)
        
        # Spall Mask
        spall_mask_img = Image.fromarray(np.uint8(spall_mask))
        spall_mask_img = spall_mask_img.convert("RGB")
        spall_mask_img.save("textures/spall_mask.jpg", quality=95)
        
        
        
    ## Put it all together and create a damage conc tex
    def damageConcTextures(self, dam_map_fn, actW, conc_sample_file_name, conc_act_w, rebar, scale, yesCracks, long_rebar_fn, stir_rebar_fn):
        """
        Creates realistic damaged concrete textures from FEM damage map and one concrete sample
        
        dam_map_fn =            string. Filepath to FEM damage map image
        actW =                  float. Actual width of structure component
        conc_sample_file_name = string. Filepath to concrete material sample image.
        conc_act_w =            float. Actual width of concrete sample patch
        rebar =                 int. Is there rebar? 0 = No. 1 = Yes.
        scale =                 int. How many times larger do you want textures to be compared to the Damage map size?
        yesCracks =             int. Do you want cracks or to make it all spalling? 0 = No. 1 = Yes.
        long_rebar_fn =         longitudinal rebar file name
        stir_rebar_fn =         Stirrup rebar file name
        
        Does not return anything. Saves textures to folder.
        """
        
        ## Damage Image
        damage_img = cv2.imread(dam_map_fn)  # Damage Image
    
        plt.figure(figsize=(8,8))
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Damage Map')
        plt.imshow(cv2.cvtColor(damage_img, cv2.COLOR_BGR2RGB))
    
    
        ## Crop
        cropped, topLeftX, topLeftY = self.cropBorder(damage_img, 0, (0, 0, 0), scale)
    
        plt.figure(figsize=(8,8))
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Cropped Damage Map')
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
        
        ## Find Damage Contours
        cnts = self.findDamage(cropped)
        
        
        ## Read Concrete Material Sample
        sample_img = cv2.cvtColor(cv2.imread(conc_sample_file_name), cv2.COLOR_BGR2RGB)
        plt.imshow(sample_img)
        plt.tick_params(left = False, right = False , labelleft = False ,
         labelbottom = False, bottom = False)
        plt.title('Concrete Sample')
        plt.show()
    
        ## Create New Concrete Material Image that is same size as damage map
        (h, w, d) = cropped.shape
        (h_s, w_s, d_s) = sample_img.shape
    
        conc_mat = self.ut.createTexture(conc_sample_file_name, w, h, actW, conc_act_w)  # Create texture
    
        conc_mat = conc_mat[0:h, 0:w, :]  # Trim concrete material so it's same size as damage texture
    
        #print(cropped.shape)
        #print(conc_mat.shape)
    
        if conc_mat is not None:
            plt.figure(figsize=(8,8))
            #plt.figure()
            plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
            plt.title('Concrete')
            plt.imshow(conc_mat)
            
            
        ## Material Roughness and Displacement Map
        height_im = cv2.cvtColor(conc_mat, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(8,8))
        plt.imshow(height_im, cmap='gray')
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Gray Concrete Sample')
        plt.show()
    
        mat_rough_im, mat_displ_im = self.ut.roughAndDispl(height_im)
    
        plt.figure(figsize=(8,8))
        plt.imshow(mat_rough_im, cmap='gray')
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Material Roughness Map')
        plt.show()
    
        plt.figure(figsize=(8,8))
        plt.imshow(mat_displ_im, cmap='gray')
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Material Displacement Map')
        plt.show()
    
            
        ## Sort Damage by Cracks and Spalling. Create Cracks
        spall_mask, cracks, cracks_line = self.drawCracks(cropped, conc_mat, cnts, scale, yesCracks)
    
        plt.figure(figsize=(8,8))
        #plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Cracks Displacement')
        plt.imshow(cracks, cmap='gray')
        
        plt.figure(figsize=(8,8))
        #plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Cracks Line Displacement')
        plt.imshow(cracks_line, cmap='gray')
    
        plt.figure(figsize=(8,8))
        #plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Spalling Mask')
        plt.imshow(spall_mask, cmap='gray')
        
        
        ## Add Spalling
        # Create empty images
        rgbImage = Image.new('L', (w, h))
        displ_img = Image.new('L', (w, h))
        bump_img = Image.new('L', (w, h))
        metal_img = Image.new('L', (w, h))
    
        # If there is spalling, then add it
        if (np.any(spall_mask)):
            rgbImage, displ_img, bump_img, metal_img = self.drawSpalling(conc_mat, spall_mask, actW, rebar, scale, yesCracks, long_rebar_fn, stir_rebar_fn)
    
        # Otherwise, there is no spalling, so rgbImage = material 
        else:
            rgbImage = Image.fromarray(np.uint8(conc_mat))
            rgbImage = rgbImage.convert("RGB")
    
        # Show images
        plt.figure(figsize=(8,8))
        #plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('RGB Image')
        plt.imshow(rgbImage, cmap='gray')
    
        plt.figure(figsize=(8,8))
        #plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Displ. Image')
        plt.imshow(displ_img, cmap='gray')
    
        plt.figure(figsize=(8,8))
        #plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Bump Image')
        plt.imshow(bump_img, cmap='gray')
    
        plt.figure(figsize=(8,8))
        #plt.figure()
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Metal Image')
        plt.imshow(metal_img, cmap='gray')
        
        
        ## Add Crack Lines to RGB
        #cracks_line_img = Image.fromarray(np.uint8(cracks_line))
        #cracks_line_img = cracks_line_img.convert("L")
        #rgbImage.paste(ImageChops.invert(cracks_line_img), (0, 0), cracks_line_img)
        
        ## Add Cracks2 to RGB
        cracks_img = Image.fromarray(np.uint8(cracks))
        cracks_img = cracks_img.convert("L")
        rgbImage = rgbImage.copy()
        rgbImage.paste(ImageChops.invert(cracks_img), (0, 0), cracks_img)
        
        plt.figure(figsize=(8,8))
        plt.imshow(rgbImage)
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Crack Lines on RGB')
        plt.show()
        
        
        ## Create Normal Map
        # Add larger cracks to rgb for the normal map
        #cracks_img = Image.fromarray(cracks)  # Convert cracks image to Image type
        #cracks_img = cracks_img.convert("L")  # Convert cracks image to black and white
        #rgbImage2 = rgbImage.copy()
        #rgbImage2 = ImageChops.multiply(rgbImage2, cracks_img)
        
        #plt.figure(figsize=(8,8))
        #plt.imshow(rgbImage2)
        #plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        #plt.title('Cracks on RGB')
        #plt.show()
        
        # Convert RGB Image2 to Grayscale
        rgb_im_arr2 = np.array(rgbImage)
        height_im_arr2 = cv2.cvtColor(rgb_im_arr2, cv2.COLOR_BGR2GRAY)
        
        plt.figure(figsize=(8,8))
        plt.imshow(height_im_arr2, cmap='gray')
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('RGB with Cracks for Normal Map')
        plt.show()
        
        mat_normal_im = self.ut.createNormalMap(height_im_arr2)
    
        print(np.amax(mat_normal_im))
        print(np.amin(mat_normal_im))
    
        plt.figure(figsize=(8,8))
        plt.imshow(mat_normal_im)
        plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
        plt.title('Normal Map')
        plt.show()
        
        
        ## Add some blur to the spalling mask
        sigma = scale * 2
        spall_mask = cv2.GaussianBlur(spall_mask, (0, 0), sigma, sigma)
        
        ## Save Images
        self.saveImages(cracks_img, rgbImage, displ_img, bump_img, metal_img, mat_normal_im, mat_rough_im, mat_displ_im, spall_mask)
    
    
    
    
    
